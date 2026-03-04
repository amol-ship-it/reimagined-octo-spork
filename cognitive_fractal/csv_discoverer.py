"""
CSVFunctionDiscoverer — discovers the generating function from a CSV file.

Wraps the core fractal system (SymbolicEngine, InvertedCompositionFractal,
SequencePredictor) with a CSV-aware interface that:
  1. Reads a CSV of (x, y) pairs or a single y column
  2. Streams the data through the engine to build candidate hypotheses
  3. Runs a direct-fit pass over the full dataset for each function family
  4. Reports the best discovered formula with validation metrics

Usage:
    discoverer = CSVFunctionDiscoverer("data.csv")
    result = discoverer.run()
    print(result.formula)        # e.g. "sin(2.0000*x + 3.0000)"
    print(result.rmse)           # e.g. 1.2e-14
    print(result.evaluate(100))  # predict y at x=100
"""

import csv
import copy
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Union

from .symbolic_engine import SymbolicEngine
from .memory import Memory
from .function_fractal import (
    FunctionFractal,
    ConstantFractal,
    LinearFractal,
    QuadraticFractal,
    PolynomialFractal,
    SinFractal,
    CosFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)
from .base_functions import BASE_FUNCTIONS
from .inverted_composition import InvertedCompositionFractal
from .mixed_inner import MixedInnerFractal
from .pattern_store import PatternStore


# ================================================================
# Result container
# ================================================================

@dataclass
class DiscoveryResult:
    """The outcome of a function discovery run."""

    formula: str
    """Symbolic representation of the discovered function."""

    rmse: float
    """Root mean squared error on the input data."""

    r_squared: float
    """Coefficient of determination (1.0 = perfect fit)."""

    fractal: Optional[FunctionFractal]
    """The winning FunctionFractal object (can evaluate / inspect)."""

    runner_up: Optional[str] = None
    """Formula of the second-best candidate, if any."""

    runner_up_rmse: Optional[float] = None
    """RMSE of the second-best candidate."""

    candidates_tried: int = 0
    """Total number of candidate hypotheses evaluated."""

    n_points: int = 0
    """Number of data points used."""

    x: np.ndarray = field(default_factory=lambda: np.array([]))
    """Input x values from the CSV."""

    y: np.ndarray = field(default_factory=lambda: np.array([]))
    """Input y values from the CSV."""

    def evaluate(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the discovered function at given x value(s)."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.fractal is None:
            return np.full_like(x_arr, np.nan)
        return self.fractal.evaluate(x_arr)

    def predict(self, n_future: int) -> np.ndarray:
        """Predict the next n_future values beyond the input data."""
        if self.fractal is None or len(self.x) == 0:
            return np.zeros(n_future)
        x_max = self.x[-1]
        step = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        x_future = np.arange(x_max + step, x_max + step * (n_future + 0.5), step)
        return self.fractal.evaluate(x_future[:n_future])

    def summary(self) -> str:
        """Human-readable summary of the discovery."""
        lines = [
            f"Discovered function:  y = {self.formula}",
            f"RMSE:                 {self.rmse:.10g}",
            f"R-squared:            {self.r_squared:.10g}",
            f"Data points:          {self.n_points}",
            f"Candidates evaluated: {self.candidates_tried}",
        ]
        if self.runner_up:
            lines.append(f"Runner-up:            y = {self.runner_up}  (RMSE={self.runner_up_rmse:.6g})")
        return "\n".join(lines)


# ================================================================
# Main class
# ================================================================

class CSVFunctionDiscoverer:
    """Discovers the generating function from CSV data.

    The discovery pipeline:
      1. Parse CSV into (x, y) arrays
      2. Stream data through SymbolicEngine (builds hypotheses incrementally)
      3. Direct-fit pass: try every function family on the full dataset
      4. Rank all candidates by RMSE and return the best

    Args:
        source: Path to CSV file, or a list/array of y-values, or a
            tuple of (x_array, y_array).
        max_degree: Maximum polynomial degree to try inside compositions.
            Default 3 means the system will try F(ax+b), F(ax²+bx+c),
            and F(ax³+bx²+cx+d) for each base function F.
        window_size: Sliding window for the streaming engine.
        passes: Number of times to stream the data through the engine.
            More passes give the fitness-based selection more signal.
        verbose: Print progress to stdout.
        db: Path to the pattern store file. Patterns from previous runs
            are loaded at startup and new discoveries are saved after
            each run. Set to None to disable persistence.
    """

    def __init__(
        self,
        source: Union[str, Path, list, tuple, np.ndarray],
        max_degree: int = 3,
        window_size: int = 50,
        passes: int = 2,
        verbose: bool = True,
        db: Optional[str] = "default",
    ):
        self.max_degree = max_degree
        self.window_size = window_size
        self.passes = passes
        self.verbose = verbose

        # Persistence
        if db == "default":
            self._store = PatternStore()
        elif db is None:
            self._store = None
        else:
            self._store = PatternStore(path=db)

        # Source hint for tagging saved patterns
        self._source_hint = str(source) if isinstance(source, (str, Path)) else ""

        self._x, self._y = self._load(source)
        self._memory = Memory()

    # ----------------------------------------------------------
    # CSV parsing
    # ----------------------------------------------------------

    @staticmethod
    def _load(
        source: Union[str, Path, list, tuple, np.ndarray],
    ) -> tuple:
        """Parse the input source into (x, y) numpy arrays.

        Handles:
          - CSV file path (auto-detects header, 1 or 2 columns)
          - (x_array, y_array) tuple
          - Plain list/array of y-values (x = 0, 1, 2, ...)
        """
        if isinstance(source, (str, Path)):
            return CSVFunctionDiscoverer._load_csv(str(source))
        if isinstance(source, tuple) and len(source) == 2:
            x = np.asarray(source[0], dtype=float)
            y = np.asarray(source[1], dtype=float)
            return x, y
        # Assume a flat sequence of y-values
        y = np.asarray(source, dtype=float)
        x = np.arange(len(y), dtype=float)
        return x, y

    @staticmethod
    def _load_csv(path: str) -> tuple:
        """Read a CSV file into (x, y) arrays.

        Auto-detects:
          - Whether the first row is a header (non-numeric)
          - Whether there are 1 column (y only) or 2+ columns (x, y)
        """
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"CSV file is empty: {path}")

        # Detect header: if first row contains non-numeric values
        has_header = False
        try:
            [float(v) for v in rows[0]]
        except ValueError:
            has_header = True

        data_rows = rows[1:] if has_header else rows
        if not data_rows:
            raise ValueError(f"CSV file has no data rows: {path}")

        n_cols = len(data_rows[0])
        if n_cols == 1:
            y = np.array([float(r[0]) for r in data_rows])
            x = np.arange(len(y), dtype=float)
        else:
            x = np.array([float(r[0]) for r in data_rows])
            y = np.array([float(r[1]) for r in data_rows])

        return x, y

    # ----------------------------------------------------------
    # Main discovery pipeline
    # ----------------------------------------------------------

    # An R² above this threshold means we already have a near-perfect
    # formula — no need to redo the expensive search.
    _CACHE_R2_THRESHOLD = 0.9999

    # R² threshold above which we skip the streaming phase
    # (direct-fit already found a near-perfect match).
    _SKIP_STREAM_R2 = 0.999

    def run(self) -> DiscoveryResult:
        """Run the full discovery pipeline. Returns a DiscoveryResult."""
        x, y = self._x, self._y
        n = len(y)

        if self.verbose:
            print(f"Data: {n} points, x in [{x[0]:.4g}, {x[-1]:.4g}], "
                  f"y in [{y.min():.6g}, {y.max():.6g}]")

        # Phase 0: load prior patterns from disk
        prior_candidates: List[FunctionFractal] = []
        if self._store is not None:
            prior_candidates = self._store.load()
            if self.verbose and prior_candidates:
                print(f"Loaded {len(prior_candidates)} patterns from previous runs")

        # Phase 0b: validate prior patterns on the current data.
        # If a loaded pattern already gives a near-perfect fit, skip
        # the expensive search phases entirely.
        cache_hit = self._check_cache_hit(prior_candidates, x, y)
        if cache_hit is not None:
            if self.verbose:
                print(f"Cache hit — prior pattern matches current data "
                      f"(R²={cache_hit.r_squared:.10g})")
                print()
                print(cache_hit.summary())
            return cache_hit

        # Phase 1: direct-fit all function families (tiered, early exit).
        # This runs first because it's fast for simple functions and
        # can skip the expensive streaming phase entirely.
        if self.verbose:
            print("Phase 1: Direct-fit pass over full dataset...")
        direct_candidates = self._direct_fit_phase(x, y)

        # Check if direct-fit found a near-perfect match (R²-based).
        direct_best_rmse = float("inf")
        for c in direct_candidates:
            try:
                yp = c.evaluate(x)
                if np.all(np.isfinite(yp)):
                    r = float(np.sqrt(np.mean((y - yp) ** 2)))
                    if r < direct_best_rmse:
                        direct_best_rmse = r
            except Exception:
                pass

        y_var = float(np.var(y))
        if y_var > 0 and direct_best_rmse < float("inf"):
            direct_r2 = 1.0 - (direct_best_rmse ** 2) / y_var
        else:
            direct_r2 = 0.0

        # Phase 2: stream through the engine (only if direct-fit
        # didn't find a near-perfect match).
        engine = None
        if direct_r2 < self._SKIP_STREAM_R2:
            if self.verbose:
                print(f"Phase 2: Streaming {self.passes} pass(es) through engine...")
            engine = self._stream_phase(x, y)
        elif self.verbose:
            print(f"  Direct-fit R²={direct_r2:.8f} — skipping streaming")

        # Re-fit prior patterns on the current data so they compete fairly
        for c in prior_candidates:
            try:
                c.fit(x, y)
            except Exception:
                pass

        # Phase 3: collect all candidates, rank by RMSE
        if self.verbose:
            print("Phase 3: Ranking all candidates...")
        result = self._rank_candidates(
            engine, direct_candidates + prior_candidates, x, y
        )

        # Phase 4: save discoveries to disk for future runs
        if self._store is not None:
            engine_candidates = list(engine.candidates) if engine else []
            all_candidates = engine_candidates + direct_candidates + prior_candidates
            n_saved = self._store.save(
                all_candidates, x, y,
                source_hint=self._source_hint,
            )
            if self.verbose:
                print(f"Saved {n_saved} patterns to {self._store.path}")

        if self.verbose:
            print()
            print(result.summary())

        return result

    # ----------------------------------------------------------
    # Cache-hit check
    # ----------------------------------------------------------

    def _check_cache_hit(
        self,
        prior_candidates: List[FunctionFractal],
        x: np.ndarray,
        y: np.ndarray,
    ) -> Optional[DiscoveryResult]:
        """Check if any loaded pattern already fits the current data.

        Returns a DiscoveryResult if a near-perfect match is found,
        None otherwise.  This avoids re-running the entire expensive
        discovery pipeline on subsequent runs with the same data.
        """
        if not prior_candidates:
            return None

        y_var = float(np.var(y))
        if y_var == 0:
            return None

        scored: List[tuple] = []
        for c in prior_candidates:
            try:
                y_pred = c.evaluate(x)
                if not np.all(np.isfinite(y_pred)):
                    continue
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
                ss_res = rmse ** 2 * len(y)
                ss_tot = y_var * len(y)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                scored.append((rmse, r2, c))
            except Exception:
                continue

        if not scored:
            return None

        scored.sort(key=lambda t: t[0])
        best_rmse, best_r2, best_frac = scored[0]

        if best_r2 < self._CACHE_R2_THRESHOLD:
            return None

        # We have a cache hit — build the result
        best_formula = best_frac.symbolic_repr()

        # Runner-up (different formula)
        runner_up_formula = None
        runner_up_rmse = None
        for rmse, _, c in scored[1:]:
            f = c.symbolic_repr()
            if f != best_formula:
                runner_up_formula = f
                runner_up_rmse = rmse
                break

        return DiscoveryResult(
            formula=best_formula,
            rmse=best_rmse,
            r_squared=best_r2,
            fractal=best_frac,
            runner_up=runner_up_formula,
            runner_up_rmse=runner_up_rmse,
            candidates_tried=len(scored),
            n_points=len(y),
            x=x,
            y=y,
        )

    # ----------------------------------------------------------
    # Phase 1: streaming
    # ----------------------------------------------------------

    def _stream_phase(self, x: np.ndarray, y: np.ndarray) -> SymbolicEngine:
        """Stream data through the SymbolicEngine for self.passes passes."""
        engine = SymbolicEngine(
            window_size=self.window_size,
            predict_horizon=1,
            max_candidates=30 + len(BASE_FUNCTIONS) * self.max_degree,
            composition_interval=max(40, len(y) // 5),
            composition_threshold=0.1,
            nested_composition_interval=max(80, len(y) // 3),
            memory=self._memory,
        )

        for pass_num in range(self.passes):
            for i, yi in enumerate(y):
                engine.step(float(yi))

            if self.verbose:
                formula, fitness, _ = engine.get_best()
                print(f"  Pass {pass_num + 1}: best = {formula}  "
                      f"(fitness={fitness:.4f}, candidates={len(engine.candidates)})")

        return engine

    # ----------------------------------------------------------
    # Phase 2: direct fit (tiered with signal prefilter)
    # ----------------------------------------------------------

    # Early-exit RMSE thresholds per tier.
    _TIER_THRESHOLDS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-4, 1e-4]

    # Redundant two-level pairs to skip (outer, inner).
    _SKIP_TWO_LEVEL = {
        ("exp", "log"), ("exp", "log2"), ("exp", "log10"),
        ("log", "exp"), ("log2", "exp"), ("log10", "exp"),
        ("sqrt", "sqrt"),
        # log2/log10 as outer are scalar multiples of log
    }

    @staticmethod
    def _compute_signal_profile(
        x: np.ndarray, y: np.ndarray,
    ) -> Dict[str, bool]:
        """Cheap O(n) signal features for pruning irrelevant families."""
        y_min, y_max = float(y.min()), float(y.max())
        diffs = np.diff(y)
        return {
            "y_bounded": y_max <= 1.05 and y_min >= -1.05,
            "is_monotonic": bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12)),
            "has_negatives": bool(y_min < -1e-12),
            "all_positive": bool(y_min > 1e-12),
        }

    def _outer_allowed(
        self, func_name: str, profile: Dict[str, bool],
    ) -> bool:
        """Check if an outer function is plausible given the signal."""
        if not profile["y_bounded"]:
            if func_name in ("sin", "cos", "tanh"):
                return False
        if profile["has_negatives"]:
            if func_name in ("sqrt", "exp"):
                return False
        return True

    def _outer_allowed_mixed(
        self, func_name: str, profile: Dict[str, bool],
    ) -> bool:
        """Extra filter for mixed-inner (periodic outers need non-monotonic)."""
        if not self._outer_allowed(func_name, profile):
            return False
        if profile["is_monotonic"] and func_name in ("sin", "cos"):
            return False
        return True

    # Number of threads for parallel fitting (0 = sequential).
    _N_WORKERS = min(4, os.cpu_count() or 1)

    @staticmethod
    def _fit_worker(
        fractal: FunctionFractal,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Fit a single candidate in a worker thread.

        Returns (fractal, rmse).  NumPy releases the GIL during
        polyfit/lstsq, so threads give real parallelism.
        """
        try:
            fractal.fit(x, y)
            yp = fractal.evaluate(x)
            if np.all(np.isfinite(yp)):
                r = float(np.sqrt(np.mean((y - yp) ** 2)))
                return fractal, r
        except Exception:
            pass
        return fractal, float("inf")

    def _fit_batch(
        self,
        fractals: List[FunctionFractal],
        x: np.ndarray,
        y: np.ndarray,
        candidates: List[FunctionFractal],
        best_rmse: float,
    ) -> float:
        """Fit a batch of candidates (parallel when >= 2 items).

        Appends all fitted fractals to candidates and returns the
        updated best_rmse.
        """
        if not fractals:
            return best_rmse

        if len(fractals) < 4 or self._N_WORKERS < 2:
            # Sequential for small batches
            for f in fractals:
                try:
                    f.fit(x, y)
                    candidates.append(f)
                    yp = f.evaluate(x)
                    if np.all(np.isfinite(yp)):
                        r = float(np.sqrt(np.mean((y - yp) ** 2)))
                        if r < best_rmse:
                            best_rmse = r
                except Exception:
                    pass
            return best_rmse

        with ThreadPoolExecutor(max_workers=self._N_WORKERS) as pool:
            futures = [
                pool.submit(self._fit_worker, f, x, y) for f in fractals
            ]
            for fut in as_completed(futures):
                frac, rmse = fut.result()
                candidates.append(frac)
                if rmse < best_rmse:
                    best_rmse = rmse

        return best_rmse

    def _direct_fit_phase(
        self, x: np.ndarray, y: np.ndarray,
    ) -> List[FunctionFractal]:
        """Tiered direct-fit with signal prefilter and early exit.

        Tries cheap families first; skips expensive tiers if a
        near-perfect fit is already found.  Within each tier, fits
        degree-by-degree (1 → 2 → 3) so a cheap perfect fit at
        low degree avoids expensive high-degree enumeration.
        Heavy tiers (2, 3, 4) use parallel fitting.
        """
        candidates: List[FunctionFractal] = []
        profile = self._compute_signal_profile(x, y)
        best_rmse = float("inf")

        def _fit_one(fractal) -> float:
            """Fit a single candidate (sequential)."""
            nonlocal best_rmse
            try:
                fractal.fit(x, y)
                candidates.append(fractal)
                yp = fractal.evaluate(x)
                if np.all(np.isfinite(yp)):
                    r = float(np.sqrt(np.mean((y - yp) ** 2)))
                    if r < best_rmse:
                        best_rmse = r
            except Exception:
                pass
            return best_rmse

        # ---- Tier 0: simple types (< 10ms, sequential) ----
        for factory in [
            ConstantFractal, LinearFractal, QuadraticFractal,
            lambda: PolynomialFractal(degree=3),
            lambda: PolynomialFractal(degree=4),
            lambda: PolynomialFractal(degree=5),
            SinFractal, CosFractal, ExponentialFractal, LogFractal,
        ]:
            _fit_one(factory())
        if best_rmse < self._TIER_THRESHOLDS[0]:
            if self.verbose:
                print(f"  Direct-fit: {len(candidates)} candidates (Tier 0 exit)")
            return candidates

        # ---- Tier 1: non-branching inverted F(poly(x)) (fast, sequential) ----
        nb_funcs = [
            (name, bf)
            for name, bf in BASE_FUNCTIONS.items()
            if not bf.has_branches and self._outer_allowed(name, profile)
        ]
        for deg in range(1, self.max_degree + 1):
            for _name, bf in nb_funcs:
                _fit_one(InvertedCompositionFractal(bf, degree=deg))
            if best_rmse < self._TIER_THRESHOLDS[1]:
                break
        if best_rmse < self._TIER_THRESHOLDS[1]:
            if self.verbose:
                print(f"  Direct-fit: {len(candidates)} candidates (Tier 1 exit)")
            return candidates

        # ---- Tier 2: branching inverted F(poly(x)) (parallel per degree) ----
        br_funcs = [
            (name, bf)
            for name, bf in BASE_FUNCTIONS.items()
            if bf.has_branches and self._outer_allowed(name, profile)
        ]
        for deg in range(1, self.max_degree + 1):
            batch = [InvertedCompositionFractal(bf, degree=deg) for _name, bf in br_funcs]
            best_rmse = self._fit_batch(batch, x, y, candidates, best_rmse)
            if best_rmse < self._TIER_THRESHOLDS[2]:
                break
        if best_rmse < self._TIER_THRESHOLDS[2]:
            if self.verbose:
                print(f"  Direct-fit: {len(candidates)} candidates (Tier 2 exit)")
            return candidates

        # ---- Tier 2b: scaled bounded compositions y = a*F(poly(x)) + b ----
        # Catches functions like 0.5*sin(2x) where the inversion approach
        # fails without normalisation (arcsin(0.5*sin(2x)) ≠ polynomial).
        y_max, y_min = float(y.max()), float(y.min())
        est_a = (y_max - y_min) / 2.0
        est_b = (y_max + y_min) / 2.0
        # Only try when scaling is meaningfully different from identity
        if est_a > 0.01 and (abs(est_a - 1.0) > 0.05 or abs(est_b) > 0.05):
            bounded_outers = [
                bf for name, bf in BASE_FUNCTIONS.items()
                if name in ("sin", "cos", "tanh")
            ]
            for deg in range(1, self.max_degree + 1):
                batch: List[FunctionFractal] = []
                for bf in bounded_outers:
                    batch.append(InvertedCompositionFractal(
                        bf, degree=deg, outer_scale=est_a, outer_offset=est_b,
                    ))
                    # Also try negative amplitude (e.g. y = -0.5*sin(2x))
                    batch.append(InvertedCompositionFractal(
                        bf, degree=deg, outer_scale=-est_a, outer_offset=est_b,
                    ))
                best_rmse = self._fit_batch(batch, x, y, candidates, best_rmse)
                if best_rmse < self._TIER_THRESHOLDS[2]:
                    break
            if best_rmse < self._TIER_THRESHOLDS[2]:
                if self.verbose:
                    print(f"  Direct-fit: {len(candidates)} candidates (Tier 2b exit)")
                return candidates

        # ---- Tier 3: two-level F(G(poly(x))) (parallel) ----
        two_level_batch: List[FunctionFractal] = []
        for oname, ofunc in BASE_FUNCTIONS.items():
            if not self._outer_allowed(oname, profile):
                continue
            if oname in ("log2", "log10"):
                continue
            for iname, ifunc in BASE_FUNCTIONS.items():
                if (oname, iname) in self._SKIP_TWO_LEVEL:
                    continue
                two_level_batch.append(
                    InvertedCompositionFractal([ofunc, ifunc], degree=1)
                )
        best_rmse = self._fit_batch(two_level_batch, x, y, candidates, best_rmse)
        if best_rmse < self._TIER_THRESHOLDS[3]:
            if self.verbose:
                print(f"  Direct-fit: {len(candidates)} candidates (Tier 3 exit)")
            return candidates

        # ---- Tier 4: mixed inner F(poly(x) + G(inner_poly(x))) (parallel per degree) ----
        mi_funcs = [
            (name, bf)
            for name, bf in BASE_FUNCTIONS.items()
            if self._outer_allowed_mixed(name, profile)
        ]
        for deg in range(1, self.max_degree + 1):
            batch = [MixedInnerFractal(bf, poly_degree=deg) for _name, bf in mi_funcs]
            best_rmse = self._fit_batch(batch, x, y, candidates, best_rmse)
            if best_rmse < self._TIER_THRESHOLDS[4]:
                break
        if best_rmse < self._TIER_THRESHOLDS[4]:
            if self.verbose:
                print(f"  Direct-fit: {len(candidates)} candidates (Tier 4 exit)")
            return candidates

        # ---- Tier 5: 3-level compositions + multiplicative ----
        _THREE_LEVEL = [
            ("exp", "sin"), ("exp", "cos"),
            ("log", "sin"), ("log", "cos"),
            ("sin", "exp"), ("cos", "exp"),
            ("sin", "log"), ("cos", "log"),
            ("tanh", "sin"), ("tanh", "cos"),
        ]
        three_level_batch: List[FunctionFractal] = []
        for outer_name, mid_name in _THREE_LEVEL:
            if not self._outer_allowed(outer_name, profile):
                continue
            ofunc = BASE_FUNCTIONS.get(outer_name)
            mfunc = BASE_FUNCTIONS.get(mid_name)
            if ofunc is None or mfunc is None:
                continue
            for iname in ("sin", "cos", "exp", "log"):
                ifunc = BASE_FUNCTIONS.get(iname)
                if ifunc is None:
                    continue
                three_level_batch.append(
                    InvertedCompositionFractal([ofunc, mfunc, ifunc], degree=1)
                )
        best_rmse = self._fit_batch(three_level_batch, x, y, candidates, best_rmse)

        # Multiplicative: F(x) * G(x) for already-fitted leaf candidates
        leaf = [c for c in candidates if not isinstance(c, ComposedFunctionFractal)]
        for i, c1 in enumerate(leaf):
            for c2 in leaf[i + 1:]:
                try:
                    y1 = c1.evaluate(x)
                    y2 = c2.evaluate(x)
                    prod = y1 * y2
                    if not np.all(np.isfinite(prod)):
                        continue
                    r = float(np.sqrt(np.mean((y - prod) ** 2)))
                    if r < 0.1:
                        composed = ComposedFunctionFractal(
                            copy.deepcopy(c1), copy.deepcopy(c2), "multiply"
                        )
                        candidates.append(composed)
                except Exception:
                    continue

        if self.verbose:
            print(f"  Direct-fit: {len(candidates)} candidates evaluated")

        return candidates

    # ----------------------------------------------------------
    # Phase 3: ranking
    # ----------------------------------------------------------

    def _rank_candidates(
        self,
        engine: Optional[SymbolicEngine],
        direct_candidates: List[FunctionFractal],
        x: np.ndarray,
        y: np.ndarray,
    ) -> DiscoveryResult:
        """Rank all candidates by RMSE on the full dataset."""
        all_candidates: List[FunctionFractal] = []
        if engine is not None:
            all_candidates.extend(engine.candidates)
        all_candidates.extend(direct_candidates)

        # De-duplicate by id
        seen_ids = set()
        unique = []
        for c in all_candidates:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique.append(c)

        # Score each candidate
        scored: List[tuple] = []  # (rmse, formula, candidate)
        y_var = float(np.var(y))

        for c in unique:
            try:
                y_pred = c.evaluate(x)
                if not np.all(np.isfinite(y_pred)):
                    continue
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
                scored.append((rmse, c.symbolic_repr(), c))
            except Exception:
                continue

        if not scored:
            return DiscoveryResult(
                formula="none",
                rmse=float("inf"),
                r_squared=0.0,
                fractal=None,
                candidates_tried=len(unique),
                n_points=len(y),
                x=x,
                y=y,
            )

        scored.sort(key=lambda t: t[0])
        best_rmse, best_formula, best_frac = scored[0]

        # R-squared
        ss_res = best_rmse ** 2 * len(y)
        ss_tot = y_var * len(y) if y_var > 0 else 1.0
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Runner-up (different formula)
        runner_up_formula = None
        runner_up_rmse = None
        for rmse, formula, _ in scored[1:]:
            if formula != best_formula:
                runner_up_formula = formula
                runner_up_rmse = rmse
                break

        return DiscoveryResult(
            formula=best_formula,
            rmse=best_rmse,
            r_squared=r_squared,
            fractal=best_frac,
            runner_up=runner_up_formula,
            runner_up_rmse=runner_up_rmse,
            candidates_tried=len(unique),
            n_points=len(y),
            x=x,
            y=y,
        )
