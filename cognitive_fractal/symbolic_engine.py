"""
The SymbolicEngine — discovers generating functions by composing
mathematical primitives.

Unlike FractalEngine (which routes each input to ONE best-matching
fractal), this engine updates ALL candidates every step. Each candidate
is a mathematical function hypothesis with learnable coefficients.

The engine's job:
  1. Maintain a population of candidate function types
  2. Fit all candidates to the data window each step
  3. Track fitness — which function best explains the data?
  4. Compose functions via residual analysis when single types plateau
  5. Prune weak candidates to keep the population bounded
  6. Report the discovered formula in symbolic form

This IS the Four Pillars:
  - Feedback Loops:  predict → observe → update metrics (every step)
  - Approximability: coefficients converge via polyfit / gradient descent
  - Composability:   residual analysis decomposes y = f(x) + g(x)
  - Exploration:     population of diverse candidates explores function space
"""

import copy
import time
import numpy as np
from typing import List, Tuple, Optional, Dict

from .memory import Memory
from .types import Signal, Feedback
from .function_fractal import (
    FunctionFractal,
    ConstantFractal,
    LinearFractal,
    QuadraticFractal,
    PolynomialFractal,
    SinFractal,
    CosFractal,
    GradientSinFractal,
    GradientCosFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)
from .nested_fractal import NestedComposedFractal


class SymbolicEngine:
    """Discovers generating functions by composing mathematical primitives."""

    def __init__(
        self,
        window_size: int = 20,
        predict_horizon: int = 1,
        max_candidates: int = 20,
        composition_interval: int = 50,
        composition_threshold: float = 0.3,
        nested_composition_interval: int = 100,
        memory: Optional[Memory] = None,
    ):
        self.window_size = window_size
        self.predict_horizon = predict_horizon
        self.max_candidates = max_candidates
        self.composition_interval = composition_interval
        self.nested_composition_interval = nested_composition_interval
        self.composition_threshold = composition_threshold

        self.memory = memory if memory is not None else Memory()
        self.candidates: List[FunctionFractal] = []

        # History
        self._history_x: List[float] = []
        self._history_y: List[float] = []
        self._step_count: int = 0

        # Per-candidate tracking
        self._last_predictions: Dict[str, np.ndarray] = {}
        self._best_candidate: Optional[FunctionFractal] = None

    def _initialize_candidates(self) -> None:
        """Create the default population of function hypotheses.

        If shared memory contains learned patterns, seed matching
        candidates with those coefficients (transfer learning).
        Also loads high-fitness composed templates from the library.
        """
        candidates = [
            ConstantFractal(),
            LinearFractal(),
            QuadraticFractal(),
            PolynomialFractal(degree=3),
            PolynomialFractal(degree=4),
            SinFractal(),
            CosFractal(),
            GradientSinFractal(),
            GradientCosFractal(),
            ExponentialFractal(),
            LogFractal(),
        ]
        for c in candidates:
            c.predict_horizon = self.predict_horizon
            self._seed_from_library(c)
            self.candidates.append(c)
            self.memory.store(c)

        # Load high-fitness composed patterns from prior streams
        self._load_composed_templates()

    def _seed_from_library(self, candidate: FunctionFractal) -> bool:
        """Try to seed a candidate's coefficients from a stored pattern.

        Looks for high-fitness patterns of the same function type
        in shared memory and copies their learned coefficients.

        Returns True if seeding occurred.
        """
        best_match = None
        best_fitness = 0.1  # Minimum fitness threshold
        for frac in self.memory.hot.values():
            if (
                isinstance(frac, FunctionFractal)
                and type(frac).__name__ == type(candidate).__name__
                and frac.metrics.fitness() > best_fitness
                and frac.id != candidate.id
            ):
                best_fitness = frac.metrics.fitness()
                best_match = frac
        if best_match is not None:
            candidate.seed_from(best_match)
            return True
        return False

    def step(self, y_value: float) -> Tuple[float, dict]:
        """Process one value from the stream.

        Returns:
            (prediction: float, diagnostics: dict)
        """
        self._step_count += 1
        t = self._step_count - 1  # 0-indexed time
        self._history_x.append(float(t))
        self._history_y.append(y_value)

        # Initialize on first call
        if self._step_count == 1:
            self._initialize_candidates()
            return 0.0, {
                "step": 1,
                "best_name": "none",
                "best_fitness": 0.0,
                "best_formula": "none",
                "prediction_error": 0.0,
                "num_candidates": len(self.candidates),
            }

        # Need at least a few points for fitting
        if len(self._history_y) < 3:
            return 0.0, {
                "step": self._step_count,
                "best_name": "none",
                "best_fitness": 0.0,
                "best_formula": "none",
                "prediction_error": 0.0,
                "num_candidates": len(self.candidates),
            }

        # Extract window
        start = max(0, len(self._history_x) - self.window_size)
        x_window = np.array(self._history_x[start:])
        y_window = np.array(self._history_y[start:])

        # --- Feedback phase: evaluate previous predictions ---
        feedback_errors = {}
        now = time.time()
        for candidate in self.candidates:
            if candidate.id in self._last_predictions:
                actual = np.array([y_value])
                feedback = Feedback(actual=actual, reward=0.0, timestamp=now)
                err = candidate.learn(feedback)
                feedback_errors[candidate.id] = err

        # --- Forward phase: fit all candidates, collect predictions ---
        x_offset = int(x_window[0])
        for candidate in self.candidates:
            candidate._x_offset = x_offset
            candidate.predict_horizon = self.predict_horizon
            signal = Signal(data=y_window, timestamp=now)
            output, novelty = candidate.process(signal)
            self._last_predictions[candidate.id] = output.data

        # --- Update signatures in memory for pattern library search ---
        for candidate in self.candidates:
            if hasattr(candidate, '_signature') and np.any(candidate._signature != 0):
                self.memory.update_signature(candidate.id, candidate._signature)

        # --- Select best candidate ---
        self._best_candidate = max(
            self.candidates, key=lambda c: c.metrics.fitness()
        )
        best_prediction = self._last_predictions[self._best_candidate.id]

        # --- Periodically attempt composition ---
        if (
            self._step_count % self.composition_interval == 0
            and self._step_count > self.window_size
        ):
            self._attempt_composition(x_window, y_window)

        # --- Periodically attempt nested composition g(f(x)) ---
        if (
            self._step_count % self.nested_composition_interval == 0
            and self._step_count > self.window_size * 2
        ):
            self._attempt_nested_composition(x_window, y_window)

        # --- Prune if over capacity ---
        if len(self.candidates) > self.max_candidates:
            self._prune()

        # Build diagnostics
        best_err = feedback_errors.get(self._best_candidate.id, 0.0)
        diagnostics = {
            "step": self._step_count,
            "best_name": self._best_candidate.func_name,
            "best_fitness": self._best_candidate.metrics.fitness(),
            "best_formula": self._best_candidate.symbolic_repr(),
            "prediction_error": best_err,
            "num_candidates": len(self.candidates),
        }

        return float(best_prediction[0]), diagnostics

    def _compute_residual_signature(
        self, residuals: np.ndarray
    ) -> np.ndarray:
        """Compute a signature vector from residual data for library lookup.

        Resamples residuals to the canonical signature length and normalizes.
        """
        n_points = FunctionFractal.SIGNATURE_POINTS
        if len(residuals) >= n_points:
            indices = np.linspace(0, len(residuals) - 1, n_points, dtype=int)
            sig = residuals[indices].astype(float)
        else:
            sig = np.zeros(n_points)
            sig[: len(residuals)] = residuals
        norm = np.linalg.norm(sig)
        if norm > 1e-8:
            return sig / norm
        return np.zeros(n_points)

    def _attempt_composition(
        self, x: np.ndarray, y: np.ndarray
    ) -> None:
        """Try composing top candidates via residual analysis.

        Enhanced: queries memory for patterns that match the residual
        shape, enabling transfer learning from prior discoveries.

        Additive: y ≈ f(x) + g(x)
          1. Best leaf explains the dominant pattern
          2. Residual = y - best(x)
          3. Fit local candidates AND library matches to the residual
          4. If combined RMSE improves significantly, create composition
        """
        # Only compose from leaf candidates (not already composed)
        leaf_candidates = [
            c for c in self.candidates
            if not isinstance(c, (ComposedFunctionFractal, NestedComposedFractal))
        ]
        if len(leaf_candidates) < 2:
            return

        # Find best leaf
        best_leaf = max(leaf_candidates, key=lambda c: c.metrics.fitness())
        y_pred_best = best_leaf.evaluate(x)
        best_rmse = float(np.sqrt(np.mean((y - y_pred_best) ** 2)))

        # Compute residuals
        residuals = y - y_pred_best

        # Skip if residuals are negligible
        if np.std(residuals) < 1e-6:
            return

        # Build trial sources: local leaves + shallow composed + library matches
        trial_sources = [
            c for c in leaf_candidates if c.id != best_leaf.id
        ]
        # Allow shallow composed patterns (depth <= 1) as trial sources
        for c in self.candidates:
            if (
                isinstance(c, ComposedFunctionFractal)
                and self._composition_depth(c) <= 1
                and c.id != best_leaf.id
            ):
                trial_sources.append(c)

        # Query library for patterns matching the residual shape
        residual_sig = self._compute_residual_signature(residuals)
        library_matches = self.memory.find_similar_by_signature(
            residual_sig, domain="symbolic", top_k=3, min_fitness=0.1
        )
        local_ids = {c.id for c in trial_sources}
        for lib_frac, lib_sim in library_matches:
            if (
                isinstance(lib_frac, FunctionFractal)
                and lib_frac.id != best_leaf.id
                and lib_frac.id not in local_ids
                and (
                    not isinstance(lib_frac, ComposedFunctionFractal)
                    or self._composition_depth(lib_frac) <= 1
                )
            ):
                trial_sources.append(lib_frac)

        # Try fitting each trial source to the residuals
        best_combined_rmse = best_rmse
        best_trial = None

        for candidate in trial_sources:
            trial = copy.deepcopy(candidate)
            trial.fit(x, residuals)
            trial_pred = trial.evaluate(x)
            combined_pred = y_pred_best + trial_pred
            combined_rmse = float(np.sqrt(np.mean((y - combined_pred) ** 2)))

            if combined_rmse < best_combined_rmse * (1 - self.composition_threshold):
                best_combined_rmse = combined_rmse
                best_trial = trial

        # Create composition if a significant improvement was found
        if best_trial is not None:
            child1 = copy.deepcopy(best_leaf)
            child2 = best_trial  # Already a deepcopy
            composed = ComposedFunctionFractal(child1, child2, operation="add")
            composed.predict_horizon = self.predict_horizon
            self.candidates.append(composed)
            self.memory.store(composed)

    def _prune(self) -> None:
        """Remove lowest-fitness candidates, keeping population bounded.

        High-fitness candidates are preserved in memory even after removal
        from the active population, so their patterns remain available
        for future transfer learning. After pruning, restores high-fitness
        patterns from the library to fill empty slots.
        """
        if len(self.candidates) <= 7:
            return

        scored = sorted(
            self.candidates,
            key=lambda c: c.metrics.fitness(),
            reverse=True,
        )

        # Keep top max_candidates
        to_remove = scored[self.max_candidates:]
        for c in to_remove:
            self.candidates.remove(c)
            # Only remove from memory if fitness is very low;
            # high-fitness patterns stay for transfer learning
            if c.metrics.fitness() < 0.05:
                if c.id in self.memory.hot:
                    del self.memory.hot[c.id]
                self.memory._signatures.pop(c.id, None)
            if c.id in self._last_predictions:
                del self._last_predictions[c.id]

        # Refill empty slots from the pattern library
        self._restore_from_library()

    def _attempt_nested_composition(
        self, x: np.ndarray, y: np.ndarray
    ) -> None:
        """Try discovering nested compositions g(f(x)) = y.

        For each candidate inner function f with non-degenerate output:
          For each function type T as potential outer:
            1. Compute z = f.evaluate(x)
            2. Create fresh T, fit T to (z, y)
            3. Measure RMSE of T(f(x)) vs y
            4. If improvement over best single candidate, create
               NestedComposedFractal

        Complexity: O(n_inner * n_types) ≈ 77 trials per call.
        """
        # Only use leaf candidates as potential inner functions
        leaf_candidates = [
            c for c in self.candidates
            if not isinstance(c, (ComposedFunctionFractal, NestedComposedFractal))
        ]
        if len(leaf_candidates) < 2:
            return

        # Current best RMSE as baseline
        best_overall = max(self.candidates, key=lambda c: c.metrics.fitness())
        best_pred = best_overall.evaluate(x)
        best_rmse = float(np.sqrt(np.mean((y - best_pred) ** 2)))

        # Skip if best is already very good
        if best_rmse < 1e-6:
            return

        # Pre-filter inner candidates: must produce non-degenerate output
        viable_inners = []
        for c in leaf_candidates:
            try:
                z = c.evaluate(x)
                if np.all(np.isfinite(z)) and np.std(z) > 1e-8:
                    viable_inners.append(c)
            except Exception:
                continue

        # Function types to try as outer (skip Constant — no point)
        outer_factories = [
            LinearFractal,
            QuadraticFractal,
            lambda: PolynomialFractal(degree=3),
            SinFractal,
            CosFractal,
            ExponentialFractal,
            LogFractal,
        ]

        best_nested_rmse = best_rmse
        best_inner = None
        best_outer = None

        for inner_candidate in viable_inners:
            z = inner_candidate.evaluate(x)
            z_clipped = np.clip(z, -1e6, 1e6)

            if np.std(z_clipped) < 1e-8:
                continue

            for factory in outer_factories:
                # Skip identity-like: linear(linear) = linear
                if (isinstance(inner_candidate, LinearFractal)
                        and factory is LinearFractal):
                    continue

                try:
                    outer_trial = factory()
                    outer_trial.fit(z_clipped, y)
                    nested_pred = outer_trial.evaluate(z_clipped)

                    if not np.all(np.isfinite(nested_pred)):
                        continue

                    nested_rmse = float(
                        np.sqrt(np.mean((y - nested_pred) ** 2))
                    )

                    if nested_rmse < best_nested_rmse * (
                        1 - self.composition_threshold
                    ):
                        best_nested_rmse = nested_rmse
                        best_inner = inner_candidate
                        best_outer = outer_trial
                except Exception:
                    continue

        # Create nested composition if significant improvement found
        if best_inner is not None and best_outer is not None:
            inner_copy = copy.deepcopy(best_inner)
            nested = NestedComposedFractal(inner_copy, best_outer)
            nested.predict_horizon = self.predict_horizon
            self.candidates.append(nested)
            self.memory.store(nested)

    def _restore_from_library(self) -> None:
        """Restore high-fitness patterns from memory to fill empty candidate slots.

        After pruning removes weak candidates, this method refills slots with
        high-fitness patterns from the shared library. This enables patterns
        discovered in prior streams to re-enter the active population.

        Diversity limit: max 2 restored candidates per type to avoid homogeneity.
        """
        if len(self.candidates) >= self.max_candidates:
            return

        active_ids = {c.id for c in self.candidates}

        # Count types already in candidate pool for diversity limit
        type_counts: Dict[str, int] = {}
        for c in self.candidates:
            tname = type(c).__name__
            type_counts[tname] = type_counts.get(tname, 0) + 1

        # Collect high-fitness library patterns not in active pool
        library_candidates = []
        for frac in self.memory.hot.values():
            if (
                isinstance(frac, FunctionFractal)
                and not isinstance(frac, ComposedFunctionFractal)
                and frac.id not in active_ids
                and frac.metrics.fitness() > 0.1
            ):
                library_candidates.append(frac)

        # Sort by fitness descending
        library_candidates.sort(key=lambda c: c.metrics.fitness(), reverse=True)

        for lib_frac in library_candidates:
            if len(self.candidates) >= self.max_candidates:
                break

            tname = type(lib_frac).__name__
            if type_counts.get(tname, 0) >= 2:
                continue

            # Deep copy so the restored candidate is independent
            restored = copy.deepcopy(lib_frac)
            restored.predict_horizon = self.predict_horizon
            self.candidates.append(restored)
            self.memory.store(restored)
            type_counts[tname] = type_counts.get(tname, 0) + 1

    def _load_composed_templates(self) -> None:
        """Load high-fitness composed patterns from the library.

        Called during initialization when shared memory contains composed
        patterns discovered by prior streams. Up to 3 highest-fitness
        composed patterns are added as candidates.
        """
        composed_templates = []
        for frac in self.memory.hot.values():
            if (
                isinstance(frac, ComposedFunctionFractal)
                and frac.metrics.fitness() > 0.15
            ):
                composed_templates.append(frac)

        # Sort by fitness, take top 3
        composed_templates.sort(key=lambda c: c.metrics.fitness(), reverse=True)
        for frac in composed_templates[:3]:
            if len(self.candidates) < self.max_candidates:
                restored = copy.deepcopy(frac)
                restored.predict_horizon = self.predict_horizon
                self.candidates.append(restored)
                self.memory.store(restored)

    @staticmethod
    def _composition_depth(frac: FunctionFractal) -> int:
        """Return the nesting depth of a fractal.

        Leaf fractals have depth 0. A composition of two leaves has depth 1.
        A composition where one child is itself composed has depth 2, etc.
        """
        if isinstance(frac, NestedComposedFractal):
            d_inner = SymbolicEngine._composition_depth(frac.inner)
            d_outer = SymbolicEngine._composition_depth(frac.outer)
            return 1 + max(d_inner, d_outer)
        if not isinstance(frac, ComposedFunctionFractal):
            return 0
        d1 = SymbolicEngine._composition_depth(frac.child1)
        d2 = SymbolicEngine._composition_depth(frac.child2)
        return 1 + max(d1, d2)

    def get_best(self) -> Tuple[str, float, Optional[FunctionFractal]]:
        """Return (formula_string, fitness, fractal) for the best candidate."""
        if self._best_candidate is None:
            return ("none", 0.0, None)
        return (
            self._best_candidate.symbolic_repr(),
            self._best_candidate.metrics.fitness(),
            self._best_candidate,
        )

    def get_predictions(self, n_future: int) -> np.ndarray:
        """Predict the next n_future values using the best candidate."""
        if self._best_candidate is None:
            return np.zeros(n_future)
        t = len(self._history_x)
        x_future = np.arange(t, t + n_future, dtype=float)
        return self._best_candidate.evaluate(x_future)

    def get_stats(self) -> dict:
        """Return engine and candidate statistics."""
        return {
            "step_count": self._step_count,
            "num_candidates": len(self.candidates),
            "best_formula": (
                self._best_candidate.symbolic_repr()
                if self._best_candidate
                else "none"
            ),
            "best_fitness": (
                self._best_candidate.metrics.fitness()
                if self._best_candidate
                else 0.0
            ),
            "memory": self.memory.stats(),
            "candidate_details": [
                {
                    "name": c.func_name,
                    "fitness": c.metrics.fitness(),
                    "formula": c.symbolic_repr(),
                }
                for c in sorted(
                    self.candidates,
                    key=lambda c: c.metrics.fitness(),
                    reverse=True,
                )
            ],
        }
