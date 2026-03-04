"""
PatternStore — persists discovered function patterns to disk.

After each discovery run the best candidates are saved as JSON.
On the next run, saved patterns are loaded and injected into the
candidate pool so the system starts from prior knowledge instead
of scratch.

Default location: ~/.cognitive_fractal/patterns.json

Stored pattern format (one per JSON object in a list):
  {
    "kind": "inverted_composition" | "simple",
    "func_name": "sin_poly(deg=2)",
    "outer_func_name": "sin",           # only for inverted_composition
    "inner_degree": 2,                   # only for inverted_composition
    "simple_type": "LinearFractal",      # only for simple
    "coefficients": [0.1, 0.5, 1.0],
    "rmse": 0.001,
    "formula": "sin(0.1*x^2 + 0.5*x + 1.0)",
    "source": "data.csv",
    "timestamp": "2026-02-28T12:00:00"
  }
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

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
)
from .base_functions import BASE_FUNCTIONS
from .inverted_composition import InvertedCompositionFractal
from .mixed_inner import MixedInnerFractal


# Maps type name -> factory.  Factories that take no args.
_SIMPLE_FACTORIES = {
    "ConstantFractal": ConstantFractal,
    "LinearFractal": LinearFractal,
    "QuadraticFractal": QuadraticFractal,
    "SinFractal": SinFractal,
    "CosFractal": CosFractal,
    "GradientSinFractal": GradientSinFractal,
    "GradientCosFractal": GradientCosFractal,
    "ExponentialFractal": ExponentialFractal,
    "LogFractal": LogFractal,
}

# Polynomial requires a degree parameter
_POLY_FACTORIES = {
    3: lambda: PolynomialFractal(degree=3),
    4: lambda: PolynomialFractal(degree=4),
    5: lambda: PolynomialFractal(degree=5),
}

DEFAULT_STORE_PATH = os.path.join(
    os.path.expanduser("~"), ".cognitive_fractal", "patterns.json"
)


class PatternStore:
    """Saves and loads discovered function patterns to/from disk.

    Args:
        path: File path for the JSON store.  Defaults to
              ~/.cognitive_fractal/patterns.json
        max_patterns: Maximum patterns to keep (oldest pruned first).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        max_patterns: int = 200,
    ):
        self.path = path or DEFAULT_STORE_PATH
        self.max_patterns = max_patterns

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    def save(
        self,
        candidates: List[FunctionFractal],
        x: np.ndarray,
        y: np.ndarray,
        source_hint: str = "",
        top_k: int = 20,
    ) -> int:
        """Save the best candidates from a discovery run.

        Scores each candidate on (x, y), keeps the top_k by RMSE,
        and appends them to the store on disk.

        Returns the number of new patterns saved.
        """
        # Score candidates
        scored = []
        for c in candidates:
            try:
                y_pred = c.evaluate(x)
                if not np.all(np.isfinite(y_pred)):
                    continue
                rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
                scored.append((rmse, c))
            except Exception:
                continue

        scored.sort(key=lambda t: t[0])
        best = scored[:top_k]

        # Serialize
        records = []
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for rmse, c in best:
            rec = self._serialize_candidate(c, rmse, source_hint, ts)
            if rec is not None:
                records.append(rec)

        if not records:
            return 0

        # Load existing, merge, prune, write
        existing = self._read_raw()
        existing.extend(records)

        # De-duplicate: keep the lowest RMSE for each formula
        by_formula = {}
        for rec in existing:
            key = rec["formula"]
            if key not in by_formula or rec["rmse"] < by_formula[key]["rmse"]:
                by_formula[key] = rec
        merged = sorted(by_formula.values(), key=lambda r: r["rmse"])

        # Prune to max_patterns
        merged = merged[: self.max_patterns]

        self._write_raw(merged)
        return len(records)

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    def load(self) -> List[FunctionFractal]:
        """Load saved patterns from disk and reconstruct live fractals.

        Returns a list of FunctionFractal objects with coefficients
        pre-set from prior runs.
        """
        raw = self._read_raw()
        results = []
        for rec in raw:
            frac = self._deserialize_candidate(rec)
            if frac is not None:
                results.append(frac)
        return results

    def count(self) -> int:
        """Return the number of stored patterns."""
        return len(self._read_raw())

    def clear(self) -> None:
        """Delete all stored patterns."""
        if os.path.exists(self.path):
            os.remove(self.path)

    # ----------------------------------------------------------
    # Serialize / deserialize
    # ----------------------------------------------------------

    def _serialize_candidate(
        self,
        c: FunctionFractal,
        rmse: float,
        source: str,
        timestamp: str,
    ) -> Optional[dict]:
        """Convert a FunctionFractal to a JSON-friendly dict."""
        coeffs = c.coefficients.tolist()

        if isinstance(c, MixedInnerFractal):
            rec = {
                "kind": "mixed_inner",
                "func_name": c.func_name,
                "outer_func_name": c.outer_func.name,
                "poly_degree": c.poly_degree,
                "inner_func_name": c.inner_func.name if c.inner_func else None,
                "inner_scale": c.inner_scale,
                "coefficients": coeffs,
                "rmse": rmse,
                "formula": c.symbolic_repr(),
                "source": source,
                "timestamp": timestamp,
            }
            if c.inner_coeffs is not None:
                rec["inner_coeffs"] = c.inner_coeffs.tolist()
            return rec

        if isinstance(c, InvertedCompositionFractal):
            rec = {
                "kind": "inverted_composition",
                "func_name": c.func_name,
                "chain": [f.name for f in c.chain],
                "inner_degree": c.degree,
                "coefficients": coeffs,
                "rmse": rmse,
                "formula": c.symbolic_repr(),
                "source": source,
                "timestamp": timestamp,
            }
            # Backward compat: single chains also emit outer_func_name
            if len(c.chain) == 1:
                rec["outer_func_name"] = c.chain[0].name
            return rec

        # Simple types
        type_name = type(c).__name__
        if type_name in _SIMPLE_FACTORIES:
            return {
                "kind": "simple",
                "func_name": c.func_name,
                "simple_type": type_name,
                "coefficients": coeffs,
                "rmse": rmse,
                "formula": c.symbolic_repr(),
                "source": source,
                "timestamp": timestamp,
            }

        # PolynomialFractal
        if isinstance(c, PolynomialFractal):
            return {
                "kind": "polynomial",
                "func_name": c.func_name,
                "degree": c.degree,
                "coefficients": coeffs,
                "rmse": rmse,
                "formula": c.symbolic_repr(),
                "source": source,
                "timestamp": timestamp,
            }

        return None  # Unknown type — skip

    def _deserialize_candidate(self, rec: dict) -> Optional[FunctionFractal]:
        """Reconstruct a FunctionFractal from a saved dict."""
        kind = rec.get("kind")
        coeffs = np.array(rec["coefficients"])

        if kind == "mixed_inner":
            outer_name = rec.get("outer_func_name")
            if not outer_name or outer_name not in BASE_FUNCTIONS:
                return None
            poly_degree = rec.get("poly_degree", 2)
            frac = MixedInnerFractal(
                BASE_FUNCTIONS[outer_name], poly_degree=poly_degree
            )
            frac.coefficients = coeffs
            frac._best_fit_rmse = rec.get("rmse", 999.0)
            frac.inner_scale = rec.get("inner_scale", 1.0)
            inner_name = rec.get("inner_func_name")
            if inner_name and inner_name in BASE_FUNCTIONS:
                frac.inner_func = BASE_FUNCTIONS[inner_name]
                inner_coeffs = rec.get("inner_coeffs")
                if inner_coeffs is not None:
                    frac.inner_coeffs = np.array(inner_coeffs)
            return frac

        if kind == "double_composition":
            # Backward compat: old DoubleCompositionFractal records
            # are now loaded as chain-based InvertedCompositionFractal
            outer_name = rec.get("outer_func_name")
            inner_name = rec.get("inner_func_name")
            degree = rec.get("inner_degree", 1)
            if outer_name not in BASE_FUNCTIONS or inner_name not in BASE_FUNCTIONS:
                return None
            frac = InvertedCompositionFractal(
                [BASE_FUNCTIONS[outer_name], BASE_FUNCTIONS[inner_name]],
                degree=degree,
            )
            frac.coefficients = coeffs
            frac._best_fit_rmse = rec.get("rmse", 999.0)
            return frac

        if kind == "inverted_composition":
            degree = rec.get("inner_degree", 1)
            # New format: chain list
            chain_names = rec.get("chain")
            if chain_names:
                chain_funcs = []
                for name in chain_names:
                    if name not in BASE_FUNCTIONS:
                        return None
                    chain_funcs.append(BASE_FUNCTIONS[name])
                frac = InvertedCompositionFractal(chain_funcs, degree=degree)
            else:
                # Old format: single outer_func_name
                outer_name = rec.get("outer_func_name")
                if not outer_name or outer_name not in BASE_FUNCTIONS:
                    return None
                frac = InvertedCompositionFractal(
                    BASE_FUNCTIONS[outer_name], degree=degree
                )
            frac.coefficients = coeffs
            frac._best_fit_rmse = rec.get("rmse", 999.0)
            return frac

        if kind == "simple":
            type_name = rec.get("simple_type")
            factory = _SIMPLE_FACTORIES.get(type_name)
            if factory is None:
                return None
            frac = factory()
            # Only set coefficients if shapes match
            if len(coeffs) == len(frac.coefficients):
                frac.coefficients = coeffs
            return frac

        if kind == "polynomial":
            degree = rec.get("degree", 3)
            frac = PolynomialFractal(degree=degree)
            if len(coeffs) == len(frac.coefficients):
                frac.coefficients = coeffs
            return frac

        return None

    # ----------------------------------------------------------
    # Raw file I/O
    # ----------------------------------------------------------

    def _read_raw(self) -> list:
        """Read the JSON file, returning [] if missing or corrupt."""
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, IOError):
            return []

    def _write_raw(self, records: list) -> None:
        """Write records to the JSON file, creating dirs as needed."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(records, f, indent=2)
