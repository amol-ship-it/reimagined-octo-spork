"""
Nested Function Composition — true function composition g(f(x)).

Unlike ComposedFunctionFractal which does element-wise operations
(f(x) + g(x), f(x) * g(x)), NestedComposedFractal chains functions:
  evaluate(x) = outer(inner(x))

This enables discovering patterns like sin(x^2), exp(linear(x)),
log(quadratic(x)), etc.

Fitting strategy:
  - Inner function is kept fixed (from discovery search).
  - Outer function is re-fitted to (inner(x), y) each step.
  - This is well-defined: outer.fit(z, y) where z = inner.evaluate(x).
"""

import numpy as np
from typing import Optional

from .function_fractal import FunctionFractal
from .types import Feedback


class NestedComposedFractal(FunctionFractal):
    """A function built by nesting: evaluate(x) = outer(inner(x)).

    The inner function transforms the x-domain.
    The outer function maps transformed x to y.

    Fitting: inner is held fixed; outer is re-fitted to
    (inner(x), y) at each step.
    """

    def __init__(
        self,
        inner: FunctionFractal,
        outer: FunctionFractal,
    ):
        combined_name = f"{outer.func_name}({inner.func_name}(x))"
        combined_coeffs = len(inner.coefficients) + len(outer.coefficients)
        super().__init__(
            n_coeffs=combined_coeffs,
            func_name=combined_name,
            learning_rate=max(inner.learning_rate, outer.learning_rate),
        )
        self.inner = inner
        self.outer = outer

        # Register as children in the Fractal hierarchy
        self.add_child(inner)
        self.add_child(outer)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        z = self.inner.evaluate(x)
        # Guard against NaN/inf from inner before passing to outer
        z = np.clip(z, -1e6, 1e6)
        z = np.where(np.isfinite(z), z, 0.0)
        return self.outer.evaluate(z)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Re-fit outer to (inner(x), y). Inner stays fixed."""
        z = self.inner.evaluate(x)
        # Guard: skip if inner produces degenerate output
        if not np.all(np.isfinite(z)) or np.std(z) < 1e-10:
            return
        z = np.clip(z, -1e6, 1e6)
        self.outer.fit(z, y)

    def symbolic_repr(self) -> str:
        inner_str = self.inner.symbolic_repr()
        return f"Nested({self.outer.func_name}( {inner_str} ))"

    def learn(self, feedback: Feedback) -> float:
        """Update metrics and propagate to children."""
        error = super().learn(feedback)
        return error

    def compress(self) -> dict:
        base = super().compress()
        base["composition_type"] = "nested"
        base["inner_data"] = self.inner.compress()
        base["outer_data"] = self.outer.compress()
        return base
