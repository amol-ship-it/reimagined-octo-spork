"""
Inverted Composition Fractal — discovers y = F₁(F₂(...Fₙ(poly(x))...))
via chained inversion.

Supports arbitrary nesting depth:
  depth 1: InvertedCompositionFractal(sin, degree=1)            →  sin(poly(x))
  depth 2: InvertedCompositionFractal([sin, cos], degree=1)     →  sin(cos(poly(x)))
  depth 3: InvertedCompositionFractal([sin, cos, exp], degree=1)→  sin(cos(exp(poly(x))))

The inversion approach works inside-out:
  1. Apply F₁⁻¹ to y   →  z₁ = F₂(...Fₙ(poly(x))...)
  2. Apply F₂⁻¹ to z₁  →  z₂ = F₃(...Fₙ(poly(x))...)
  ...
  n. Apply Fₙ⁻¹ to zₙ₋₁  →  w = poly(x)
  n+1. polyfit(x, w) recovers the polynomial coefficients

For branching inverses (sin, cos, tan): the outermost inversions use
smooth unwrapping (picks closest branch per point for continuity).
The innermost inversion uses full Vandermonde branch enumeration for
maximum accuracy.

For non-branching inverses (exp, log, tanh): single inverse call.
"""

import numpy as np
from itertools import product as iter_product
from scipy.optimize import minimize as _scipy_minimize
from typing import List, Optional, Union

from .function_fractal import FunctionFractal
from .base_functions import BaseFunction


class InvertedCompositionFractal(FunctionFractal):
    """f(x) = F₁(F₂(...Fₙ(poly(x))...)) — discovered via chained inversion.

    Args:
        outer_func: A single BaseFunction, or a list of BaseFunctions
                    ordered outermost-first.
                    - BaseFunction: sin → sin(poly(x))
                    - [sin, cos]   → sin(cos(poly(x)))
                    - [sin, cos, exp] → sin(cos(exp(poly(x))))
        degree: Polynomial degree for the innermost poly(x).
    """

    def __init__(
        self,
        outer_func: Union[BaseFunction, List[BaseFunction]],
        degree: int = 1,
        learning_rate: float = 0.01,
        outer_scale: float = 1.0,
        outer_offset: float = 0.0,
    ):
        # Normalise to a chain (list of BaseFunctions, outermost first)
        if isinstance(outer_func, (list, tuple)):
            self.chain: List[BaseFunction] = list(outer_func)
        else:
            self.chain = [outer_func]

        if not self.chain:
            raise ValueError("chain must contain at least one BaseFunction")

        self.degree = degree

        # Outer scaling: y = scale * F(poly(x)) + offset
        self._outer_scale = float(outer_scale)
        self._outer_offset = float(outer_offset)

        n_coeffs = degree + 1

        # Backward-compat: self.outer_func points to chain[0]
        # (only meaningful for single-function chains)
        self.outer_func = self.chain[0]

        # Build human-readable name
        name = self._make_name()

        super().__init__(
            n_coeffs=n_coeffs, func_name=name, learning_rate=learning_rate
        )
        self._max_k = 20 if degree == 1 else 15
        # Total enumeration budget — adaptive K trims per-point branches
        # to keep total combos under this limit.
        self._max_branch_combos = 500_000
        self._best_fit_rmse = 999.0
        self._fit_count = 0

    @property
    def _is_scaled(self) -> bool:
        return abs(self._outer_scale - 1.0) > 1e-9 or abs(self._outer_offset) > 1e-9

    def _make_name(self) -> str:
        deg = self.degree
        if len(self.chain) == 1:
            f = self.chain[0].name
            base = f"{f}_poly" if deg == 1 else f"{f}_poly(deg={deg})"
        else:
            parts = [f.name for f in self.chain]
            base = "(".join(parts) + f"_poly(deg={deg})" + ")" * (len(parts) - 1)
        if self._is_scaled:
            return f"scaled_{base}"
        return base

    # ----------------------------------------------------------
    # Evaluate — chain all functions innermost-first
    # ----------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        val = np.polyval(self.coefficients, x)
        for func in reversed(self.chain):
            val = func.forward(val)
        if self._is_scaled:
            val = self._outer_scale * val + self._outer_offset
        return val

    # ----------------------------------------------------------
    # Fit — dispatcher
    # ----------------------------------------------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < self.degree + 2:
            return

        # Normalise for scaled fitting: y = scale * F(chain(poly(x))) + offset
        # => F(chain(poly(x))) = (y - offset) / scale
        y_fit = y
        if self._is_scaled:
            y_fit = (y - self._outer_offset) / self._outer_scale

        # Domain check on the (possibly normalised) data
        valid = self.chain[0].domain_check(y_fit)
        if not np.all(valid):
            return

        self._fit_count += 1

        # If we already have a great fit, just verify and skip re-search
        if self._best_fit_rmse < 0.01:
            pred = self.evaluate(x)
            if np.all(np.isfinite(pred)):
                r = float(np.sqrt(np.mean((y - pred) ** 2)))
                if r < 0.05:
                    return  # Still good
            self._best_fit_rmse = 999.0  # Degraded — re-search

        # After the first successful fit, only re-search periodically
        if self._fit_count > 1 and self._best_fit_rmse < 0.5:
            if self._fit_count % 50 != 0:
                return

        if len(self.chain) == 1:
            # Single function: original highly-optimised path
            if self.chain[0].has_branches:
                self._fit_branching(x, y_fit)
            else:
                self._fit_direct(x, y_fit)
        else:
            # Multi-level: invert the outer chain, then fit the innermost
            self._fit_chain(x, y_fit)

        # Scipy fallback — jointly optimise scale + poly when inversion
        # is imprecise (branch crossovers in chains, or normalisation
        # clipping in scaled compositions).
        if self._is_scaled or len(self.chain) > 1:
            pred = self.evaluate(x)
            orig_rmse = float("inf")
            if np.all(np.isfinite(pred)):
                orig_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            if orig_rmse > 1e-4:
                self._refine_full_params(x, y)

        # Canonicalize trig representations for cleaner symbolic output.
        # Must run AFTER all fitting/refinement since it transforms
        # coefficients and may swap sin↔cos.
        self._canonicalize()

    # ==========================================================
    # SINGLE-FUNCTION FITTING (chain length == 1)
    # ==========================================================

    def _fit_direct(self, x: np.ndarray, y: np.ndarray) -> None:
        """Single non-branching function: inverse + polyfit."""
        z = self.outer_func.inverse(y)
        if not np.all(np.isfinite(z)):
            return

        try:
            coeffs = np.polyfit(x, z, self.degree)
        except (np.linalg.LinAlgError, ValueError):
            return

        trial_inner = np.polyval(coeffs, x)
        trial_pred = self.outer_func.forward(trial_inner)
        if not np.all(np.isfinite(trial_pred)):
            return

        r = float(np.sqrt(np.mean((y - trial_pred) ** 2)))
        if r < self._best_fit_rmse:
            self._best_fit_rmse = r
            self.coefficients = coeffs

    def _fit_branching(self, x: np.ndarray, y: np.ndarray) -> None:
        """Single branching function: Vandermonde branch enumeration."""
        branch_gen = self.outer_func.branch_generator
        if branch_gen is None:
            return

        n_solve = self.degree + 1
        x_pts = x[:n_solve]
        y_pts = y[:n_solve]

        # Validation points for early pruning
        n_val = min(3, len(x) - n_solve)
        x_val = x[n_solve : n_solve + n_val]
        y_val = y[n_solve : n_solve + n_val]

        K = self._max_k
        best_rmse = self._best_fit_rmse
        best_coeffs = self.coefficients.copy()

        # Build Vandermonde matrix:  V @ coeffs = [u1, u2, ...]
        V = np.column_stack(
            [x_pts ** (self.degree - i) for i in range(n_solve)]
        )
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            return

        # Generate per-point branch lists with adaptive budget trimming.
        # Total combos = product of per-point branch counts.
        # To keep total under budget, cap each point's branch list to
        # budget^(1/n_solve).  Trimmed branches are sorted by |z| so
        # the simplest polynomial solutions (small coefficients) are
        # tried first.
        max_per_point = max(
            4, int(self._max_branch_combos ** (1.0 / n_solve))
        )

        per_point_branches = []
        for i in range(n_solve):
            branches_i = branch_gen(float(y_pts[i]), K)
            if not branches_i:
                return
            if len(branches_i) > max_per_point:
                branches_i.sort(key=lambda z: abs(z))
                branches_i = branches_i[:max_per_point]
            per_point_branches.append(branches_i)

        # Enumerate all branch combinations
        for combo in iter_product(*per_point_branches):
            u = np.array(combo)
            c_trial = V_inv @ u

            # Early prune on validation points
            if n_val > 0:
                val_inner = np.polyval(c_trial, x_val)
                val_pred = self.outer_func.forward(val_inner)
                if (
                    not np.all(np.isfinite(val_pred))
                    or np.max(np.abs(val_pred - y_val)) > 0.05
                ):
                    continue

            # Full evaluation
            trial_inner = np.polyval(c_trial, x)
            trial_pred = self.outer_func.forward(trial_inner)
            if not np.all(np.isfinite(trial_pred)):
                continue

            r = float(np.sqrt(np.mean((y - trial_pred) ** 2)))
            if r < best_rmse:
                best_rmse = r
                best_coeffs = c_trial.copy()

        if best_rmse < self._best_fit_rmse:
            self._best_fit_rmse = best_rmse
            self.coefficients = best_coeffs

    # ==========================================================
    # MULTI-LEVEL CHAIN FITTING (chain length >= 2)
    # ==========================================================

    def _fit_chain(self, x: np.ndarray, y: np.ndarray) -> None:
        """Invert outer chain, then fit innermost via branch enumeration.

        1. Recursively invert chain[:-1] (all except innermost) to
           produce candidate z arrays.  Branching functions use smooth
           unwrapping; non-branching use a single inverse call.
        2. For each z, create a temporary single-function
           InvertedCompositionFractal for the innermost function and
           fit it to (x, z).
        3. Validate against the original y through the full chain.
        4. If inversion-based fitting yields poor RMSE, refine
           coefficients via direct forward-loss optimisation.
        """
        outers = self.chain[:-1]
        z_candidates = self._invert_outer_chain(y, outers)

        for z in z_candidates:
            self._try_innermost_fit(x, z, y)
        # NOTE: scipy refinement is now unified in fit() via
        # _refine_full_params for both chains and scaled compositions.

    def _invert_outer_chain(
        self, y: np.ndarray, funcs: List[BaseFunction]
    ) -> List[np.ndarray]:
        """Recursively invert a chain of outer functions.

        Returns a list of candidate z arrays.  Each non-branching
        function produces exactly one z; each branching function
        fans out via smooth unwrapping from multiple starting branches.
        """
        if not funcs:
            return [y]

        func = funcs[0]
        rest = funcs[1:]

        results: List[np.ndarray] = []

        if func.has_branches:
            branch_gen = func.branch_generator
            if branch_gen is None:
                return results
            K_start = 3
            starting_branches = branch_gen(float(y[0]), K_start)
            for z0 in starting_branches:
                z = self._smooth_unwrap(y, branch_gen, z0)
                if z is not None:
                    results.extend(self._invert_outer_chain(z, rest))
        else:
            z = func.inverse(y)
            if np.all(np.isfinite(z)):
                results.extend(self._invert_outer_chain(z, rest))

        return results

    @staticmethod
    def _smooth_unwrap(
        y: np.ndarray,
        branch_gen,
        z0: float,
        K: int = 3,
    ) -> Optional[np.ndarray]:
        """Unwrap an inverse smoothly by choosing the closest branch.

        Assumes the true intermediate signal is smooth (continuous),
        so consecutive z-values should be close together.  For each
        data point, picks the branch candidate nearest to the
        previous z value.
        """
        n = len(y)
        z = np.zeros(n)
        z[0] = z0
        for i in range(1, n):
            candidates = branch_gen(float(y[i]), K)
            if not candidates:
                return None
            z[i] = min(candidates, key=lambda c: abs(c - z[i - 1]))

        if np.max(np.abs(z)) > 1e6:
            return None
        return z

    def _try_innermost_fit(
        self,
        x: np.ndarray,
        z: np.ndarray,
        y_orig: np.ndarray,
    ) -> None:
        """Fit innermost(poly(x)) to (x, z), validate against y_orig.

        Creates a fresh single-function InvertedCompositionFractal for
        the innermost function, reusing its branch-enumeration logic.
        Then validates the full chain against the original data.
        """
        innermost = self.chain[-1]

        fitter = InvertedCompositionFractal(innermost, degree=self.degree)
        fitter._max_k = 10  # smaller K for chain fits

        try:
            fitter.fit(x, z)
        except Exception:
            return

        # Validate the complete chain against original y
        trial_coeffs = fitter.coefficients
        trial_pred = self._eval_with_coeffs(trial_coeffs, x)
        if not np.all(np.isfinite(trial_pred)):
            return

        r = float(np.sqrt(np.mean((y_orig - trial_pred) ** 2)))
        if r < self._best_fit_rmse:
            self._best_fit_rmse = r
            self.coefficients = trial_coeffs.copy()

    def _eval_with_coeffs(
        self, coeffs: np.ndarray, x: np.ndarray
    ) -> np.ndarray:
        """Evaluate the full chain with given polynomial coefficients."""
        val = np.polyval(coeffs, x)
        for func in reversed(self.chain):
            val = func.forward(val)
        return val

    # ----------------------------------------------------------
    # Direct optimisation fallback
    # ----------------------------------------------------------

    def _refine_full_params(
        self, x: np.ndarray, y: np.ndarray
    ) -> None:
        """Jointly optimise scale, offset, and polynomial coefficients.

        Used as a fallback when inversion-based fitting fails:
        - Branch crossovers in multi-level chains (cos symmetry)
        - Imprecise normalisation in scaled compositions (arcsin clipping)

        Operates on the ORIGINAL y data; the loss includes scaling.
        """
        chain_rev = list(reversed(self.chain))
        is_scaled = self._is_scaled

        def _eval_full(params: np.ndarray) -> np.ndarray:
            if is_scaled:
                scale, offset = params[0], params[1]
                coeffs = params[2:]
            else:
                scale, offset = 1.0, 0.0
                coeffs = params
            val = np.polyval(coeffs, x)
            for func in chain_rev:
                val = func.forward(val)
            return scale * val + offset

        def _loss(params: np.ndarray) -> float:
            val = _eval_full(params)
            if not np.all(np.isfinite(val)):
                return 1e12
            return float(np.mean((y - val) ** 2))

        # Build initial guesses — each is a full parameter vector
        n_poly = self.degree + 1
        guesses: list = []

        # Current best
        if self._best_fit_rmse < 999.0:
            if is_scaled:
                guesses.append(np.concatenate([
                    [self._outer_scale, self._outer_offset],
                    self.coefficients,
                ]))
            else:
                guesses.append(self.coefficients.copy())

        # Standard polynomial seeds
        poly_seeds: list = []
        if self.degree == 1:
            poly_seeds = [
                [1, 0], [-1, 0], [2, 0], [-2, 0],
                [1, 1], [-1, 1], [0.5, 0], [3, 0],
            ]
        elif self.degree == 2:
            poly_seeds = [
                [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 0, 0],
            ]
        else:
            c = [0.0] * n_poly
            c[-2] = 1.0
            poly_seeds = [c]

        if is_scaled:
            s, o = self._outer_scale, self._outer_offset
            for ps in poly_seeds:
                guesses.append(np.array([s, o] + ps))
                guesses.append(np.array([-s, o] + ps))
        else:
            for ps in poly_seeds:
                guesses.append(np.array(ps))

        # Track best across guesses
        best_loss = float("inf")
        pred = self.evaluate(x)
        if np.all(np.isfinite(pred)):
            best_loss = float(np.mean((y - pred) ** 2))
        best_params = guesses[0].copy() if guesses else None

        for guess in guesses:
            try:
                res = _scipy_minimize(
                    _loss,
                    np.asarray(guess, dtype=float),
                    method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-16},
                )
                if res.fun < best_loss:
                    val = _eval_full(res.x)
                    if np.all(np.isfinite(val)):
                        best_loss = res.fun
                        best_params = res.x.copy()
            except Exception:
                continue

        if best_params is None:
            return

        # Unpack and apply
        rmse = float(np.sqrt(best_loss))
        if is_scaled:
            new_scale, new_offset = best_params[0], best_params[1]
            new_coeffs = best_params[2:]
        else:
            new_scale, new_offset = 1.0, 0.0
            new_coeffs = best_params

        # Only accept if meaningfully better
        pred_new = self.evaluate(x)  # current
        cur_rmse = float("inf")
        if np.all(np.isfinite(pred_new)):
            cur_rmse = float(np.sqrt(np.mean((y - pred_new) ** 2)))

        if rmse < cur_rmse:
            self._outer_scale = float(new_scale)
            self._outer_offset = float(new_offset)
            self.coefficients = new_coeffs
            self._best_fit_rmse = rmse

    # ----------------------------------------------------------
    # Trig canonicalization
    # ----------------------------------------------------------

    def _canonicalize(self) -> None:
        """Normalize trig representations to canonical form.

        When the innermost function in the chain is sin or cos, applies:
        1. Positive leading coefficient (using even/odd symmetry)
        2. Positive outer scale (absorbing sign into phase)
        3. Phase reduction mod 2π to [-π, π)
        4. sin↔cos comparison, preferring simpler phase
        5. Coefficient snapping to nearby integers/fractions

        Examples:
          cos(-2x - 24.56)      → sin(2x + 1)      (phase reduction + sin swap)
          -0.5*sin(2x - π)      → 0.5*sin(2x)       (absorb negative scale)
          exp(cos(-2x - 24.56)) → exp(sin(2x + 1))   (works through chains)
        """
        innermost = self.chain[-1]
        if innermost.name not in ("sin", "cos"):
            return

        if self.degree < 1:
            return  # No x-dependent coefficient to orient

        coeffs = self.coefficients.copy()
        scale = self._outer_scale
        func_name = innermost.name

        # --- Step 1: Ensure positive leading polynomial coefficient ---
        # Find first non-negligible x-dependent coefficient
        lead_idx = None
        for i in range(len(coeffs) - 1):  # Exclude constant term
            if abs(coeffs[i]) > 1e-10:
                lead_idx = i
                break

        if lead_idx is not None and coeffs[lead_idx] < 0:
            if func_name == "cos":
                # cos is even: cos(-z) = cos(z)
                coeffs = -coeffs
            else:
                # sin is odd: sin(-z) = -sin(z)
                coeffs = -coeffs
                scale = -scale

        # --- Step 2: Absorb negative outer scale into phase ---
        # -s * sin(θ) = s * sin(θ + π)   [since sin(θ+π) = -sin(θ)]
        # -s * cos(θ) = s * cos(θ + π)   [since cos(θ+π) = -cos(θ)]
        if scale < -1e-10:
            coeffs[-1] += np.pi
            scale = -scale

        # --- Step 3: Reduce phase mod 2π to [-π, π) ---
        TWO_PI = 2.0 * np.pi
        phase = float(coeffs[-1])
        phase = ((phase + np.pi) % TWO_PI) - np.pi
        coeffs[-1] = phase

        # --- Step 4: Compare sin vs cos, prefer simpler phase ---
        def _phase_simplicity(p: float) -> float:
            """Lower = simpler.  Measures distance to nearest integer."""
            p = ((p + np.pi) % TWO_PI) - np.pi
            return abs(p - round(p))

        cur_simplicity = _phase_simplicity(phase)

        if func_name == "sin":
            # sin(z) = cos(z - π/2)
            alt_phase = ((phase - np.pi / 2 + np.pi) % TWO_PI) - np.pi
            if _phase_simplicity(alt_phase) < cur_simplicity - 1e-6:
                func_name = "cos"
                phase = alt_phase
        else:
            # cos(z) = sin(z + π/2)
            alt_phase = ((phase + np.pi / 2 + np.pi) % TWO_PI) - np.pi
            if _phase_simplicity(alt_phase) < cur_simplicity - 1e-6:
                func_name = "sin"
                phase = alt_phase

        coeffs[-1] = phase

        # --- Step 5: Snap coefficients to nearby clean values ---
        _SNAP_TOL = 5e-4
        _SNAP_TARGETS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]

        for i in range(len(coeffs)):
            val = coeffs[i]
            for t in _SNAP_TARGETS:
                for sign in (1, -1):
                    if abs(val - sign * t) < _SNAP_TOL:
                        coeffs[i] = float(sign * t)
                        break

        # Snap scale
        for t in _SNAP_TARGETS:
            for sign in (1, -1):
                if abs(scale - sign * t) < _SNAP_TOL:
                    scale = float(sign * t)
                    break

        # Snap offset
        offset = self._outer_offset
        for t in _SNAP_TARGETS:
            for sign in (1, -1):
                if abs(offset - sign * t) < _SNAP_TOL:
                    offset = float(sign * t)
                    break

        # --- Apply changes ---
        from .base_functions import BASE_FUNCTIONS
        if func_name != innermost.name:
            self.chain[-1] = BASE_FUNCTIONS[func_name]
            if len(self.chain) == 1:
                self.outer_func = self.chain[0]
        self.coefficients = coeffs
        self._outer_scale = scale
        self._outer_offset = offset
        self.func_name = self._make_name()

    # ----------------------------------------------------------
    # Symbolic representation
    # ----------------------------------------------------------

    def symbolic_repr(self) -> str:
        poly_str = _format_polynomial(self.coefficients, self.degree)
        if len(self.chain) == 1:
            inner = f"{self.chain[0].name}({poly_str})"
        else:
            # Build nested: sin(cos(exp(poly_str)))
            result = poly_str
            for func in reversed(self.chain):
                result = f"{func.name}({result})"
            inner = result

        if not self._is_scaled:
            return inner

        # Format: scale*F(...) + offset
        parts: list = []
        if abs(self._outer_scale - 1.0) > 1e-6:
            parts.append(f"{self._outer_scale:.4f}*{inner}")
        else:
            parts.append(inner)
        if abs(self._outer_offset) > 1e-6:
            parts.append(f" + {self._outer_offset:.4f}")
        return "".join(parts)

    # ----------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------

    def compress(self) -> dict:
        base = super().compress()
        base["chain"] = [f.name for f in self.chain]
        base["inner_degree"] = self.degree
        if self._is_scaled:
            base["outer_scale"] = self._outer_scale
            base["outer_offset"] = self._outer_offset
        # Backward compat: single chains also emit outer_func_name
        if len(self.chain) == 1:
            base["outer_func_name"] = self.chain[0].name
        return base


# ================================================================
# Helpers
# ================================================================

def _format_polynomial(coeffs: np.ndarray, degree: int) -> str:
    """Pretty-print polynomial coefficients as a human-readable string."""
    parts = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-10:
            continue
        if power == 0:
            parts.append(f"{c:.4f}")
        elif power == 1:
            parts.append(f"{c:.4f}*x")
        else:
            parts.append(f"{c:.6f}*x^{power}")
    return " + ".join(parts) if parts else "0"
