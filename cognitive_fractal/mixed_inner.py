"""
MixedInnerFractal — discovers y = F(poly(x) + G(inner_poly(x)))
via residual analysis after inversion.

This extends the system's hypothesis space beyond pure polynomials inside
outer functions.  For example:
  sin(x² + log(x) + 3)   →  F=sin, poly=x²+3, G=log, inner_poly=x

Algorithm:
  1. Apply F⁻¹(y) → z  (get the inner argument)
  2. Polyfit(x, z, degree) → poly_coeffs  (captures polynomial part)
  3. residual = z - poly(x)  (reveals non-polynomial component)
  4. For each G in BASE_FUNCTIONS: fit residual as G(inner_poly(x))
     using InvertedCompositionFractal
  5. Best G that minimizes RMSE wins
  6. One refinement iteration: subtract G contribution, refit poly
"""

import numpy as np
from typing import Optional, List

from .function_fractal import FunctionFractal
from .base_functions import BaseFunction, BASE_FUNCTIONS
from .inverted_composition import InvertedCompositionFractal, _format_polynomial


class MixedInnerFractal(FunctionFractal):
    """f(x) = F(poly(x) + G(inner_poly(x))) — discovered via residual analysis.

    Args:
        outer_func: The outermost function F (e.g., sin).
        poly_degree: Degree of the polynomial component (default 2).
    """

    def __init__(
        self,
        outer_func: BaseFunction,
        poly_degree: int = 2,
        learning_rate: float = 0.01,
    ):
        self.outer_func = outer_func
        self.poly_degree = poly_degree
        self.inner_func: Optional[BaseFunction] = None
        self.inner_coeffs: Optional[np.ndarray] = None  # inner poly coefficients
        self.inner_scale: float = 1.0  # scale factor from joint regression

        # Coefficient layout: [poly_coeffs..., inner_a, inner_b]
        # poly has (poly_degree + 1) coefficients
        # inner poly has 2 coefficients (degree-1 inner polynomial)
        n_coeffs = poly_degree + 1 + 2
        name = f"mixed_{outer_func.name}_poly(deg={poly_degree})"

        super().__init__(
            n_coeffs=n_coeffs, func_name=name, learning_rate=learning_rate
        )
        self._best_fit_rmse = 999.0
        self._fit_count = 0

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate F(poly(x) + G(inner_poly(x)))."""
        poly_coeffs = self.coefficients[: self.poly_degree + 1]
        poly_val = np.polyval(poly_coeffs, x)

        if self.inner_func is not None and self.inner_coeffs is not None:
            inner_poly_val = np.polyval(self.inner_coeffs, x)
            g_val = self.inner_func.forward(inner_poly_val)
            combined = poly_val + self.inner_scale * g_val
        else:
            # Graceful fallback: just F(poly(x))
            combined = poly_val

        return self.outer_func.forward(combined)

    # ----------------------------------------------------------
    # Fit
    # ----------------------------------------------------------

    # Maximum number of points to use for discovery (subsample larger datasets)
    _MAX_FIT_POINTS = 200

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < self.poly_degree + 4:
            return

        # Domain check on the outermost function
        valid = self.outer_func.domain_check(y)
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

        # For large datasets, use the first N consecutive points for
        # discovery (keeps spacing small for reliable unwrapping), then
        # validate candidates on the full dataset.
        x_fit, y_fit = x, y
        if len(x) > self._MAX_FIT_POINTS:
            x_fit = x[: self._MAX_FIT_POINTS]
            y_fit = y[: self._MAX_FIT_POINTS]

        # Get z candidates by inverting the outer function
        z_candidates = self._get_z_candidates(x_fit, y_fit)

        # Collect ALL good candidates from all branches, then pick simplest
        all_candidates: list = []  # (rmse, g_func, g_coeffs, poly_coeffs, scale)
        for z in z_candidates:
            branch_hits = self._collect_from_z(x_fit, z, y_fit)
            all_candidates.extend(branch_hits)

        if not all_candidates:
            return

        # Re-validate top candidates on the FULL dataset for accurate RMSE
        validated: list = []
        # Pre-filter: only keep candidates with training RMSE within 10x of best
        best_train = min(c[0] for c in all_candidates)
        for c in all_candidates:
            if c[0] > best_train * 10 and c[0] > 0.1:
                continue
            rmse_c, g_func_c, g_coeffs_c, poly_c, scale_c = c
            try:
                g_vals = g_func_c.forward(np.polyval(g_coeffs_c, x))
                combined = np.polyval(poly_c, x) + scale_c * g_vals
                full_pred = self.outer_func.forward(combined)
                if np.all(np.isfinite(full_pred)):
                    full_rmse = float(np.sqrt(np.mean((y - full_pred) ** 2)))
                    validated.append((full_rmse, g_func_c, g_coeffs_c, poly_c, scale_c))
            except Exception:
                continue

        if not validated:
            return

        # Find the best RMSE threshold (within 1% of best)
        best_rmse = min(c[0] for c in validated)
        tolerance = max(best_rmse * 0.01, 1e-10)
        near_optimal = [c for c in validated if c[0] <= best_rmse + tolerance]

        # Among near-optimal, pick the simplest
        near_optimal.sort(key=lambda c: self._simplicity_score(c[3], c[4], c[1]))
        winner = near_optimal[0]
        w_rmse, w_func, w_coeffs, w_poly, w_scale = winner

        if w_rmse < self._best_fit_rmse or (
            abs(w_rmse - self._best_fit_rmse) < tolerance
            and self._simplicity_score(w_poly, w_scale, w_func)
            < self._simplicity_score(
                self.coefficients[: self.poly_degree + 1],
                self.inner_scale,
                self.inner_func,
            )
        ):
            self._best_fit_rmse = w_rmse
            self.inner_func = w_func
            self.inner_coeffs = w_coeffs.copy()
            self.inner_scale = w_scale
            self.coefficients[: self.poly_degree + 1] = w_poly
            self.coefficients[self.poly_degree + 1 :] = w_coeffs[:2]

    # Periods for known periodic outer functions
    _PERIODS = {"sin": 2 * np.pi, "cos": 2 * np.pi, "tan": np.pi}

    def _get_z_candidates(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """Invert the outer function to get candidate z arrays."""
        candidates: List[np.ndarray] = []

        if not self.outer_func.has_branches:
            z = self.outer_func.inverse(y)
            if np.all(np.isfinite(z)):
                candidates.append(z)
        else:
            branch_gen = self.outer_func.branch_generator
            if branch_gen is None:
                return candidates

            period = self._PERIODS.get(self.outer_func.name)

            if period is not None:
                # Combinatorial branch search: enumerates branch
                # combinations for the first few seed points, filters by
                # polynomial fit quality, then extends via adaptive unwrap.
                # Handles aliased signals where sequential unwrap fails.
                candidates = self._combinatorial_unwrap(
                    x, y, branch_gen, period
                )
            else:
                # Non-periodic branching (e.g. sqrt) — use fixed K
                K_start = 5
                starting_branches = branch_gen(float(y[0]), K_start)
                for z0 in starting_branches:
                    z = self._predictive_unwrap(y, branch_gen, z0, K=12)
                    if z is not None:
                        candidates.append(z)

        return candidates

    @staticmethod
    def _adaptive_unwrap(
        y: np.ndarray,
        branch_gen,
        z0: float,
        period: float,
        K_local: int = 3,
    ) -> Optional[np.ndarray]:
        """Unwrap a periodic inverse using adaptive branch centering.

        For each step, generates branches centered around the PREDICTED
        value instead of around 0.  This handles z ranges of any size
        (e.g., x² spanning millions of radians inside sin).

        At each step:
          1. Predict z[i] via linear extrapolation from z[i-1], z[i-2]
          2. Get the base branches from branch_gen(y[i], K=0)
          3. For each base branch b, find k_center = round((predicted - b) / period)
          4. Generate candidates b + k*period for k near k_center
          5. Pick the candidate closest to the prediction
        """
        n = len(y)
        z = np.zeros(n)
        z[0] = z0

        for i in range(1, n):
            if i == 1:
                predicted = z[0]
            else:
                predicted = z[i - 1] + (z[i - 1] - z[i - 2])

            # Get the two base branches (K=0 gives just the principal values)
            base_branches = branch_gen(float(y[i]), 0)
            if not base_branches:
                return None

            # For each base branch, center around predicted value
            best_c = None
            best_dist = float("inf")
            for b in base_branches:
                k_center = round((predicted - b) / period)
                for dk in range(-K_local, K_local + 1):
                    c = b + (k_center + dk) * period
                    d = abs(c - predicted)
                    if d < best_dist:
                        best_dist = d
                        best_c = c

            if best_c is None:
                return None
            z[i] = best_c

        if np.max(np.abs(z)) > 1e12:
            return None
        return z

    def _combinatorial_unwrap(
        self,
        x: np.ndarray,
        y: np.ndarray,
        branch_gen,
        period: float,
        B: int = 50,
    ) -> List[np.ndarray]:
        """Find z arrays via 3-point exhaustive search + quadratic extension.

        For aliased periodic signals where sequential unwrap fails, this:

        1. Enumerate all branch combinations for the first 3 data points.
           Per-point K values keep this to ~1K-5K combos.
        2. For each (z₀, z₁, z₂) triple, extend to the full dataset using
           QUADRATIC extrapolation (3-point predictor).  This is exact for
           degree-D polynomials, so the prediction error comes only from the
           non-polynomial part (e.g., log(x)), giving error ~ 0.02 << π.
        3. Validate each full trajectory by polynomial-fit residual norm
           over ALL points (many DOF, very discriminating).
        4. Return the top B trajectories.

        Returns a list of candidate z arrays sorted by fit quality.
        """
        n = len(x)
        D = self.poly_degree

        if n < 4:
            return []

        # Per-point K for the first 3 points (cover expected z range).
        # z ~ coeff * x^D;  use coeff ~ 3 as a moderate bound.
        n_init = 3
        init_branches: list = []
        for i in range(n_init):
            xi = max(abs(float(x[i])), 1.0)
            z_est = 3 * xi ** max(D, 1) + 5
            Ki = max(2, int(np.ceil(z_est / period)))
            Ki = min(Ki, 30)
            branches = branch_gen(float(y[i]), Ki)
            init_branches.append(branches)

        # Cap total combos to ~10K
        total = 1
        for br in init_branches:
            total *= len(br)
        while total > 10_000 and any(len(br) > 6 for br in init_branches):
            sizes = [len(br) for br in init_branches]
            biggest = int(np.argmax(sizes))
            new_K = max(1, int(np.ceil(sizes[biggest] / 2 - 1) / 2) - 1)
            init_branches[biggest] = branch_gen(float(y[biggest]), new_K)
            total = 1
            for br in init_branches:
                total *= len(br)

        # ==============================================================
        # Pre-compute invariants OUTSIDE the triple loop.
        # These only depend on x, not on z.
        # ==============================================================

        # Pre-compute base branches for all extension points (K=0).
        # Avoids redundant branch_gen calls inside the inner loop.
        base_branches_all = [
            branch_gen(float(y[i]), 0) for i in range(n)
        ]

        # Pre-compute QR decompositions for mixed-model validation.
        # For each G function, the design matrix M = [poly_cols, G(x)]
        # is constant across all z candidates.  Using QR decomposition,
        # the residual norm ||z - M(M^+)z||^2 = ||z||^2 - ||Q^T z||^2,
        # reducing per-z cost from O(n^2) lstsq to O(k*n) projection.
        poly_cols = np.column_stack(
            [x ** (D - j) for j in range(D + 1)]
        )

        # G models: (name, Q, R) where M = QR
        g_models: list = []  # (name, Q_transposed_nxk, R)
        for gn, gf in BASE_FUNCTIONS.items():
            try:
                g_vals = gf.forward(x)
                if not np.all(np.isfinite(g_vals)):
                    continue
                if np.std(g_vals) < 1e-10:
                    continue
                M = np.column_stack([poly_cols, g_vals])
                Q, R = np.linalg.qr(M, mode="reduced")
                g_models.append((gn, Q.T.copy(), R))
            except Exception:
                continue

        # Pure polynomial QR
        Q_poly, R_poly = np.linalg.qr(poly_cols, mode="reduced")
        Qt_poly = Q_poly.T.copy()

        # ---- Phase 1: extend each (z0, z1, z2) triple to all n points ----
        scored: list = []

        for z0 in init_branches[0]:
            for z1 in init_branches[1]:
                for z2 in init_branches[2]:
                    z = np.zeros(n)
                    z[0] = z0
                    z[1] = z1
                    z[2] = z2

                    ok = True
                    for i in range(n_init, n):
                        predicted = (
                            3.0 * z[i - 1] - 3.0 * z[i - 2] + z[i - 3]
                        )
                        base_br = base_branches_all[i]
                        best_c = None
                        best_dist = float("inf")
                        for b in base_br:
                            k_ctr = round((predicted - b) / period)
                            for dk in range(-2, 3):
                                c = b + (k_ctr + dk) * period
                                d = abs(c - predicted)
                                if d < best_dist:
                                    best_dist = d
                                    best_c = c
                        if best_c is None:
                            ok = False
                            break
                        z[i] = best_c

                    if not ok or not np.all(np.isfinite(z)):
                        continue
                    if np.max(np.abs(z)) > 1e12:
                        continue

                    # ---- Phase 2: validate by mixed-model fit quality ----
                    # Using pre-computed QR: rn = ||z||^2 - ||Q^T z||^2
                    z_norm_sq = float(np.dot(z, z))
                    best_mixed = float("inf")
                    best_model_idx = -1  # -1 = pure poly

                    for idx, (_gn, Qt, _R) in enumerate(g_models):
                        q_proj = Qt @ z
                        rn = z_norm_sq - float(np.dot(q_proj, q_proj))
                        rn = max(rn, 0.0)  # numerical safety
                        if rn < best_mixed:
                            best_mixed = rn
                            best_model_idx = idx

                    # Pure polynomial
                    q_proj = Qt_poly @ z
                    rn_poly = z_norm_sq - float(np.dot(q_proj, q_proj))
                    rn_poly = max(rn_poly, 0.0)
                    if rn_poly < best_mixed:
                        best_mixed = rn_poly
                        best_model_idx = -1

                    # Simplicity tiebreaker: recover coefficients for
                    # the winning model and score closeness to integers.
                    simp = 0.0
                    try:
                        if best_model_idx >= 0:
                            Qt_win = g_models[best_model_idx][1]
                            R_win = g_models[best_model_idx][2]
                            best_coeffs = np.linalg.solve(R_win, Qt_win @ z)
                        else:
                            best_coeffs = np.linalg.solve(R_poly, Qt_poly @ z)
                        for c in best_coeffs[: D + 1]:
                            nearest = round(float(c))
                            simp += abs(float(c) - nearest)
                        simp += sum(
                            abs(float(c)) for c in best_coeffs[: D + 1]
                        ) * 1e-6
                    except (np.linalg.LinAlgError, ValueError):
                        pass

                    scored.append((best_mixed, simp, z))

        # ---- Phase 3: return top B by (fit quality, simplicity) ----
        # Quantize best_mixed into buckets so that near-zero scores
        # (which differ only at machine-precision) land in the same
        # bucket and the simplicity tiebreaker can act.
        if scored:
            best_score = min(s[0] for s in scored)
            bucket = max(best_score * 100, 1e-6)

            def _sort_key(s):
                quantized = int(s[0] / bucket) if bucket > 0 else 0
                return (quantized, s[1])

            scored.sort(key=_sort_key)

        return [s[2] for s in scored[:B]]

    @staticmethod
    def _predictive_unwrap(
        y: np.ndarray,
        branch_gen,
        z0: float,
        K: int = 12,
    ) -> Optional[np.ndarray]:
        """Unwrap an inverse using derivative extrapolation (fixed K).

        Fallback for non-periodic branching functions (e.g. sqrt).
        """
        n = len(y)
        z = np.zeros(n)
        z[0] = z0

        for i in range(1, n):
            candidates = branch_gen(float(y[i]), K)
            if not candidates:
                return None

            if i == 1:
                z[i] = min(candidates, key=lambda c: abs(c - z[0]))
            else:
                predicted = z[i - 1] + (z[i - 1] - z[i - 2])
                z[i] = min(candidates, key=lambda c: abs(c - predicted))

        if np.max(np.abs(z)) > 1e6:
            return None
        return z

    @staticmethod
    def _simplicity_score(
        poly_coeffs: np.ndarray,
        scale: float,
        g_func: Optional[BaseFunction],
    ) -> float:
        """Score how "simple" a candidate formula is (lower = simpler).

        Prefers:
          - Coefficients close to small integers (0, 1, 2, 3, ...)
          - Scale close to ±1 or 0
          - Smaller absolute constant term
          - "Natural" functions: log over log2/log10
        """
        score = 0.0

        if poly_coeffs is not None:
            for c in poly_coeffs:
                nearest_int = round(float(c))
                score += abs(float(c) - nearest_int)
            # Penalise large constant term
            score += abs(float(poly_coeffs[-1])) * 0.01

        # Penalise scale far from simple values (0, ±1)
        s = float(scale)
        best_s_dist = min(abs(s), abs(s - 1.0), abs(s + 1.0))
        score += best_s_dist * 2.0

        # Prefer natural log over log2/log10
        if g_func is not None:
            if g_func.name in ("log2", "log10"):
                score += 0.5

        return score

    def _collect_from_z(
        self,
        x: np.ndarray,
        z: np.ndarray,
        y_orig: np.ndarray,
    ) -> list:
        """Collect all good candidates from a single branch z.

        Returns list of (rmse, g_func, g_coeffs, poly_coeffs, scale).
        """
        hits: list = []

        try:
            rough_poly = np.polyfit(x, z, self.poly_degree)
        except (np.linalg.LinAlgError, ValueError):
            return hits

        rough_residual = z - np.polyval(rough_poly, x)

        # Skip if residual is negligible (pure polynomial, handled by ICF)
        if np.std(rough_residual) < 1e-6:
            return hits
        if not np.all(np.isfinite(rough_residual)):
            return hits

        for _name, g_func in BASE_FUNCTIONS.items():
            try:
                result = self._fit_single_g(
                    x, z, y_orig, rough_residual, g_func
                )
                if result is not None:
                    g_rmse, g_coeffs, g_poly, g_scale = result
                    hits.append((g_rmse, g_func, g_coeffs, g_poly, g_scale))
            except Exception:
                continue

        return hits

    def _fit_single_g(
        self,
        x: np.ndarray,
        z: np.ndarray,
        y_orig: np.ndarray,
        rough_residual: np.ndarray,
        g_func: BaseFunction,
    ) -> Optional[tuple]:
        """Try a single G function via joint regression.

        1. Find inner poly by fitting G to the rough residual via ICF
        2. Build joint design matrix: [x^d, ..., x, 1, G(inner_poly(x))]
        3. Least-squares solves for poly coefficients AND G scale jointly
        4. Validate full chain F(poly(x) + scale*G(inner_poly(x))) ≈ y

        Returns (best_rmse, best_g_coeffs, best_poly_coeffs, scale) or None.
        """
        best_rmse = 999.0
        best_g_coeffs = None
        best_poly = None
        best_scale = 1.0

        # --- Strategy A: find inner poly via ICF (degree=1) on rough residual ---
        fitter = InvertedCompositionFractal(g_func, degree=1)
        fitter._max_k = 10
        fitter.fit(x, rough_residual)
        inner_candidates = [fitter.coefficients.copy()]

        # --- Strategy B: also try G(x) directly (inner = identity) ---
        inner_candidates.append(np.array([1.0, 0.0]))

        # --- Strategy C: degree-2 inner poly (only for non-branching G) ---
        # Covers cases like sin(x² + log(x²)).  Skip for branching G
        # (sin, cos, etc.) to avoid expensive branch enumeration.
        if not g_func.has_branches:
            fitter2 = InvertedCompositionFractal(g_func, degree=2)
            fitter2._max_k = 8
            fitter2.fit(x, rough_residual)
            inner_candidates.append(fitter2.coefficients.copy())

        for inner_coeffs in inner_candidates:
            g_vals = g_func.forward(np.polyval(inner_coeffs, x))
            if not np.all(np.isfinite(g_vals)):
                continue
            if np.std(g_vals) < 1e-10:
                continue

            # Joint regression: z = c0*x^d + ... + cd + scale*G(inner(x))
            # Build design matrix
            poly_cols = [x ** (self.poly_degree - i) for i in range(self.poly_degree + 1)]
            M = np.column_stack(poly_cols + [g_vals])

            try:
                result = np.linalg.lstsq(M, z, rcond=None)
                joint_coeffs = result[0]
            except np.linalg.LinAlgError:
                continue

            poly_coeffs = joint_coeffs[: self.poly_degree + 1]
            scale = joint_coeffs[self.poly_degree + 1]

            # Validate full chain
            combined = np.polyval(poly_coeffs, x) + scale * g_vals
            full_pred = self.outer_func.forward(combined)
            if not np.all(np.isfinite(full_pred)):
                continue

            r = float(np.sqrt(np.mean((y_orig - full_pred) ** 2)))

            # Scale the inner coefficients to absorb the scale factor
            # G(inner) * scale ≈ G(scaled_inner) for monotonic G
            # but it's cleaner to keep scale separate by adjusting inner
            scaled_inner = inner_coeffs.copy()
            if abs(scale) > 1e-10 and abs(scale - 1.0) > 1e-8:
                # For log: scale*log(ax+b) = log((ax+b)^scale)
                # Approximation: absorb into inner poly or keep as-is
                # Simplest: store scale in inner coeffs: scale*G(ax+b) ≈ G(scale*(ax+b)) only for linear G
                # For safety, just adjust the effective inner coefficients
                pass  # Keep as-is and validate below

            if r < best_rmse:
                best_rmse = r
                best_g_coeffs = inner_coeffs.copy()
                best_poly = poly_coeffs.copy()
                best_scale = scale

        if best_g_coeffs is None:
            return None

        return best_rmse, best_g_coeffs, best_poly, best_scale

    # ----------------------------------------------------------
    # Symbolic representation
    # ----------------------------------------------------------

    @staticmethod
    def _snap_coefficients(coeffs: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Round coefficients that are very close to integers."""
        snapped = coeffs.copy()
        for i in range(len(snapped)):
            nearest = round(float(snapped[i]))
            if abs(float(snapped[i]) - nearest) < tol:
                snapped[i] = nearest
        return snapped

    def symbolic_repr(self) -> str:
        poly_coeffs = self._snap_coefficients(
            self.coefficients[: self.poly_degree + 1]
        )
        poly_str = _format_polynomial(poly_coeffs, self.poly_degree)

        if self.inner_func is not None and self.inner_coeffs is not None:
            inner_snapped = self._snap_coefficients(self.inner_coeffs)
            inner_poly_str = _format_polynomial(inner_snapped, len(inner_snapped) - 1)
            snapped_scale = self.inner_scale
            if abs(snapped_scale - round(snapped_scale)) < 1e-3:
                snapped_scale = round(snapped_scale)
            if abs(snapped_scale - 1.0) < 1e-6:
                g_str = f"{self.inner_func.name}({inner_poly_str})"
            else:
                g_str = f"{snapped_scale:.4f}*{self.inner_func.name}({inner_poly_str})"
            return f"{self.outer_func.name}({poly_str} + {g_str})"
        else:
            return f"{self.outer_func.name}({poly_str})"

    # ----------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------

    def compress(self) -> dict:
        base = super().compress()
        base["outer_func_name"] = self.outer_func.name
        base["poly_degree"] = self.poly_degree
        base["inner_func_name"] = self.inner_func.name if self.inner_func else None
        base["inner_scale"] = self.inner_scale
        if self.inner_coeffs is not None:
            base["inner_coeffs"] = self.inner_coeffs.tolist()
        return base
