"""
Function Fractals — mathematical function hypotheses as learning units.

Each FunctionFractal represents a candidate generating function with
learnable coefficients. Instead of opaque weight matrices, these fractals
use known mathematical forms (polynomials, trig) and discover the
coefficients that best explain the data.

The power: if the generating function is guessed correctly, the model is
tiny (a few coefficients) and perfectly accurate.

Composition via residual analysis: fit the dominant pattern first, then
fit the remainder with a second function, and combine them.
"""

import numpy as np
import copy
import time
from typing import Tuple, Optional

from .fractal import Fractal
from .types import Signal, Feedback


# ================================================================
# BASE CLASS
# ================================================================

class FunctionFractal(Fractal):
    """A fractal that represents a mathematical function hypothesis.

    Subclasses Fractal to reuse Memory, Metrics, IDs, children, and
    serialization. Completely overrides process() and learn() to use
    parametric function evaluation instead of linear prediction.
    """

    # Canonical domain for function signature computation.
    # Signatures capture the function's *shape* independent of scale.
    SIGNATURE_POINTS = 32
    SIGNATURE_X = np.linspace(0, 2 * np.pi, SIGNATURE_POINTS)

    def __init__(
        self,
        n_coeffs: int,
        func_name: str,
        learning_rate: float = 0.05,
    ):
        # dim=1: we bypass the base linear prediction entirely.
        # Metrics, Memory, IDs, children all still work.
        super().__init__(dim=1, domain="symbolic", learning_rate=learning_rate)
        self.func_name = func_name
        self.coefficients = np.zeros(n_coeffs)

        # Context set by the engine before process()
        self._x_offset: int = 0
        self.predict_horizon: int = 1

        # Runtime state
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None
        self._y_scale: float = 1.0  # For error normalization

        # Function signature for pattern library similarity search
        self._signature: np.ndarray = np.zeros(self.SIGNATURE_POINTS)

    # --- Signature computation ---

    def compute_signature(self) -> np.ndarray:
        """Compute a fixed-length vector representing this function's shape.

        Evaluates the function at canonical points and normalizes to unit
        length. Two functions with similar shapes (e.g. same frequency,
        different amplitude) produce nearly identical signatures.
        """
        try:
            y = self.evaluate(self.SIGNATURE_X)
            norm = np.linalg.norm(y)
            if norm > 1e-8:
                return y / norm
            return np.zeros(self.SIGNATURE_POINTS)
        except Exception:
            return np.zeros(self.SIGNATURE_POINTS)

    def seed_from(self, other: "FunctionFractal") -> None:
        """Initialize coefficients from another fractal of the same type.

        Used for transfer learning: copy learned coefficients from a
        library pattern into a fresh candidate. The copy is independent —
        mutating one does not affect the other.
        """
        if (
            type(self) == type(other)
            and len(self.coefficients) == len(other.coefficients)
        ):
            self.coefficients = other.coefficients.copy()
            self._signature = self.compute_signature()

    # --- Abstract interface (subclasses must implement) ---

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate f(x) using current coefficients."""
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit coefficients to (x, y) data."""
        raise NotImplementedError

    def symbolic_repr(self) -> str:
        """Return a human-readable formula string."""
        raise NotImplementedError

    # --- Override Fractal.process() ---

    def process(self, input_signal: Signal) -> Tuple[Signal, float]:
        """Forward pass: fit function to window, predict next values.

        input_signal.data = y-values of the current window.
        x-values derived from self._x_offset.
        """
        now = time.time()
        self.metrics.last_active = now
        self.metrics.total_exposures += 1

        y_window = input_signal.data
        n = len(y_window)
        x_window = np.arange(self._x_offset, self._x_offset + n, dtype=float)

        # Track data scale for error normalization
        self._y_scale = max(np.std(y_window), 1e-8)

        # Fit coefficients to the window
        self.fit(x_window, y_window)

        # Update signature for pattern library similarity search
        self._signature = self.compute_signature()

        # Compute window RMSE as novelty
        y_pred_window = self.evaluate(x_window)
        novelty = float(np.sqrt(np.mean((y_window - y_pred_window) ** 2)))

        # Predict future values
        x_future = np.arange(
            self._x_offset + n,
            self._x_offset + n + self.predict_horizon,
            dtype=float,
        )
        prediction = self.evaluate(x_future)

        # Store for learn phase
        self._last_x = x_window
        self._last_y = y_window
        self._last_prediction = prediction

        output = Signal(data=prediction, timestamp=now, source_id=self.id)
        return output, novelty

    # --- Override Fractal.learn() ---

    def learn(self, feedback: Feedback) -> float:
        """Backward pass: compare prediction to actual, update metrics.

        Coefficient updates happen in process()/fit() since we have the
        full window there. Here we just track prediction quality.
        """
        if self._last_prediction is None:
            return 0.0

        actual = feedback.actual
        if len(actual) != len(self._last_prediction):
            actual = actual[: len(self._last_prediction)]

        pred_error = actual - self._last_prediction
        error_magnitude = float(np.sqrt(np.mean(pred_error ** 2)))

        # Normalize by data scale so fitness is comparable across functions
        normalized_error = min(error_magnitude / self._y_scale, 1.0)

        # Update metrics using same EMA as base Fractal
        ema_alpha = 0.1
        self.metrics.prediction_error_ema = (
            (1 - ema_alpha) * self.metrics.prediction_error_ema
            + ema_alpha * normalized_error
        )
        accuracy = max(0.0, 1.0 - normalized_error)
        self.metrics.accuracy_ema = (
            (1 - ema_alpha) * self.metrics.accuracy_ema + ema_alpha * accuracy
        )

        # Clear runtime state
        self._last_prediction = None

        return error_magnitude

    # --- Serialization extension ---

    def compress(self) -> dict:
        base = super().compress()
        base["coefficients"] = self.coefficients.tolist()
        base["func_name"] = self.func_name
        base["func_type"] = type(self).__name__
        base["signature"] = self._signature.tolist()
        return base

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"fitness={self.metrics.fitness():.3f}, "
            f"formula={self.symbolic_repr()})"
        )


# ================================================================
# POLYNOMIAL TYPES (closed-form fitting via np.polyfit)
# ================================================================

class ConstantFractal(FunctionFractal):
    """f(x) = c"""

    def __init__(self, learning_rate: float = 0.05):
        super().__init__(n_coeffs=1, func_name="constant", learning_rate=learning_rate)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.coefficients[0], dtype=float)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.coefficients[0] = np.mean(y)

    def symbolic_repr(self) -> str:
        return f"{self.coefficients[0]:.4f}"


class LinearFractal(FunctionFractal):
    """f(x) = ax + b"""

    def __init__(self, learning_rate: float = 0.05):
        super().__init__(n_coeffs=2, func_name="linear", learning_rate=learning_rate)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b = self.coefficients
        return a * x + b

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) >= 2:
            self.coefficients = np.polyfit(x, y, 1)

    def symbolic_repr(self) -> str:
        a, b = self.coefficients
        return f"{a:.4f}*x + {b:.4f}"


class QuadraticFractal(FunctionFractal):
    """f(x) = ax^2 + bx + c"""

    def __init__(self, learning_rate: float = 0.05):
        super().__init__(n_coeffs=3, func_name="quadratic", learning_rate=learning_rate)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.coefficients
        return a * x ** 2 + b * x + c

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) >= 3:
            self.coefficients = np.polyfit(x, y, 2)

    def symbolic_repr(self) -> str:
        a, b, c = self.coefficients
        return f"{a:.4f}*x^2 + {b:.4f}*x + {c:.4f}"


class PolynomialFractal(FunctionFractal):
    """f(x) = a_n*x^n + ... + a_1*x + a_0"""

    def __init__(self, degree: int, learning_rate: float = 0.05):
        super().__init__(
            n_coeffs=degree + 1,
            func_name=f"poly(deg={degree})",
            learning_rate=learning_rate,
        )
        self.degree = degree

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.coefficients, x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) >= self.degree + 1:
            self.coefficients = np.polyfit(x, y, self.degree)

    def symbolic_repr(self) -> str:
        parts = []
        degree = self.degree
        for i, c in enumerate(self.coefficients):
            power = degree - i
            if power == 0:
                parts.append(f"{c:.4f}")
            elif power == 1:
                parts.append(f"{c:.4f}*x")
            else:
                parts.append(f"{c:.4f}*x^{power}")
        return " + ".join(parts)


# ================================================================
# TRIGONOMETRIC TYPES (FFT + linear regression fitting)
# ================================================================

def _estimate_frequency_fft(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate dominant angular frequency using zero-padded FFT.

    Zero-padding increases frequency resolution without needing
    more data points. With n_padded=512, resolution is ~0.012 rad.
    """
    centered = y - np.mean(y)
    n = len(centered)
    dt = x[1] - x[0] if n > 1 else 1.0

    # Zero-pad for high frequency resolution
    n_padded = max(512, n * 8)
    fft_vals = np.fft.rfft(centered, n=n_padded)
    freqs = np.fft.rfftfreq(n_padded, d=dt)
    magnitudes = np.abs(fft_vals[1:])  # Skip DC

    if len(magnitudes) > 0 and np.max(magnitudes) > 1e-8:
        peak_idx = np.argmax(magnitudes) + 1  # +1 for skipped DC
        return 2.0 * np.pi * freqs[peak_idx]
    return 2.0 * np.pi / max(n, 1)


class SinFractal(FunctionFractal):
    """f(x) = a*sin(b*x + c) + d

    Fitting strategy:
      1. Estimate frequency b from FFT peak
      2. Rewrite as y = A*sin(bx) + B*cos(bx) + d (linear in A, B, d)
      3. Solve via least squares (exact, no gradient descent)
      4. Recover a = sqrt(A^2 + B^2), c = atan2(B, A)
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(n_coeffs=4, func_name="sin", learning_rate=learning_rate)
        self.coefficients = np.array([1.0, 1.0, 0.0, 0.0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b, c, d = self.coefficients
        return a * np.sin(b * x + c) + d

    def _estimate_frequency(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate dominant frequency using zero-padded FFT."""
        return _estimate_frequency_fft(x, y)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 8:
            return

        # Step 1: Estimate frequency from FFT
        b = self._estimate_frequency(x, y)
        if b < 1e-6:
            b = 0.01

        # Step 2: Linear regression for A, B, d
        # y = A*sin(bx) + B*cos(bx) + d
        sin_bx = np.sin(b * x)
        cos_bx = np.cos(b * x)
        ones = np.ones_like(x)
        X = np.column_stack([sin_bx, cos_bx, ones])

        # Solve via least squares
        result = np.linalg.lstsq(X, y, rcond=None)
        A, B, d = result[0]

        # Step 3: Recover amplitude and phase
        a = np.sqrt(A ** 2 + B ** 2)
        c = np.arctan2(B, A)

        if a < 1e-10:
            a = 1e-10

        self.coefficients = np.array([a, b, c, d])

    def symbolic_repr(self) -> str:
        a, b, c, d = self.coefficients
        return f"{a:.4f}*sin({b:.4f}*x + {c:.4f}) + {d:.4f}"


class CosFractal(FunctionFractal):
    """f(x) = a*cos(b*x + c) + d

    Same FFT + linear regression strategy as SinFractal.
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(n_coeffs=4, func_name="cos", learning_rate=learning_rate)
        self.coefficients = np.array([1.0, 1.0, 0.0, 0.0])

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b, c, d = self.coefficients
        return a * np.cos(b * x + c) + d

    def _estimate_frequency(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate dominant frequency using zero-padded FFT."""
        return _estimate_frequency_fft(x, y)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 8:
            return

        # Step 1: Estimate frequency from FFT
        b = self._estimate_frequency(x, y)
        if b < 1e-6:
            b = 0.01

        # Step 2: Linear regression
        # y = A*cos(bx) + B*sin(bx) + d
        cos_bx = np.cos(b * x)
        sin_bx = np.sin(b * x)
        ones = np.ones_like(x)
        X = np.column_stack([cos_bx, sin_bx, ones])

        result = np.linalg.lstsq(X, y, rcond=None)
        A, B, d = result[0]

        # Step 3: Recover amplitude and phase
        # a*cos(bx+c) = a*cos(c)*cos(bx) - a*sin(c)*sin(bx)
        # So A = a*cos(c), B = -a*sin(c)
        a = np.sqrt(A ** 2 + B ** 2)
        c = np.arctan2(-B, A)

        if a < 1e-10:
            a = 1e-10

        self.coefficients = np.array([a, b, c, d])

    def symbolic_repr(self) -> str:
        a, b, c, d = self.coefficients
        return f"{a:.4f}*cos({b:.4f}*x + {c:.4f}) + {d:.4f}"


# ================================================================
# GRADIENT-BASED TRIGONOMETRIC TYPES (warm-start fitting)
# ================================================================

class GradientSinFractal(SinFractal):
    """f(x) = a*sin(b*x + c) + d — fitted via gradient descent.

    Unlike SinFractal (FFT+lstsq, recomputes from scratch), this uses
    gradient descent starting from current coefficients. When seeded
    with learned coefficients via seed_from(), it refines rather than
    restarts — enabling true transfer learning.

    First call uses FFT+lstsq for good initialization; subsequent
    calls use gradient descent from the current state.
    """

    def __init__(self, learning_rate: float = 0.005):
        super().__init__(learning_rate=learning_rate)
        self.func_name = "grad_sin"
        self._fit_count: int = 0
        self._fit_iterations: int = 15
        self._grad_clip: float = 5.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 8:
            return

        self._fit_count += 1

        # First call: use FFT+lstsq for good initialization
        if self._fit_count == 1 and np.allclose(self.coefficients, [1.0, 1.0, 0.0, 0.0]):
            super().fit(x, y)
            return

        # Gradient descent from current coefficients
        a, b, c, d = self.coefficients
        lr = self.learning_rate
        n = len(x)

        for _ in range(self._fit_iterations):
            inner = b * x + c
            sin_val = np.sin(inner)
            cos_val = np.cos(inner)
            residual = (a * sin_val + d) - y

            # Analytical gradients
            da = np.sum(residual * sin_val) / n
            db = np.sum(residual * a * x * cos_val) / n
            dc = np.sum(residual * a * cos_val) / n
            dd = np.sum(residual) / n

            # Gradient clipping
            for grad in [da, db, dc, dd]:
                grad = np.clip(grad, -self._grad_clip, self._grad_clip)
            da = np.clip(da, -self._grad_clip, self._grad_clip)
            db = np.clip(db, -self._grad_clip, self._grad_clip)
            dc = np.clip(dc, -self._grad_clip, self._grad_clip)
            dd = np.clip(dd, -self._grad_clip, self._grad_clip)

            a -= lr * da
            b -= lr * db
            c -= lr * dc
            d -= lr * dd

        if abs(a) < 1e-10:
            a = 1e-10

        self.coefficients = np.array([a, b, c, d])


class GradientCosFractal(CosFractal):
    """f(x) = a*cos(b*x + c) + d — fitted via gradient descent.

    Warm-start variant of CosFractal, analogous to GradientSinFractal.
    """

    def __init__(self, learning_rate: float = 0.005):
        super().__init__(learning_rate=learning_rate)
        self.func_name = "grad_cos"
        self._fit_count: int = 0
        self._fit_iterations: int = 15
        self._grad_clip: float = 5.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 8:
            return

        self._fit_count += 1

        # First call: use FFT+lstsq for good initialization
        if self._fit_count == 1 and np.allclose(self.coefficients, [1.0, 1.0, 0.0, 0.0]):
            super().fit(x, y)
            return

        # Gradient descent from current coefficients
        a, b, c, d = self.coefficients
        lr = self.learning_rate
        n = len(x)

        for _ in range(self._fit_iterations):
            inner = b * x + c
            cos_val = np.cos(inner)
            sin_val = np.sin(inner)
            residual = (a * cos_val + d) - y

            # Analytical gradients
            da = np.sum(residual * cos_val) / n
            db = np.sum(residual * (-a * x * sin_val)) / n
            dc = np.sum(residual * (-a * sin_val)) / n
            dd = np.sum(residual) / n

            # Gradient clipping
            da = np.clip(da, -self._grad_clip, self._grad_clip)
            db = np.clip(db, -self._grad_clip, self._grad_clip)
            dc = np.clip(dc, -self._grad_clip, self._grad_clip)
            dd = np.clip(dd, -self._grad_clip, self._grad_clip)

            a -= lr * da
            b -= lr * db
            c -= lr * dc
            d -= lr * dd

        if abs(a) < 1e-10:
            a = 1e-10

        self.coefficients = np.array([a, b, c, d])


# ================================================================
# EXOTIC FUNCTION TYPES (gradient-based fitting)
# ================================================================

class ExponentialFractal(FunctionFractal):
    """f(x) = a*exp(b*x) + c

    Gradient descent fitting with overflow protection.
    First call uses log-transform lstsq for initialization.
    """

    def __init__(self, learning_rate: float = 0.002):
        super().__init__(n_coeffs=3, func_name="exp", learning_rate=learning_rate)
        self.coefficients = np.array([1.0, 0.01, 0.0])
        self._fit_count: int = 0
        self._fit_iterations: int = 30
        self._grad_clip: float = 2.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.coefficients
        bx = np.clip(b * x, -50, 50)
        return a * np.exp(bx) + c

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 4:
            return

        self._fit_count += 1

        # First call: try log-transform initialization
        if self._fit_count == 1 and np.allclose(self.coefficients, [1.0, 0.01, 0.0]):
            self._init_log_transform(x, y)

        # Gradient descent from current coefficients
        a, b, c = self.coefficients
        lr = self.learning_rate
        n = len(x)

        for _ in range(self._fit_iterations):
            bx = np.clip(b * x, -50, 50)
            exp_bx = np.exp(bx)
            residual = (a * exp_bx + c) - y

            # Analytical gradients
            da = np.sum(residual * exp_bx) / n
            db = np.sum(residual * a * x * exp_bx) / n
            dc = np.sum(residual) / n

            # Gradient clipping
            da = np.clip(da, -self._grad_clip, self._grad_clip)
            db = np.clip(db, -self._grad_clip, self._grad_clip)
            dc = np.clip(dc, -self._grad_clip, self._grad_clip)

            a -= lr * da
            b -= lr * db
            c -= lr * dc

            # Clamp b to prevent runaway exponentials
            b = np.clip(b, -2.0, 2.0)

        self.coefficients = np.array([a, b, c])

    def _init_log_transform(self, x: np.ndarray, y: np.ndarray) -> None:
        """Try log-transform lstsq for initial coefficient estimate."""
        try:
            y_shifted = y - np.min(y) + 1e-6
            log_y = np.log(y_shifted)
            if len(x) >= 2:
                coeffs = np.polyfit(x, log_y, 1)
                b = np.clip(coeffs[0], -2.0, 2.0)
                a = np.exp(coeffs[1])
                c = np.min(y) - 1e-6
                self.coefficients = np.array([a, b, c])
        except Exception:
            pass

    def symbolic_repr(self) -> str:
        a, b, c = self.coefficients
        return f"{a:.4f}*exp({b:.4f}*x) + {c:.4f}"


class LogFractal(FunctionFractal):
    """f(x) = a*log(b*x + c) + d

    Gradient descent fitting with domain safety (inner > 0).
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(n_coeffs=4, func_name="log", learning_rate=learning_rate)
        self.coefficients = np.array([1.0, 1.0, 1.0, 0.0])
        self._fit_count: int = 0
        self._fit_iterations: int = 30
        self._grad_clip: float = 5.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        a, b, c, d = self.coefficients
        inner = np.maximum(b * x + c, 1e-8)
        return a * np.log(inner) + d

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) < 4:
            return

        self._fit_count += 1

        # Gradient descent from current coefficients
        a, b, c, d = self.coefficients
        lr = self.learning_rate
        n = len(x)

        for _ in range(self._fit_iterations):
            inner = np.maximum(b * x + c, 1e-8)
            log_inner = np.log(inner)
            residual = (a * log_inner + d) - y

            # Analytical gradients
            da = np.sum(residual * log_inner) / n
            db = np.sum(residual * a * x / inner) / n
            dc = np.sum(residual * a / inner) / n
            dd = np.sum(residual) / n

            # Gradient clipping
            da = np.clip(da, -self._grad_clip, self._grad_clip)
            db = np.clip(db, -self._grad_clip, self._grad_clip)
            dc = np.clip(dc, -self._grad_clip, self._grad_clip)
            dd = np.clip(dd, -self._grad_clip, self._grad_clip)

            a -= lr * da
            b -= lr * db
            c -= lr * dc
            d -= lr * dd

            # Ensure inner stays positive for all x in typical range
            if b * 0 + c < 1e-8:
                c = max(c, 1e-4)

        self.coefficients = np.array([a, b, c, d])

    def symbolic_repr(self) -> str:
        a, b, c, d = self.coefficients
        return f"{a:.4f}*log({b:.4f}*x + {c:.4f}) + {d:.4f}"


# ================================================================
# COMPOSITION
# ================================================================

class ComposedFunctionFractal(FunctionFractal):
    """A function built by combining two child functions.

    Supports addition, subtraction, multiplication, and division.
    Uses the base Fractal's add_child() mechanism, so is_composed=True
    and the existing Memory/hierarchy infrastructure works.
    """

    def __init__(
        self,
        child1: FunctionFractal,
        child2: FunctionFractal,
        operation: str = "add",
    ):
        combined_name = f"({child1.func_name} {operation} {child2.func_name})"
        combined_coeffs = len(child1.coefficients) + len(child2.coefficients)
        super().__init__(
            n_coeffs=combined_coeffs,
            func_name=combined_name,
            learning_rate=max(child1.learning_rate, child2.learning_rate),
        )
        self.operation = operation
        self.child1 = child1
        self.child2 = child2

        # Register as children in the Fractal hierarchy
        self.add_child(child1)
        self.add_child(child2)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y1 = self.child1.evaluate(x)
        y2 = self.child2.evaluate(x)
        if self.operation == "add":
            return y1 + y2
        elif self.operation == "subtract":
            return y1 - y2
        elif self.operation == "multiply":
            return y1 * y2
        elif self.operation == "divide":
            # Safe division: clamp denominator away from zero
            safe_y2 = np.where(
                np.abs(y2) < 1e-8,
                np.sign(y2 + 1e-16) * 1e-8,
                y2,
            )
            return np.clip(y1 / safe_y2, -1e6, 1e6)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Children were already fitted during residual analysis.
        # On subsequent calls, re-fit children to the data.
        if self.operation == "add":
            # Fit child1 to data, then child2 to residuals
            self.child1.fit(x, y)
            residual = y - self.child1.evaluate(x)
            self.child2.fit(x, residual)
        elif self.operation == "subtract":
            # y = f(x) - g(x)  =>  g(x) = f(x) - y
            self.child1.fit(x, y)
            neg_residual = self.child1.evaluate(x) - y
            self.child2.fit(x, neg_residual)
        elif self.operation == "multiply":
            # Keep children's existing coefficients (multiplicative
            # decomposition is harder to re-fit incrementally)
            pass
        elif self.operation == "divide":
            # Division decomposition is ill-conditioned; keep existing
            pass

    def learn(self, feedback: Feedback) -> float:
        """Update metrics and propagate to children."""
        error = super().learn(feedback)
        return error

    def symbolic_repr(self) -> str:
        op_map = {"add": " + ", "subtract": " - ", "multiply": " * ", "divide": " / "}
        op_str = op_map.get(self.operation, f" {self.operation} ")
        return f"({self.child1.symbolic_repr()}{op_str}{self.child2.symbolic_repr()})"

    def compress(self) -> dict:
        base = super().compress()
        base["operation"] = self.operation
        base["child1_data"] = self.child1.compress()
        base["child2_data"] = self.child2.compress()
        return base
