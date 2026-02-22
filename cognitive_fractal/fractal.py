"""
The Fractal — the atomic learning unit of the recursive learning system.

This is NOT a data container. It is an active agent that:
  1. Holds a prototype (what it "expects" — shaped by experience)
  2. Compares input to that prototype via subtraction
  3. Generates a prediction using learned weights
  4. Updates itself from feedback (prototype EMA + variance + weight gradient)
  5. Composes with other fractals — same code at every level

A leaf fractal operates on raw input.
A composed fractal routes input through its children first,
then operates on their concatenated outputs.
The class is identical in both cases — self-similar by construction.
"""

import numpy as np
import uuid
import time
from typing import Optional, List, Tuple

from .types import Signal, Feedback, Metrics
from .compare import deviation, deviation_magnitude


class Fractal:
    """The smallest self-contained learning unit."""

    def __init__(
        self, dim: int, domain: str = "default", learning_rate: float = 0.1
    ):
        # --- Identity ---
        self.id: str = str(uuid.uuid4())[:8]
        self.domain: str = domain
        self.dim: int = dim

        # --- Learned State (Type A memory: compressed, reusable pattern) ---
        self.prototype: np.ndarray = np.zeros(dim)
        self.variance: np.ndarray = np.ones(dim)
        self.prediction_weights: np.ndarray = np.eye(dim) * 0.01
        self._prediction_bias: np.ndarray = np.zeros(dim)
        # Starts near-zero: "I don't know anything yet"

        # --- Hyperparameters ---
        self.learning_rate: float = learning_rate
        self.variance_rate: float = 0.05

        # --- Composition ---
        self.children: List["Fractal"] = []
        self.parent: Optional["Fractal"] = None

        # --- Metrics ---
        self.metrics: Metrics = Metrics()

        # --- Runtime state (cleared after each learn() call) ---
        self._last_input: Optional[np.ndarray] = None
        self._last_deviation: Optional[np.ndarray] = None
        self._last_prediction: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_composed(self) -> bool:
        return len(self.children) > 0

    # ================================================================
    # CORE OPERATION 1: PROCESS (forward pass)
    # ================================================================

    def process(self, input_signal: Signal) -> Tuple[Signal, float]:
        """Forward pass: input -> compare to prototype -> produce prediction.

        For a LEAF fractal:
            Input is raw data. Compare directly to prototype.

        For a COMPOSED fractal:
            Route input through children first.
            Children's outputs are aggregated into a feature vector.
            Compare THAT to this fractal's own prototype.

        Returns:
            (output_signal, novelty_score)
        """
        now = time.time()
        self.metrics.last_active = now
        self.metrics.total_exposures += 1

        if self.is_composed:
            # Process through children, build feature vector
            child_outputs = []
            for child in self.children:
                child_out, _ = child.process(input_signal)
                child_outputs.append(child_out.data)
            effective_input = self._aggregate_child_outputs(child_outputs)
        else:
            # Leaf: use raw input directly
            effective_input = self._conform_input(input_signal.data)

        # Compare to prototype
        dev = deviation(effective_input, self.prototype)
        novelty = deviation_magnitude(effective_input, self.prototype)

        # Generate prediction: what do I think comes next?
        # Uses the full input (not deviation) so prediction doesn't collapse
        # as prototype converges. The weights learn: "given THIS input,
        # what NEXT input follows?"
        prediction = self.prediction_weights @ effective_input + self._prediction_bias

        # Store for feedback phase
        self._last_input = effective_input
        self._last_deviation = dev
        self._last_prediction = prediction

        output = Signal(data=prediction, timestamp=now, source_id=self.id)
        return output, novelty

    # ================================================================
    # CORE OPERATION 2: LEARN (feedback pass)
    # ================================================================

    def learn(self, feedback: Feedback) -> float:
        """Backward pass: receive outcome, compute error, update model.

        This is ACTUAL learning — not just confidence adjustment:
          1. Prototype moves toward the input (EMA update)
          2. Variance tracks the spread of inputs seen
          3. Prediction weights adjust to reduce prediction error
             (outer-product gradient rule: dW = lr * error * input^T)
          4. Metrics update with actual accuracy

        Returns the prediction error (scalar).
        """
        if self._last_input is None:
            return 0.0

        actual = self._conform_input(feedback.actual)

        # --- Update 1: Move prototype toward recent input (EMA) ---
        proto_error = self._last_input - self.prototype
        self.prototype += self.learning_rate * proto_error

        # --- Update 2: Track variance (how spread out are inputs?) ---
        var_error = (self._last_input - self.prototype) ** 2 - self.variance
        self.variance += self.variance_rate * var_error
        self.variance = np.maximum(self.variance, 1e-6)  # Floor

        # --- Update 3: Adjust prediction weights (gradient step) ---
        pred_error = actual - self._last_prediction
        error_magnitude = float(np.sqrt(np.mean(pred_error**2)))

        # Outer product learning rule: dW = lr * error * input^T
        # Uses the full input vector so learning doesn't vanish as
        # prototype converges (deviation → 0 would kill gradients).
        input_norm = np.linalg.norm(self._last_input)
        if input_norm > 1e-8:
            normalized_input = self._last_input / input_norm
            weight_update = self.learning_rate * np.outer(
                pred_error, normalized_input
            )
            self.prediction_weights += weight_update
            self._prediction_bias += self.learning_rate * pred_error

        # --- Update 4: Track accuracy metrics ---
        ema_alpha = 0.1
        self.metrics.prediction_error_ema = (
            1 - ema_alpha
        ) * self.metrics.prediction_error_ema + ema_alpha * min(
            error_magnitude, 1.0
        )
        accuracy = max(0.0, 1.0 - error_magnitude)
        self.metrics.accuracy_ema = (
            1 - ema_alpha
        ) * self.metrics.accuracy_ema + ema_alpha * accuracy

        # --- Propagate feedback to children ---
        if self.is_composed:
            for child in self.children:
                child.learn(feedback)

        # Clear runtime state
        self._last_input = None
        self._last_deviation = None
        self._last_prediction = None

        return error_magnitude

    # ================================================================
    # COMPOSITION INTERFACE
    # ================================================================

    def add_child(self, child: "Fractal"):
        """Attach a child fractal. This fractal becomes composed."""
        child.parent = self
        self.children.append(child)

    def _aggregate_child_outputs(
        self, child_outputs: List[np.ndarray]
    ) -> np.ndarray:
        """Combine children's output vectors into a single feature vector
        matching this fractal's dimensionality.

        Strategy: concatenate then project via chunked averaging (downsample)
        or zero-pad (upsample).
        """
        concatenated = np.concatenate(child_outputs)
        if len(concatenated) == self.dim:
            return concatenated
        elif len(concatenated) > self.dim:
            # Downsample: chunked averaging
            indices = np.array_split(np.arange(len(concatenated)), self.dim)
            return np.array([concatenated[idx].mean() for idx in indices])
        else:
            # Pad with zeros
            result = np.zeros(self.dim)
            result[: len(concatenated)] = concatenated
            return result

    def _conform_input(self, data: np.ndarray) -> np.ndarray:
        """Pad or truncate input to match this fractal's dimensionality."""
        if len(data) == self.dim:
            return data.copy()
        elif len(data) > self.dim:
            return data[: self.dim].copy()
        else:
            result = np.zeros(self.dim)
            result[: len(data)] = data
            return result

    # ================================================================
    # SERIALIZATION (for memory tiering)
    # ================================================================

    def compress(self) -> dict:
        """Serialize to a compact dictionary for cold storage.

        This IS the Type A memory: the learned pattern, stripped of
        runtime state and live object references.
        """
        return {
            "id": self.id,
            "domain": self.domain,
            "dim": self.dim,
            "prototype": self.prototype.tolist(),
            "variance": self.variance.tolist(),
            "prediction_weights": self.prediction_weights.tolist(),
            "prediction_bias": self._prediction_bias.tolist(),
            "learning_rate": self.learning_rate,
            "variance_rate": self.variance_rate,
            "metrics": {
                "total_exposures": self.metrics.total_exposures,
                "prediction_error_ema": self.metrics.prediction_error_ema,
                "accuracy_ema": self.metrics.accuracy_ema,
                "last_active": self.metrics.last_active,
            },
            "child_ids": [c.id for c in self.children],
        }

    @classmethod
    def decompress(cls, data: dict) -> "Fractal":
        """Restore a fractal from its compressed representation.

        Note: children must be re-linked by the caller after decompression.
        """
        f = cls(
            dim=data["dim"],
            domain=data["domain"],
            learning_rate=data["learning_rate"],
        )
        f.id = data["id"]
        f.variance_rate = data.get("variance_rate", 0.05)
        f.prototype = np.array(data["prototype"])
        f.variance = np.array(data["variance"])
        f.prediction_weights = np.array(data["prediction_weights"])
        f._prediction_bias = np.array(data.get("prediction_bias", np.zeros(data["dim"]).tolist()))
        f.metrics.total_exposures = data["metrics"]["total_exposures"]
        f.metrics.prediction_error_ema = data["metrics"]["prediction_error_ema"]
        f.metrics.accuracy_ema = data["metrics"]["accuracy_ema"]
        f.metrics.last_active = data["metrics"].get("last_active", 0.0)
        return f

    def __repr__(self):
        kind = "Composed" if self.is_composed else "Leaf"
        return (
            f"Fractal({kind}, id={self.id}, domain={self.domain}, "
            f"dim={self.dim}, fitness={self.metrics.fitness():.3f}, "
            f"exposures={self.metrics.total_exposures})"
        )
