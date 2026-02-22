"""
Shared data structures for the cognitive fractal system.

Signal flows forward through the hierarchy.
Feedback flows backward.
Metrics track how well each fractal is performing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """A timestamped input or output flowing through the fractal hierarchy."""

    data: np.ndarray  # The raw vector (any dimensionality)
    timestamp: float  # When this signal was produced
    source_id: Optional[str] = None  # Which fractal produced it (None = external)


@dataclass
class Feedback:
    """Outcome information flowing backward through the hierarchy."""

    actual: np.ndarray  # What actually happened
    reward: float  # Scalar reward signal (-1 to 1)
    timestamp: float


@dataclass
class Metrics:
    """Tracked performance of a fractal."""

    total_exposures: int = 0
    prediction_error_ema: float = 1.0  # Exponential moving average of error
    accuracy_ema: float = 0.0  # Exponential moving average of accuracy
    last_active: float = 0.0

    def fitness(self) -> float:
        """Single scalar: how good is this fractal at its job?

        Combines accuracy (high is good) with low prediction error.
        Range: [0, 1]. Higher = better.
        """
        return self.accuracy_ema * (1.0 - self.prediction_error_ema)
