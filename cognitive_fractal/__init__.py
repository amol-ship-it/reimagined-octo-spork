"""
cognitive_fractal — A recursive learning system built from self-similar fractals.

Each Fractal is an active learning agent that compares, predicts, and updates.
The same mechanism operates at every level of the hierarchy.
"""

from .fractal import Fractal
from .engine import FractalEngine
from .memory import Memory
from .types import Signal, Feedback, Metrics
from .compare import deviation, deviation_magnitude, ratio, similarity

__all__ = [
    "Fractal",
    "FractalEngine",
    "Memory",
    "Signal",
    "Feedback",
    "Metrics",
    "deviation",
    "deviation_magnitude",
    "ratio",
    "similarity",
]
