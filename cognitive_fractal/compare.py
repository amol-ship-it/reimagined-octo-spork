"""
Comparison operations — the mathematical foundation of the learning system.

Two modes of comparison, directly from the theoretical framework:
  - SUBTRACTION finds edges (differences): "where does input differ from expectation?"
  - DIVISION finds ratios (relationships): "how does input scale relative to expectation?"

Plus cosine similarity for memory indexing against learned prototypes.
"""

import numpy as np


def deviation(input_vec: np.ndarray, prototype: np.ndarray) -> np.ndarray:
    """Subtraction-based comparison: find edges/differences.

    Returns the signed deviation vector.
    """
    return input_vec - prototype


def deviation_magnitude(input_vec: np.ndarray, prototype: np.ndarray) -> float:
    """Scalar novelty score: RMS deviation, dimensionality-invariant.

    A 100-dim fractal and a 3-dim fractal produce comparable scores.
    """
    diff = input_vec - prototype
    return float(np.sqrt(np.mean(diff**2)))


def ratio(
    input_vec: np.ndarray, prototype: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Division-based comparison: find scaling relationships.

    Returns element-wise ratio, safe against division by zero.
    """
    safe_proto = np.where(np.abs(prototype) < epsilon, epsilon, prototype)
    return input_vec / safe_proto


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Used for memory indexing against actual learned prototypes.
    Returns value in [-1, 1]. Returns 0.0 for zero-norm vectors.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
