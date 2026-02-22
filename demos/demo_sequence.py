#!/usr/bin/env python3
"""
Demo: Learn a repeating sequence A-B-C-A-B-C...

Each element is encoded as a one-hot vector in 8 dimensions:
  A = [1, 0, 0, 0, 0, 0, 0, 0]
  B = [0, 1, 0, 0, 0, 0, 0, 0]
  C = [0, 0, 1, 0, 0, 0, 0, 0]

The system should learn:
  After A, predict B
  After B, predict C
  After C, predict A

We measure prediction error over time. It MUST decrease.
This is proof of actual learning — not random fluctuation.
"""

import sys
import os
import numpy as np

# Add parent directory to path so we can import cognitive_fractal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_fractal import FractalEngine


def encode(symbol: str) -> np.ndarray:
    """One-hot encode a symbol."""
    mapping = {"A": 0, "B": 1, "C": 2}
    vec = np.zeros(8)
    vec[mapping[symbol]] = 1.0
    return vec


def decode(vec: np.ndarray) -> str:
    """Decode a vector to the nearest symbol."""
    mapping = {0: "A", 1: "B", 2: "C"}
    idx = int(np.argmax(vec[:3]))
    return mapping[idx]


def main():
    print("=" * 60)
    print("RECURSIVE LEARNING SYSTEM — SEQUENCE PREDICTION DEMO")
    print("=" * 60)
    print()
    print("Task: Learn to predict A -> B -> C -> A -> B -> C -> ...")
    print("Method: Bottom-up fractal comparison + prototype learning")
    print()

    engine = FractalEngine(
        dim=8,
        novelty_threshold=0.7,
        learning_rate=0.15,
    )

    sequence = ["A", "B", "C"] * 100  # 300 steps
    window_size = 30

    errors = []
    correct_count = 0
    total_count = 0
    expected_next = {"A": "B", "B": "C", "C": "A"}

    prev_symbol = None
    prev_prediction = None

    for i, symbol in enumerate(sequence):
        raw = encode(symbol)
        prediction, diag = engine.step(raw)

        # Check if previous prediction was correct
        if prev_symbol is not None and prev_prediction is not None:
            predicted_symbol = decode(prev_prediction)
            actual_symbol = symbol
            total_count += 1
            if predicted_symbol == actual_symbol:
                correct_count += 1

        errors.append(diag["prediction_error"])
        prev_prediction = prediction
        prev_symbol = symbol

        # Report every window_size steps
        if (i + 1) % window_size == 0:
            avg_err = np.mean(errors[-window_size:])
            recent_acc = (
                correct_count / total_count * 100 if total_count > 0 else 0
            )
            print(
                f"  Steps {i + 1 - window_size + 1:3d}-{i + 1:3d}: "
                f"avg_error={avg_err:.4f}  "
                f"accuracy={recent_acc:5.1f}%  "
                f"fitness={diag['fitness']:.4f}  "
                f"fractals={diag['active_fractals']}"
            )

    # Final assessment
    early_error = np.mean(errors[3:33])  # Skip first 3 (no feedback)
    late_error = np.mean(errors[-30:])
    improvement = (early_error - late_error) / (early_error + 1e-8) * 100
    final_accuracy = correct_count / total_count * 100 if total_count > 0 else 0

    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Early avg error (steps 4-33):     {early_error:.4f}")
    print(f"  Late avg error  (steps 271-300):  {late_error:.4f}")
    print(f"  Improvement:                      {improvement:.1f}%")
    print(f"  Final prediction accuracy:        {final_accuracy:.1f}%")
    print(f"  Active fractals:                  {diag['active_fractals']}")
    print(f"  Total steps:                      {engine.get_stats()['step_count']}")
    print()

    # Inspect the learned fractals
    print("LEARNED FRACTALS:")
    for fid, frac in engine.memory.hot.items():
        proto_label = decode(frac.prototype)
        print(
            f"  {frac.id}: prototype~{proto_label}  "
            f"fitness={frac.metrics.fitness():.3f}  "
            f"exposures={frac.metrics.total_exposures}"
        )
    print()

    if late_error < early_error:
        print("SUCCESS: System learned to predict the sequence.")
    else:
        print("FAILURE: System did not learn.")
        sys.exit(1)


if __name__ == "__main__":
    main()
