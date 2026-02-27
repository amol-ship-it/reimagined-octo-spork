#!/usr/bin/env python3
"""
Demo: Sequence Predictor MVP

Two test streams:
  1. Quadratic: y = 0.5*t^2 - 3*t + 10  (should be trivially discovered)
  2. Mandelbrot orbit magnitudes |z_n| for z_{n+1} = z_n^2 + c

Shows the SequencePredictor API and prediction quality.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_fractal.predictor import SequencePredictor


def mandelbrot_sequence(c: complex, n_terms: int) -> list:
    """Generate |z_n| for z_{n+1} = z_n^2 + c, z_0 = 0."""
    z = complex(0, 0)
    magnitudes = [abs(z)]
    for _ in range(n_terms - 1):
        z = z * z + c
        magnitudes.append(abs(z))
        if abs(z) > 1e10:
            break
    return magnitudes


def scenario_quadratic():
    """Discover y = 0.5*t^2 - 3*t + 10 and predict future values."""
    print("-" * 64)
    print("  SCENARIO 1: QUADRATIC SEQUENCE")
    print("  y = 0.5*t^2 - 3*t + 10")
    print("-" * 64)
    print()

    sp = SequencePredictor(window_size=25, predict_ahead=5)

    for t in range(200):
        y = 0.5 * t ** 2 - 3.0 * t + 10.0
        result = sp.feed(y)
        if (t + 1) % 50 == 0:
            acc = sp.accuracy()
            formula_short = acc["best_formula"][:60]
            print(f"  Step {t+1:4d}: {formula_short}")
            print(f"            fitness={acc['best_fitness']:.4f}  "
                  f"recent_mae={acc['recent_mae']:.6f}")

    # Future predictions
    preds = sp.predict(5)
    actuals = np.array([0.5 * t ** 2 - 3.0 * t + 10.0 for t in range(200, 205)])
    errors = np.abs(preds - actuals)

    print()
    print(f"  FUTURE PREDICTIONS (next 5 values):")
    print(f"  {'t':>6s}  {'Predicted':>14s}  {'Actual':>14s}  {'Error':>12s}")
    print(f"  {'-' * 50}")
    for i in range(5):
        print(f"  {200+i:6d}  {preds[i]:14.4f}  {actuals[i]:14.4f}  {errors[i]:12.6f}")

    mae = np.mean(errors)
    print(f"\n  Mean Absolute Error: {mae:.6f}")

    acc = sp.accuracy()
    passed = acc["best_fitness"] > 0.2 and mae < 100.0
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def scenario_mandelbrot_bounded():
    """Predict Mandelbrot orbit for c = -0.75 (bounded, near boundary)."""
    print("-" * 64)
    print("  SCENARIO 2: MANDELBROT ORBIT (BOUNDED)")
    print("  z_{n+1} = z_n^2 + c,  c = -0.75")
    print("-" * 64)
    print()

    c = -0.75
    n_terms = 200
    seq = mandelbrot_sequence(c, n_terms)

    print(f"  Sequence length: {len(seq)}")
    print(f"  First 10: [{', '.join(f'{v:.4f}' for v in seq[:10])}]")
    print(f"  Range: [{min(seq):.4f}, {max(seq):.4f}]")
    print()

    sp = SequencePredictor(window_size=30, predict_ahead=5)

    for i, v in enumerate(seq):
        result = sp.feed(v)
        if (i + 1) % 50 == 0:
            acc = sp.accuracy()
            formula_short = acc["best_formula"][:60]
            print(f"  Step {i+1:4d}: {formula_short}")
            print(f"            fitness={acc['best_fitness']:.4f}  "
                  f"recent_mae={acc['recent_mae']:.6f}")

    acc = sp.accuracy()
    print()
    print(f"  FINAL RESULTS:")
    print(f"    Formula:    {acc['best_formula'][:80]}")
    print(f"    Fitness:    {acc['best_fitness']:.4f}")
    print(f"    MAE:        {acc['mean_absolute_error']:.6f}")
    print(f"    Recent MAE: {acc['recent_mae']:.6f}")
    print(f"    Candidates: {acc['num_candidates']}")

    # Predict next 5 and compare
    more_seq = mandelbrot_sequence(c, n_terms + 5)
    if len(more_seq) >= n_terms + 5:
        future_actual = np.array(more_seq[n_terms:n_terms + 5])
        future_pred = sp.predict(5)
        future_errors = np.abs(future_pred - future_actual)
        print()
        print(f"  FUTURE PREDICTIONS:")
        print(f"  {'n':>6s}  {'Predicted':>14s}  {'Actual':>14s}  {'Error':>12s}")
        print(f"  {'-' * 50}")
        for i in range(5):
            print(f"  {n_terms+i:6d}  {future_pred[i]:14.6f}  "
                  f"{future_actual[i]:14.6f}  {future_errors[i]:12.6f}")
        mae = np.mean(future_errors)
        print(f"\n  Mean Future Error: {mae:.6f}")

    passed = acc["best_fitness"] > 0.01
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def scenario_mandelbrot_divergent():
    """Predict Mandelbrot orbit for c = 0.3+0.5j (divergent)."""
    print("-" * 64)
    print("  SCENARIO 3: MANDELBROT ORBIT (DIVERGENT)")
    print("  z_{n+1} = z_n^2 + c,  c = 0.3 + 0.5j")
    print("-" * 64)
    print()

    c = complex(0.3, 0.5)
    n_terms = 50
    seq = mandelbrot_sequence(c, n_terms)

    print(f"  Sequence length: {len(seq)}")
    print(f"  First 10: [{', '.join(f'{v:.4f}' for v in seq[:10])}]")
    if len(seq) < n_terms:
        print(f"  (Escaped to infinity at step {len(seq)})")
    print()

    sp = SequencePredictor(window_size=15, predict_ahead=3)

    for v in seq:
        result = sp.feed(v)

    acc = sp.accuracy()
    print(f"  FINAL RESULTS:")
    print(f"    Formula:    {acc['best_formula'][:80]}")
    print(f"    Fitness:    {acc['best_fitness']:.4f}")
    print(f"    MAE:        {acc['mean_absolute_error']:.6f}")
    print(f"    Steps:      {acc['steps']}")

    # This is hard to predict — just verify it doesn't crash
    preds = sp.predict(3)
    print(f"    Next 3 predictions: {[f'{p:.4f}' for p in preds]}")
    all_finite = np.all(np.isfinite(preds))
    print(f"    All finite: {all_finite}")

    passed = all_finite
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    print("=" * 64)
    print("  SEQUENCE PREDICTOR MVP")
    print("  Quadratic Discovery + Mandelbrot Prediction")
    print("=" * 64)
    print()

    results = []
    results.append(("Quadratic", scenario_quadratic()))
    results.append(("Mandelbrot (bounded)", scenario_mandelbrot_bounded()))
    results.append(("Mandelbrot (divergent)", scenario_mandelbrot_divergent()))

    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print()
    if all(p for _, p in results):
        print("  All scenarios PASSED.")
    else:
        print("  Some scenarios FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
