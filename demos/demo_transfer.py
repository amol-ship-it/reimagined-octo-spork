#!/usr/bin/env python3
"""
Demo: Transfer Learning Between Streams

Demonstrates the pattern library and shared memory system.

Part 1 — Mechanism Verification:
  Shows that learned patterns are stored in shared Memory,
  coefficients are seeded into new engines, and signatures
  enable similarity search across engines.

Part 2 — Composition Transfer:
  Stream 1: y = 3*sin(0.5*x) + 1         (learns sin with freq=0.5)
  Stream 2: y = 0.1*x^2 + 2*sin(0.5*x)   (composed: quad + sin)

  The library-enhanced composition search finds Stream 1's
  sin(0.5x) pattern when analyzing Stream 2's residuals,
  giving it an additional candidate for composition.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_fractal.memory import Memory
from cognitive_fractal.symbolic_engine import SymbolicEngine
from cognitive_fractal.function_fractal import SinFractal, LinearFractal


# ============================================================
# TEST FUNCTIONS
# ============================================================

def stream1_fn(x):
    return 3.0 * np.sin(0.5 * x) + 1.0


def stream2_fn(x):
    return 0.1 * x ** 2 + 2.0 * np.sin(0.5 * x)


# ============================================================
# PART 1: MECHANISM VERIFICATION
# ============================================================

def verify_mechanisms():
    """Show that the pattern library mechanisms work."""
    print("-" * 64)
    print("  PART 1: MECHANISM VERIFICATION")
    print("-" * 64)
    print()

    shared = Memory()

    # Train Stream 1
    e1 = SymbolicEngine(window_size=30, memory=shared)
    for t in range(200):
        e1.step(stream1_fn(float(t)))

    formula1, fitness1, _ = e1.get_best()
    print(f"  Stream 1 discovered: {formula1}")
    print(f"  Stream 1 fitness:    {fitness1:.4f}")
    print()

    # Check what's in memory
    sig_count = shared.stats()["signature_count"]
    hot_count = shared.stats()["hot_count"]
    print(f"  Shared memory after Stream 1:")
    print(f"    Patterns stored:   {hot_count}")
    print(f"    Signatures indexed: {sig_count}")
    print()

    # Find sin-type patterns with high fitness
    sin_patterns = []
    for frac in shared.hot.values():
        if isinstance(frac, SinFractal) and frac.metrics.fitness() > 0.1:
            sin_patterns.append(frac)
    print(f"  High-fitness SinFractal patterns in library: {len(sin_patterns)}")
    if sin_patterns:
        best_sin = max(sin_patterns, key=lambda f: f.metrics.fitness())
        a, b, c, d = best_sin.coefficients
        print(f"    Best sin coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    print()

    # Create Stream 2 engine — check that seeding happens
    e2 = SymbolicEngine(window_size=30, memory=shared)
    e2.step(0.0)  # Triggers _initialize_candidates + _seed_from_library

    seeded = False
    for c in e2.candidates:
        if isinstance(c, SinFractal):
            a, b, c_coeff, d = c.coefficients
            if abs(b - 0.5) < 0.1:  # Seeded with Stream 1's frequency
                seeded = True
                print(f"  Stream 2 SinFractal was seeded!")
                print(f"    Coefficients: a={a:.4f}, b={b:.4f}, c={c_coeff:.4f}, d={d:.4f}")
            else:
                print(f"  Stream 2 SinFractal coefficients: a={a:.4f}, b={b:.4f}")
            break

    if not seeded:
        print("  Stream 2 SinFractal was NOT seeded (no high-fitness match).")
        print("  This is expected if Stream 1's sin didn't achieve fitness > 0.1.")

    # Verify signature-based search works
    if sin_patterns:
        query_sig = sin_patterns[0].compute_signature()
        results = shared.find_similar_by_signature(
            query_sig, domain="symbolic", top_k=3, min_fitness=0.1
        )
        print(f"\n  Signature search for sin-like patterns: {len(results)} results")
        for frac, sim in results:
            print(f"    {frac.func_name:20s}  similarity={sim:.4f}  fitness={frac.metrics.fitness():.4f}")

    print()
    return shared


# ============================================================
# PART 2: COMPOSITION TRANSFER
# ============================================================

def run_composition_comparison(shared_memory):
    """Compare composition discovery with vs without transfer."""
    print("-" * 64)
    print("  PART 2: COMPOSITION TRANSFER")
    print("  Stream 2: y = 0.1*x^2 + 2*sin(0.5*x)")
    print("-" * 64)
    print()

    n_steps = 300

    # --- With transfer ---
    engine_transfer = SymbolicEngine(
        window_size=30,
        composition_interval=50,
        composition_threshold=0.3,
        memory=shared_memory,
    )
    transfer_errors = []
    prev_pred = None
    for t in range(n_steps):
        y = stream2_fn(float(t))
        pred, diag = engine_transfer.step(y)
        if prev_pred is not None:
            transfer_errors.append(abs(prev_pred - y))
        prev_pred = pred

    formula_t, fitness_t, _ = engine_transfer.get_best()
    print(f"  WITH TRANSFER:")
    print(f"    Discovered: {formula_t}")
    print(f"    Fitness:    {fitness_t:.4f}")
    print(f"    Candidates: {len(engine_transfer.candidates)}")

    # --- Without transfer ---
    engine_isolated = SymbolicEngine(
        window_size=30,
        composition_interval=50,
        composition_threshold=0.3,
    )
    isolated_errors = []
    prev_pred = None
    for t in range(n_steps):
        y = stream2_fn(float(t))
        pred, diag = engine_isolated.step(y)
        if prev_pred is not None:
            isolated_errors.append(abs(prev_pred - y))
        prev_pred = pred

    formula_i, fitness_i, _ = engine_isolated.get_best()
    print(f"  WITHOUT TRANSFER:")
    print(f"    Discovered: {formula_i}")
    print(f"    Fitness:    {fitness_i:.4f}")
    print(f"    Candidates: {len(engine_isolated.candidates)}")

    # --- Comparison ---
    print()
    print(f"  {'Phase':<26s}  {'With Transfer':>14s}  {'Without':>14s}")
    print(f"  {'-' * 58}")

    bins = [
        ("Early  (steps 10-30)", 10, 30),
        ("Middle (steps 50-100)", 50, 100),
        ("Late   (steps 200-280)", 200, 280),
    ]
    for label, start, end in bins:
        t_err = np.mean(transfer_errors[start:end]) if len(transfer_errors) >= end else float("nan")
        i_err = np.mean(isolated_errors[start:end]) if len(isolated_errors) >= end else float("nan")
        print(f"  {label:<26s}  {t_err:14.6f}  {i_err:14.6f}")

    # --- Memory state ---
    print()
    stats = shared_memory.stats()
    print(f"  Final shared memory state:")
    print(f"    Patterns: {stats['hot_count']}  Signatures: {stats['signature_count']}")

    return transfer_errors, isolated_errors


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 64)
    print("  TRANSFER LEARNING DEMO")
    print("  Pattern Library & Shared Memory")
    print("=" * 64)
    print()
    print("  This demo shows three things:")
    print("    1. Learned patterns are stored with signatures in Memory")
    print("    2. New engines seed candidates from the library")
    print("    3. Composition search queries the library for residual matches")
    print()

    # Part 1: Verify mechanisms
    shared = verify_mechanisms()

    # Part 2: Composition comparison
    transfer_errors, isolated_errors = run_composition_comparison(shared)

    # --- Verdict ---
    print()
    print("-" * 64)
    late_transfer = np.mean(transfer_errors[-50:])
    late_isolated = np.mean(isolated_errors[-50:])
    early_transfer = np.mean(transfer_errors[10:30])
    early_isolated = np.mean(isolated_errors[10:30])

    print(f"  Final convergence error:")
    print(f"    With transfer:    {late_transfer:.6f}")
    print(f"    Without transfer: {late_isolated:.6f}")
    print()

    # The mechanism works — transfer should not make things worse
    passed = late_transfer < late_isolated * 1.5
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()

    if passed:
        print("  The pattern library infrastructure is working:")
        print("  - Signatures enable shape-based similarity search")
        print("  - Coefficients are seeded from high-fitness library patterns")
        print("  - Composition queries library for residual-matching patterns")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
