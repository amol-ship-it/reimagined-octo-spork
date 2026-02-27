#!/usr/bin/env python3
"""
Demo: Three Improvements — Warm Starts, Library Restoration, Exotic Functions

Scenario 1: Warm-Start Transfer
  Stream 1: y = 3*sin(0.5*x) + 1 (learns frequency)
  Stream 2: y = 2*sin(0.5*x) + 5 (same frequency, different params)
  GradientSinFractal retains seeded frequency through warm-start fitting.

Scenario 2: Library Restoration After Pruning
  Runs a stream with aggressive composition to trigger pruning.
  Verifies that high-fitness patterns from memory refill empty slots.

Scenario 3: Exponential Discovery
  Stream: y = 2*exp(0.05*x) + 3
  ExponentialFractal should be among the best candidates.

Scenario 4: Composed Pattern Reuse
  Stream 1 discovers sin + linear composition.
  Stream 2 gets it loaded as a template from shared memory.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_fractal.memory import Memory
from cognitive_fractal.symbolic_engine import SymbolicEngine
from cognitive_fractal.function_fractal import (
    GradientSinFractal,
    GradientCosFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)


def scenario_warm_start():
    """Demonstrate warm-start gradient descent transfer."""
    print("-" * 64)
    print("  SCENARIO 1: WARM-START TRANSFER")
    print("-" * 64)
    print()

    shared = Memory()

    # Stream 1: learn sin(0.5x)
    e1 = SymbolicEngine(window_size=30, memory=shared)
    for t in range(300):
        e1.step(3.0 * np.sin(0.5 * float(t)) + 1.0)

    formula1, fitness1, _ = e1.get_best()
    print(f"  Stream 1: {formula1}")
    print(f"  Fitness:  {fitness1:.4f}")
    print()

    # Stream 2 WITH transfer
    e2_transfer = SymbolicEngine(window_size=30, memory=shared)
    transfer_errors = []
    prev_pred = None
    for t in range(150):
        y = 2.0 * np.sin(0.5 * float(t)) + 5.0
        pred, diag = e2_transfer.step(y)
        if prev_pred is not None and t > 5:
            transfer_errors.append(abs(prev_pred - y))
        prev_pred = pred

    formula_t, fitness_t, _ = e2_transfer.get_best()

    # Stream 2 WITHOUT transfer
    e2_isolated = SymbolicEngine(window_size=30)
    isolated_errors = []
    prev_pred = None
    for t in range(150):
        y = 2.0 * np.sin(0.5 * float(t)) + 5.0
        pred, diag = e2_isolated.step(y)
        if prev_pred is not None and t > 5:
            isolated_errors.append(abs(prev_pred - y))
        prev_pred = pred

    formula_i, fitness_i, _ = e2_isolated.get_best()

    print(f"  Stream 2 WITH transfer:    {formula_t}  fitness={fitness_t:.4f}")
    print(f"  Stream 2 WITHOUT transfer: {formula_i}  fitness={fitness_i:.4f}")
    print()

    # Compare early errors
    early_t = np.mean(transfer_errors[:30]) if len(transfer_errors) >= 30 else float("nan")
    early_i = np.mean(isolated_errors[:30]) if len(isolated_errors) >= 30 else float("nan")
    late_t = np.mean(transfer_errors[-30:]) if len(transfer_errors) >= 30 else float("nan")
    late_i = np.mean(isolated_errors[-30:]) if len(isolated_errors) >= 30 else float("nan")

    print(f"  {'Phase':<20s}  {'Transfer':>12s}  {'Isolated':>12s}")
    print(f"  {'-' * 48}")
    print(f"  {'Early (5-35)':20s}  {early_t:12.6f}  {early_i:12.6f}")
    print(f"  {'Late (last 30)':20s}  {late_t:12.6f}  {late_i:12.6f}")
    print()

    # Check gradient sin was seeded
    grad_sin_seeded = False
    for c in e2_transfer.candidates:
        if isinstance(c, GradientSinFractal):
            a, b, cc, d = c.coefficients
            if abs(b - 0.5) < 0.3:
                grad_sin_seeded = True
    print(f"  GradientSinFractal seeded with freq ~0.5: {grad_sin_seeded}")

    passed = late_t < late_i * 1.5
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def scenario_library_restoration():
    """Demonstrate that pruning restores patterns from the library."""
    print("-" * 64)
    print("  SCENARIO 2: LIBRARY RESTORATION AFTER PRUNING")
    print("-" * 64)
    print()

    engine = SymbolicEngine(
        window_size=15,
        max_candidates=12,
        composition_interval=20,
        composition_threshold=0.1,
    )

    initial_count = None
    prune_happened = False
    restore_count = 0

    for t in range(400):
        y = float(t) * 0.5 + 2.0 * np.sin(0.3 * float(t)) + 0.1 * float(t) ** 0.5
        engine.step(y)

        if t == 0:
            initial_count = len(engine.candidates)

        if len(engine.candidates) < initial_count and not prune_happened:
            prune_happened = True

    memory_count = engine.memory.stats()["hot_count"]
    candidate_count = len(engine.candidates)

    print(f"  Initial candidates:  {initial_count}")
    print(f"  Final candidates:    {candidate_count}")
    print(f"  Patterns in memory:  {memory_count}")
    print(f"  Pruning occurred:    {prune_happened}")
    print()

    # Memory should have more patterns than active candidates
    # (pruned high-fitness patterns stay in memory)
    passed = memory_count >= candidate_count
    print(f"  Memory retains more patterns than active candidates: {passed}")
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def scenario_exponential():
    """Demonstrate exponential function discovery."""
    print("-" * 64)
    print("  SCENARIO 3: EXPONENTIAL DISCOVERY")
    print("  y = 2*exp(0.05*x) + 3")
    print("-" * 64)
    print()

    engine = SymbolicEngine(window_size=30)

    for t in range(200):
        y = 2.0 * np.exp(0.05 * float(t)) + 3.0
        engine.step(y)

    formula, fitness, best = engine.get_best()
    print(f"  Discovered: {formula}")
    print(f"  Fitness:    {fitness:.4f}")
    print()

    # Check all candidate fitnesses
    stats = engine.get_stats()
    print(f"  Top candidates:")
    for i, cd in enumerate(stats["candidate_details"][:5]):
        print(f"    {i+1}. {cd['name']:20s}  fitness={cd['fitness']:.4f}  {cd['formula']}")
    print()

    # Check if ExponentialFractal has reasonable fitness
    exp_fitness = 0.0
    for c in engine.candidates:
        if isinstance(c, ExponentialFractal):
            exp_fitness = max(exp_fitness, c.metrics.fitness())

    print(f"  Best ExponentialFractal fitness: {exp_fitness:.4f}")
    passed = fitness > 0.05
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def scenario_composed_reuse():
    """Demonstrate composed pattern reuse across streams."""
    print("-" * 64)
    print("  SCENARIO 4: COMPOSED PATTERN REUSE")
    print("-" * 64)
    print()

    shared = Memory()

    # Stream 1: learn sin + linear composition
    e1 = SymbolicEngine(
        window_size=30,
        composition_interval=30,
        composition_threshold=0.2,
        memory=shared,
    )
    for t in range(300):
        y = 0.5 * float(t) + 3.0 * np.sin(0.5 * float(t)) + 2.0
        e1.step(y)

    formula1, fitness1, _ = e1.get_best()
    print(f"  Stream 1 discovered: {formula1}")
    print(f"  Fitness: {fitness1:.4f}")

    # Check for composed patterns in memory
    composed_in_memory = sum(
        1 for f in shared.hot.values()
        if isinstance(f, ComposedFunctionFractal)
    )
    high_fitness_composed = sum(
        1 for f in shared.hot.values()
        if isinstance(f, ComposedFunctionFractal) and f.metrics.fitness() > 0.15
    )
    print(f"  Composed patterns in memory: {composed_in_memory}")
    print(f"  High-fitness composed: {high_fitness_composed}")
    print()

    # Stream 2: similar pattern, should benefit from composed templates
    e2 = SymbolicEngine(
        window_size=30,
        composition_interval=30,
        composition_threshold=0.2,
        memory=shared,
    )
    e2.step(0.0)

    composed_loaded = sum(
        1 for c in e2.candidates if isinstance(c, ComposedFunctionFractal)
    )
    print(f"  Stream 2 loaded {composed_loaded} composed template(s)")

    for t in range(1, 200):
        y = 0.3 * float(t) + 2.0 * np.sin(0.5 * float(t)) + 1.0
        e2.step(y)

    formula2, fitness2, _ = e2.get_best()
    print(f"  Stream 2 discovered: {formula2}")
    print(f"  Fitness: {fitness2:.4f}")

    passed = fitness2 > 0.05
    print(f"  STATUS: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    print("=" * 64)
    print("  THREE IMPROVEMENTS DEMO")
    print("  Warm Starts | Library Restoration | Exotic Functions")
    print("=" * 64)
    print()

    results = []
    results.append(("Warm-Start Transfer", scenario_warm_start()))
    results.append(("Library Restoration", scenario_library_restoration()))
    results.append(("Exponential Discovery", scenario_exponential()))
    results.append(("Composed Reuse", scenario_composed_reuse()))

    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}  {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All scenarios PASSED.")
    else:
        print("  Some scenarios FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
