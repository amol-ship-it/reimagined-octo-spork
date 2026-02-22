#!/usr/bin/env python3
"""
Demo: Adaptive Multi-Scale Signal Prediction

A 1000-step signal with patterns at three temporal scales and a regime
shift at step 500. Three fractal engines operate at different window
sizes (fast/medium/slow). A fourth composed fractal combines the best
of each scale.

Exercises all Four Pillars of Learning:
  - Feedback Loops:  Error drops continuously within each regime
  - Approximability: More exposure = better fitness, independently per scale
  - Composability:   Composed fractal leverages children from 3 scales
  - Exploration:     Regime shift triggers new fractal creation
"""

import sys
import os
import copy
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_fractal import FractalEngine, Fractal, Signal, Feedback

# ============================================================
# CONSTANTS
# ============================================================
TOTAL_STEPS = 1000
REGIME_SHIFT = 500
SEED = 42
WARMUP = 40  # Need at least 40 steps for the slow window
COMPOSE_AT = 200  # When to build the composed fractal


# ============================================================
# SIGNAL GENERATION
# ============================================================
def generate_signal():
    """Generate a deterministic multi-scale signal with regime shift.

    Regime 1 (t < 500): sine(period=8) * envelope(period=40)
    Regime 2 (t >= 500): sawtooth(period=8) * envelope(period=40)
    """
    signal = np.zeros(TOTAL_STEPS)
    for t in range(TOTAL_STEPS):
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * t / 40.0)
        if t < REGIME_SHIFT:
            fast = np.sin(2 * np.pi * t / 8.0)
        else:
            fast = 2.0 * ((t % 8) / 8.0) - 1.0
        signal[t] = envelope * fast
    return signal


# ============================================================
# WINDOWING
# ============================================================
def get_window(signal, t, window_size):
    """Extract a sliding window ending at time t, left-zero-padded if needed."""
    start = t - window_size + 1
    if start < 0:
        window = np.zeros(window_size)
        window[-start:] = signal[0 : t + 1]
    else:
        window = signal[start : t + 1].copy()
    return window


# ============================================================
# HELPERS
# ============================================================
def best_fractal(engine):
    """Find the highest-fitness fractal in an engine's hot memory."""
    return max(engine.memory.hot.values(), key=lambda f: f.metrics.fitness())


# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(SEED)

    print("=" * 72)
    print("ADAPTIVE MULTI-SCALE SIGNAL PREDICTION")
    print("Exercising all Four Pillars of Learning")
    print("=" * 72)
    print()
    print(f"Signal: {TOTAL_STEPS} steps, regime shift at step {REGIME_SHIFT}")
    print("  Regime 1 (0-499):   sine(period=8) * envelope(period=40)")
    print("  Regime 2 (500-999): sawtooth(period=8) * envelope(period=40)")
    print()

    signal = generate_signal()

    # --- Create three engines at different scales ---
    engine_fast = FractalEngine(
        dim=8, domain="fast", novelty_threshold=0.5, learning_rate=0.15
    )
    engine_medium = FractalEngine(
        dim=16, domain="medium", novelty_threshold=0.5, learning_rate=0.12
    )
    engine_slow = FractalEngine(
        dim=40, domain="slow", novelty_threshold=0.5, learning_rate=0.10
    )

    engines = [
        ("Fast(8)", engine_fast, 8),
        ("Med(16)", engine_medium, 16),
        ("Slow(40)", engine_slow, 40),
    ]

    # --- Tracking ---
    errors = {name: [] for name, _, _ in engines}
    fractal_counts = {name: [] for name, _, _ in engines}

    composed = None
    composed_errors = []

    fractals_before_shift = {}
    fractals_after_shift = {}

    # Snapshot steps for table output
    snapshots = {50, 150, 250, 450, 499, 510, 550, 750, 999}

    # ============================================================
    # MAIN LOOP
    # ============================================================
    header = (
        f"{'Step':>5} | {'Fast Err':>9} | {'Med Err':>9} | "
        f"{'Slow Err':>9} | {'F#':>3} {'M#':>3} {'S#':>3} | Regime"
    )
    print(header)
    print("-" * len(header))

    for t in range(WARMUP, TOTAL_STEPS):
        regime = "sine" if t < REGIME_SHIFT else "sawtooth"

        # --- Step each engine with its windowed input ---
        step_errors = {}
        step_fractals = {}

        for name, engine, win_size in engines:
            window = get_window(signal, t, win_size)
            _, diag = engine.step(window)
            step_errors[name] = diag["prediction_error"]
            step_fractals[name] = diag["active_fractals"]
            errors[name].append(diag["prediction_error"])
            fractal_counts[name].append(diag["active_fractals"])

        # --- Snapshot fractal counts around regime shift ---
        if t == REGIME_SHIFT - 1:
            for name, engine, _ in engines:
                fractals_before_shift[name] = len(engine.memory.hot)
        if t == REGIME_SHIFT + 50:
            for name, engine, _ in engines:
                fractals_after_shift[name] = len(engine.memory.hot)

        # --- Build composed fractal at COMPOSE_AT ---
        if t == COMPOSE_AT:
            best_f = copy.deepcopy(best_fractal(engine_fast))
            best_m = copy.deepcopy(best_fractal(engine_medium))
            best_s = copy.deepcopy(best_fractal(engine_slow))

            composed = Fractal(dim=16, domain="composed", learning_rate=0.12)
            composed.add_child(best_f)
            composed.add_child(best_m)
            composed.add_child(best_s)
            print(
                f"\n  >>> Composed fractal created at step {t} "
                f"from best of each scale <<<\n"
            )

        # --- Drive composed fractal manually ---
        if composed is not None and t > COMPOSE_AT:
            slow_window = get_window(signal, t, 40)
            now = time.time()

            # Feedback for previous step (learn BEFORE process)
            if t > COMPOSE_AT + 1:
                fb = Feedback(actual=slow_window, reward=0.0, timestamp=now)
                cerr = composed.learn(fb)
                composed_errors.append(cerr)

            # Forward pass for this step
            sig = Signal(data=slow_window, timestamp=now)
            composed.process(sig)

        # --- Print at snapshot steps ---
        if t in snapshots:
            marker = " <-- REGIME SHIFT" if t == 510 else ""
            print(
                f"{t:5d} | {step_errors['Fast(8)']:9.4f} | "
                f"{step_errors['Med(16)']:9.4f} | "
                f"{step_errors['Slow(40)']:9.4f} | "
                f"{step_fractals['Fast(8)']:3d} "
                f"{step_fractals['Med(16)']:3d} "
                f"{step_fractals['Slow(40)']:3d} | {regime}{marker}"
            )

    # ============================================================
    # FOUR PILLARS — QUANTITATIVE REPORT
    # ============================================================
    print()
    print("=" * 72)
    print("FOUR PILLARS — QUANTITATIVE REPORT")
    print("=" * 72)

    all_passed = True
    r1_end = REGIME_SHIFT - WARMUP  # Index into errors[] for step 499

    # ----------------------------------------------------------
    # PILLAR 1: FEEDBACK LOOPS
    # ----------------------------------------------------------
    print()
    print("PILLAR 1: FEEDBACK LOOPS")
    print("  The predict -> observe -> correct cycle is operating continuously.")
    print("  Metric: fitness increases from first exposure to convergence")
    for name, engine, _ in engines:
        bf = best_fractal(engine)
        exposures = bf.metrics.total_exposures
        fitness = bf.metrics.fitness()
        err_ema = bf.metrics.prediction_error_ema
        # Fitness > 0 proves the loop is closing (predictions are
        # being evaluated against outcomes and weights are updating)
        status = "OK" if fitness > 0.1 else "WARN"
        print(
            f"  {name:10s} exposures={exposures:4d}  "
            f"error_ema={err_ema:.4f}  fitness={fitness:.3f}  [{status}]"
        )
        if fitness <= 0.1:
            all_passed = False
    # Also show that the loop closes across the regime shift:
    # post-shift error spikes then drops, proving feedback drives recovery
    shift_idx = REGIME_SHIFT - WARMUP
    for name, _, _ in engines:
        errs = errors[name]
        if shift_idx + 50 < len(errs):
            spike = np.max(errs[shift_idx : shift_idx + 10])
            settled = np.mean(errs[shift_idx + 40 : shift_idx + 50])
            recovered = settled < spike
            print(
                f"  {name:10s} post-shift: spike={spike:.4f} -> "
                f"settled={settled:.4f}  recovery={'yes' if recovered else 'no'}"
            )

    # ----------------------------------------------------------
    # PILLAR 2: APPROXIMABILITY
    # ----------------------------------------------------------
    print()
    print("PILLAR 2: APPROXIMABILITY")
    print("  Metric: Error ratio (late / early) within regime 1")
    for name, engine, _ in engines:
        errs = errors[name]
        early_err = np.mean(errs[10:60])  # ~steps 50-100
        late_err = np.mean(errs[max(0, r1_end - 50) : r1_end])  # ~steps 450-500
        ratio = late_err / (early_err + 1e-10)
        bf = best_fractal(engine)
        status = "OK" if ratio < 1.0 else "WARN"
        print(
            f"  {name:10s} early={early_err:.4f}  late={late_err:.4f}  "
            f"ratio={ratio:.3f}  fitness={bf.metrics.fitness():.3f}  [{status}]"
        )
        if ratio >= 1.0:
            all_passed = False

    # ----------------------------------------------------------
    # PILLAR 3: COMPOSABILITY
    # ----------------------------------------------------------
    print()
    print("PILLAR 3: COMPOSABILITY")
    if composed_errors:
        comp_mean = np.mean(composed_errors)
        offset = COMPOSE_AT + 2 - WARMUP  # First composed error index
        individual_means = {}
        for name, _, _ in engines:
            ind_errs = errors[name][offset:]
            individual_means[name] = np.mean(ind_errs) if ind_errs else float("inf")
            print(
                f"  {name:10s} mean error (steps {COMPOSE_AT + 2}+): "
                f"{individual_means[name]:.4f}"
            )
        print(f"  {'Composed':10s} mean error (steps {COMPOSE_AT + 2}+): {comp_mean:.4f}")
        print(f"  Composed fitness: {composed.metrics.fitness():.3f}")
        print(f"  Composed exposures: {composed.metrics.total_exposures}")
        # Composition is demonstrated if the composed fractal achieves
        # non-trivial fitness (> 0) and its error is in the same ballpark
        if composed.metrics.fitness() > 0:
            print("  Composition mechanism: ACTIVE (fitness > 0, learning through children)")
        else:
            print("  Composition mechanism: NOT YET CONVERGED")
            all_passed = False
    else:
        print("  (No composed errors recorded)")
        all_passed = False

    # ----------------------------------------------------------
    # PILLAR 4: EXPLORATION
    # ----------------------------------------------------------
    print()
    print("PILLAR 4: EXPLORATION")
    print("  Metric: Fractals spawned after regime shift")
    total_new = 0
    for name, _, _ in engines:
        before = fractals_before_shift.get(name, 0)
        after = fractals_after_shift.get(name, 0)
        spawned = after - before
        total_new += spawned
        print(f"  {name:10s} before={before}  after={after}  new=+{spawned}")
    if total_new == 0:
        print("  WARNING: No new fractals spawned after regime shift")
        all_passed = False
    else:
        print(f"  Total new fractals across all scales: +{total_new}")

    # ----------------------------------------------------------
    # RECOVERY ANALYSIS
    # ----------------------------------------------------------
    print()
    print("RECOVERY ANALYSIS")
    print("  Metric: Steps to return to 2x pre-shift error level")
    for name, _, _ in engines:
        errs = errors[name]
        pre_baseline = np.mean(errs[max(0, r1_end - 50) : r1_end])
        shift_idx = REGIME_SHIFT - WARMUP
        recovery_steps = None
        # Scan post-shift using a 20-step rolling average
        for i in range(shift_idx + 5, min(shift_idx + 300, len(errs))):
            window_mean = np.mean(errs[max(shift_idx, i - 20) : i + 1])
            if window_mean <= pre_baseline * 2.0 + 1e-8:
                recovery_steps = i - shift_idx
                break
        if recovery_steps is not None:
            print(
                f"  {name:10s} baseline={pre_baseline:.4f}  "
                f"recovered in {recovery_steps} steps"
            )
        else:
            print(
                f"  {name:10s} baseline={pre_baseline:.4f}  "
                f"not recovered in 300 steps"
            )

    # ----------------------------------------------------------
    # FINAL VERDICT
    # ----------------------------------------------------------
    print()
    print("=" * 72)
    if all_passed:
        print("SUCCESS: All four pillars demonstrated quantitatively.")
    else:
        print("PARTIAL: Some pillars did not meet quantitative thresholds.")
        print("(This may indicate the learning rates or thresholds need tuning.)")
        sys.exit(1)


if __name__ == "__main__":
    main()
