# Cognitive Fractal — Design Decisions & Conversation Summary

## Overview

Cognitive Fractal is a universal function discovery engine that, given (x, y) data, discovers the symbolic generating function (e.g., `sin(x² + log(x) + 3)`) using fractal decomposition and inversion.

---

## Phase 1: Core System Build

Built the complete discovery pipeline:
- **9 primitive functions** (sin, cos, tan, exp, log, sqrt, tanh, log2, log10) with forward/inverse/branch generators
- **InvertedCompositionFractal** — fits chains like `sin(cos(exp(poly(x))))` via backward inversion through the chain
- **MixedInnerFractal** — fits `F(poly(x) + G(inner_poly(x)))` via combinatorial unwrap (handles `sin(x² + log(x) + 3)`)
- **CSVDiscoverer** — the main pipeline: cache check → tiered direct-fit → streaming → ranking
- **PatternStore** — JSON persistence for cross-run learning (instant results on second run)

### Performance after 6-step optimization

| Function | Time | Speedup |
|---|---|---|
| `2x + 3` | 0.0s | instant |
| `exp(0.5x + 1)` | 0.0s | instant |
| `sin(x² + 3)` | 5s | 65x |
| `sin(x²+log(x)+3)` | 42s | 29x |
| Any (2nd run, cache) | 0.0s | instant |

228 tests, all passing.

### Key optimization techniques
1. **Signal prefilter** — O(n) y-analysis skips impossible families (bounded/monotonic/sign)
2. **Tiered discovery** — 6 tiers from cheap (polynomial) to expensive (3-level chains), early exit between tiers
3. **Progressive degree** — all functions at deg=1 → deg=2 → deg=3, exit when perfect fit found
4. **Phase reorder** — direct-fit runs first; R² > 0.999 skips streaming entirely
5. **QR projection** — pre-computed QR for all G models; residual = ||z||² - ||Q^T z||²
6. **Adaptive K** — branch enumeration budget caps per-point branches, sorted by |z|
7. **Parallel fitting** — ThreadPoolExecutor for tiers with 4+ candidates

---

## Phase 2: Four Pillars Evaluation

Evaluated the system against `FOUR_PILLARS_IN_ACTION.md`, which defines four learning principles around the base `Fractal` class's online process()/learn() cycle:

| Pillar | Rating | Assessment |
|---|---|---|
| Feedback Loops | PARTIAL | Selection (best fit wins), not correction (predict→observe→correct) |
| Approximability | WEAK | Pipeline-level (tiered search), not fractal-level (monotonic convergence) |
| Composability | **STRONG** | Parent-child hierarchy, residual analysis, chain composition |
| Exploration | MODERATE | Planned (tiered enumeration), not reactive (novelty-triggered spawning) |

### Fundamental gap identified

The document describes online incremental learning (process→learn→update weights), but the function discovery pipeline uses batch optimization (polyfit, Vandermonde enumeration, scipy.minimize) that bypasses this cycle entirely.

---

## Phase 3: The Inheritance Gap

`FunctionFractal` inherits from `Fractal` but was bypassing all learning machinery.

### The problem

```python
FunctionFractal.__init__() called: super().__init__(dim=1, ...)
```

This created a 1-dimensional prototype, variance, and prediction_weights that were never read or written by any FunctionFractal method. The entire online learning apparatus was dead weight.

### Design decision: Hybrid approach

Rather than choosing between:
- **Pure batch** (exact solutions but no learning state) — the original behavior
- **Pure online** (Four Pillars aligned but slower, less accurate) — the document's vision

We chose a **hybrid**: keep batch `fit()` for coefficient discovery (exact solutions via polyfit/lstsq/inversion), and add prototype/variance tracking to give the Fractal's inherited state meaning.

### Semantic mapping

| Fractal State | Was (dead) | Now (alive) |
|---|---|---|
| `dim` | 1 (hardcoded) | `n_coeffs` (matches coefficient count) |
| `prototype` | `[0.]` (never updated) | EMA of coefficients across successive fits |
| `variance` | `[1.]` (never updated) | How much coefficients change between fits |
| `prediction_weights` | `[[0.01]]` (never used) | Still unused (batch fit handles updates) |

---

## Phase 4: Implementation

Three changes to `function_fractal.py`:

### Change 1: `dim=n_coeffs`

The prototype and variance vectors now have the same dimensionality as the coefficients they track. A LinearFractal has dim=2, SinFractal has dim=4, etc.

```python
# Before:
super().__init__(dim=1, domain="symbolic", learning_rate=learning_rate)

# After:
super().__init__(dim=n_coeffs, domain="symbolic", learning_rate=learning_rate)
```

### Change 2: Coefficient stability tracking in `process()`

After each `fit()` call, we compute the delta between the new coefficients and the current prototype, then update:
- `prototype += learning_rate * (coefficients - prototype)` — smoothed coefficient average
- `variance += variance_rate * (delta² - variance)` — tracks how much coefficients jitter

A guard clause `if len(self.coefficients) == self.dim` protects against edge cases where subclasses might have extra state outside the coefficient array (e.g., InvertedCompositionFractal's `_outer_scale`).

### Change 3: `seed_from()` transfers prototype + variance

When a pattern is loaded from the library and seeded into a fresh candidate, it inherits not just the coefficients but also the donor's stability knowledge. A fractal that was very stable (low variance) signals high confidence.

### Risk mitigation

- **PatternStore independence** — PatternStore uses its own `_serialize_candidate()`, NOT `Fractal.compress()`. Changes to dim/prototype/variance cannot break persistence.
- **No existing tests check `.dim`, `.prototype`, or `.variance`** on FunctionFractal instances, so the change is backward-compatible.

---

## Verification

- **228/228 tests pass** — zero breakage
- **Smoke test confirms:**
  - Prototype converges: `[0, 0]` → `[1.9999, 2.9999]` after 200 steps of `y = 2x + 3`
  - Variance shrinks: `[1.15, 1.4]` → `[0.0002, 0.0004]` (stable coefficients = low variance)
  - `seed_from` transfers both prototype and variance (with independent copies, no aliasing)

---

## What This Enables (Looking Forward)

The prototype/variance state opens doors for future improvements aligned with the Four Pillars:

- **Confidence-weighted ranking** — candidates with low variance = stable fits = higher confidence
- **Warm-start from prototype** — instead of fitting from scratch, initialize from the smoothed prototype
- **Novelty detection** — if coefficients suddenly jump (variance spikes), the fractal can signal that the data distribution shifted
- **Exploration budget** — high-variance fractals need more exploration; low-variance ones are settled
