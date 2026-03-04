# The Four Pillars of Learning — Demonstrated in Code

This document traces exactly how the cognitive-fractal system embodies the Four Pillars of Learning. Not as abstract theory, but as measurable behavior in running code.

The system has two layers: a **base Fractal** with online learning (process → learn → update), and a **function discovery layer** of 16 FunctionFractal types that use batch optimization to discover symbolic generating functions. This document is honest about where each layer aligns with the pillars and where it does not.

```
python3 demos/demo_sequence.py      # Simple: learn A-B-C-A-B-C
python3 demos/demo_multiscale.py    # Complex: multi-scale signal with regime shift
python3 -m cognitive_fractal data.csv  # Function discovery: find f(x) from CSV data
```

---

## What Are the Four Pillars?

The Four Pillars are recursive processes that must operate continuously for a system to learn from raw experience rather than from pre-programmed rules:

| Pillar | Core Question |
|--------|--------------|
| **Feedback Loops** | Does the system compare its predictions to reality and correct itself? |
| **Approximability** | Does repeated exposure make predictions more accurate over time? |
| **Composability** | Can learned patterns be reused as building blocks for higher-level patterns? |
| **Exploration** | Does the system create new patterns when existing ones fail to explain the input? |

The pillars are not independent features. They form a single continuous cycle: exploration creates new patterns, feedback loops evaluate them, approximability refines them, and composability assembles them into hierarchies. Remove any one pillar and the system stops learning.

---

## The Two Learning Layers

### Layer 1: The Base Fractal (Online Learning)

The base `Fractal` is an active agent — the smallest self-contained unit that can learn online. Every fractal holds four pieces of learned state:

| State | What It Represents | Initialized To |
|-------|-------------------|----------------|
| `prototype` | The center of the pattern this fractal has learned | Zeros (knows nothing) |
| `variance` | How much the inputs spread around the prototype | Ones (maximum uncertainty) |
| `prediction_weights` | A linear mapping: "given this input, what comes next?" | Near-zero identity (knows nothing) |
| `_prediction_bias` | Offset term for predictions | Zeros |

A fractal with zero exposures knows nothing. A fractal with hundreds of exposures has a sharp prototype, calibrated variance, and accurate prediction weights. The distance between those two states is the entire story of learning.

The critical design property: **a composed fractal and a leaf fractal run identical code**. The `process()` method does not check whether it is operating on raw audio samples or on the outputs of ten child fractals. If children exist, it delegates to them first. If not, it operates on raw input. This is structural self-similarity enforced by a single class definition.

### Layer 2: FunctionFractals (Batch Discovery + Online Tracking)

The function discovery layer adds 16 specialized fractal types that represent mathematical function hypotheses — from `LinearFractal` (y = mx + b) to `MixedInnerFractal` (y = F(poly + G(inner_poly))). These use **batch optimization** (polyfit, FFT, scipy.minimize, chain inversion) to discover coefficients from data.

The key difference: base Fractals learn *online* through the process()/learn() cycle. FunctionFractals discover coefficients through batch `fit()` calls. This is faster and more exact, but it originally bypassed the base Fractal's learning machinery entirely — `prototype`, `variance`, and `prediction_weights` were dead state.

**The hybrid fix** (implemented): FunctionFractal now sets `dim=n_coeffs` instead of `dim=1`, making the inherited state meaningful:

| State | Was (dead) | Now (alive) |
|-------|------------|-------------|
| `dim` | 1 (hardcoded) | `n_coeffs` (matches coefficient count) |
| `prototype` | `[0.]` (never updated) | EMA of coefficients across successive fits |
| `variance` | `[1.]` (never updated) | How much coefficients change between fits |
| `prediction_weights` | `[[0.01]]` (never used) | Still unused (batch fit handles updates) |

After each `fit()` in `process()`, the stability tracking runs:

```python
# function_fractal.py, process() — after fit()
if len(self.coefficients) == self.dim:
    coeff_delta = self.coefficients - self.prototype
    var_error = coeff_delta ** 2 - self.variance
    self.variance += self.variance_rate * var_error
    self.variance = np.maximum(self.variance, 1e-6)
    self.prototype += self.learning_rate * coeff_delta
```

This means a FunctionFractal that has been fit many times on stable data will have: prototype near its true coefficients, variance near zero. A FunctionFractal fit on noisy or shifting data will have: high variance, signaling low confidence.

---

## Pillar 1: Feedback Loops

**The predict-observe-correct cycle that makes learning possible.**

A system without feedback loops is a lookup table. It can retrieve stored answers but it cannot improve. The feedback loop is the mechanism that closes the gap between what a fractal predicts and what actually happens.

### Base Fractal: Full Online Feedback ✓

The feedback loop is implemented across two methods and one orchestrator:

**`Fractal.process()`** — the forward pass. The fractal receives input, compares it to its prototype, and generates a prediction of what comes next:

```python
# fractal.py, line 109
prediction = self.prediction_weights @ effective_input + self._prediction_bias
```

This prediction is stored as `_last_prediction`. The fractal has now committed to a guess.

**`Fractal.learn()`** — the backward pass. When the next input arrives, it becomes the "actual outcome" of the previous prediction. The fractal computes the error and updates itself:

```python
# fractal.py, lines 150-163
pred_error = actual - self._last_prediction
error_magnitude = float(np.sqrt(np.mean(pred_error**2)))

# Outer product learning rule: dW = lr * error * input^T
weight_update = self.learning_rate * np.outer(pred_error, normalized_input)
self.prediction_weights += weight_update
self._prediction_bias += self.learning_rate * pred_error
```

Three things update simultaneously: the prototype moves toward recent input (EMA), variance tracks input spread, and prediction weights adjust via gradient descent. This is not a confidence score changing — these are the actual parameters of the model being reshaped by error.

**`FractalEngine.step()`** — the orchestrator that closes the loop automatically:

```python
# engine.py, lines 84-91
if self._pending_fractal is not None:
    feedback = Feedback(actual=raw_input.copy(), reward=0.0, timestamp=now)
    error = self._pending_fractal.learn(feedback)
```

The current input serves as the actual outcome for the previous step's prediction. No external teacher is required. The loop closes naturally through the temporal structure of sequential input.

### Function Discovery: Selection, Not Correction — PARTIAL

The CSV function discovery pipeline (`CSVFunctionDiscoverer`) does not use the predict-observe-correct cycle. Instead, it:

1. Creates a pool of candidate FunctionFractals (29 types across 6 tiers)
2. Calls `fit()` on each candidate with the full dataset
3. Ranks by R² / RMSE
4. Selects the best — the winner takes all

This is **selection feedback**, not **correction feedback**. The system knows which candidate won, but losers are discarded rather than corrected. The winning fractal does not iteratively improve — it arrives at its answer in one batch `fit()`.

**Where partial feedback exists:** The `GradientSinFractal` and `GradientCosFractal` are the closest to true feedback — they use FFT for initialization then gradient descent for refinement, warm-starting from current coefficients on subsequent calls. The `SymbolicEngine` streaming path also runs process()/learn() in a loop, providing genuine online feedback.

**The hybrid fix helps:** After each `fit()`, the prototype EMA and variance tracking now run. If a FunctionFractal is fit repeatedly on overlapping windows (as in the streaming engine), the prototype converges toward stable coefficients and variance shrinks — a form of feedback on coefficient stability even within batch mode.

### How the Demo Proves It

**Sequence demo**: Three fractals self-organize to model A, B, and C. Each fractal's prediction error starts at 0.11 and drops to 0.00 over 300 steps. This is not possible without a functioning feedback loop.

```
Steps   1- 30: avg_error=0.1141  fitness=0.3038
Steps  91-120: avg_error=0.0000  fitness=0.9617
Steps 271-300: avg_error=0.0000  fitness=0.9999
```

**Multi-scale demo**: The feedback loop is tested by the regime shift at step 500. When the signal changes from sine to sawtooth, prediction error spikes. But the feedback loop keeps running — the fractals observe the new errors and adjust their weights. Within 5 steps, error recovers:

```
Fast(8)    post-shift: spike=0.3474 -> settled=0.1212  recovery=yes
Med(16)    post-shift: spike=0.2961 -> settled=0.2554  recovery=yes
Slow(40)   post-shift: spike=0.3359 -> settled=0.1513  recovery=yes
```

The spike-then-recovery pattern is the signature of a feedback loop under stress. A system without feedback would spike and stay there.

---

## Pillar 2: Approximability

**The guarantee that more exposure produces better predictions.**

Approximability means that a fractal's model of its input distribution improves monotonically with experience. If this property does not hold, the system is not learning — it is merely fluctuating.

### Base Fractal: Mathematically Guaranteed Convergence ✓

Approximability emerges from three update rules inside `Fractal.learn()`, all of which are convergent:

**Prototype convergence (EMA)**:
```python
# fractal.py, lines 141-142
proto_error = self._last_input - self.prototype
self.prototype += self.learning_rate * proto_error
```

This is an exponential moving average. Each exposure moves the prototype closer to the true center of the input distribution. After `n` exposures, the prototype error is approximately `(1 - learning_rate)^n` times the initial error. Mathematically guaranteed to converge.

**Variance calibration**:
```python
# fractal.py, lines 145-147
var_error = (self._last_input - self.prototype) ** 2 - self.variance
self.variance += self.variance_rate * var_error
```

The variance tracker learns how spread out inputs are around the prototype. A well-calibrated variance means the fractal knows not just *where* its pattern is, but *how much it varies*.

**Prediction weight convergence**:
```python
# fractal.py, lines 156-163
normalized_input = self._last_input / input_norm
weight_update = self.learning_rate * np.outer(pred_error, normalized_input)
self.prediction_weights += weight_update
self._prediction_bias += self.learning_rate * pred_error
```

The outer-product learning rule adjusts the weight matrix to reduce prediction error. Over many exposures, the weights converge to a linear model of the input-to-next-input mapping.

**Fitness as the composite measure**:
```python
# types.py, lines 41-47
def fitness(self) -> float:
    return self.accuracy_ema * (1.0 - self.prediction_error_ema)
```

Fitness combines accuracy and prediction error into a single scalar in [0, 1]. The trajectory from 0 to 1 is the approximability curve.

### Function Discovery: Pipeline-Level, Not Fractal-Level — WEAK

Individual FunctionFractals do not exhibit monotonic convergence through repeated exposure. A `LinearFractal` calling `fit()` on clean data gets the exact answer on the first call — there is nothing to improve. Calling `fit()` again on the same data produces the same answer, not a better one.

Approximability in the function discovery layer operates at the **pipeline level**:

1. **Tiered search** — 6 tiers from cheap (polynomial) to expensive (3-level chains). Each tier approximates the data better than the last if the previous tier failed.
2. **Progressive degree** — within each tier, degree 1 → 2 → 3. Higher degree = better approximation.
3. **PatternStore caching** — once a function is discovered, second-run discovery is instant (perfect approximation from memory).

**The hybrid fix helps:** With coefficient stability tracking, FunctionFractals that are fit repeatedly (e.g., in the streaming engine) now show measurable approximability:

```
Step   0: prototype=[0.1, 0.15],   variance=[1.15, 1.40]     (just started)
Step  50: prototype=[1.85, 2.78],  variance=[0.36, 0.71]     (converging)
Step 200: prototype=[2.00, 3.00],  variance=[0.0002, 0.0004] (converged)
```

The prototype converges toward true coefficients and variance shrinks — monotonic improvement through repeated exposure, even though the underlying `fit()` is batch.

### How the Demo Proves It

**Sequence demo** (base Fractal): Fitness progression:

```
Step   1: fitness=0.000 (knows nothing)
Step  30: fitness=0.304 (learning)
Step 120: fitness=0.962 (nearly converged)
Step 300: fitness=1.000 (fully converged)
```

Each of the three fractals independently travels this arc. The improvement is strictly monotonic.

**Multi-scale demo** (base Fractal): Approximability measured as error ratio (late/early). Below 1.0 means improvement:

```
Fast(8)    early=0.1083  late=0.1052  ratio=0.971  [OK]
Med(16)    early=0.1398  late=0.1274  ratio=0.911  [OK]
Slow(40)   early=0.1500  late=0.0770  ratio=0.514  [OK]
```

---

## Pillar 3: Composability

**Building complex understanding from simple parts.**

This is the strongest pillar in the system. Both the base Fractal and the function discovery layer demonstrate composability, through different but complementary mechanisms.

### Base Fractal: Parent-Child Hierarchies ✓

Composability is implemented through parent-child relationships between fractals.

**Adding children**:
```python
# fractal.py, lines 193-196
def add_child(self, child: "Fractal"):
    child.parent = self
    self.children.append(child)
```

**Forward pass through a composed fractal**:
```python
# fractal.py, lines 90-96
if self.is_composed:
    child_outputs = []
    for child in self.children:
        child_out, _ = child.process(input_signal)
        child_outputs.append(child_out.data)
    effective_input = self._aggregate_child_outputs(child_outputs)
```

When a composed fractal receives input, it routes the input through each child, then operates on the aggregated child outputs. The parent sees the input through the lens of its children's expertise.

**Recursive feedback**:
```python
# fractal.py, lines 178-180
if self.is_composed:
    for child in self.children:
        child.learn(feedback)
```

When a parent learns, feedback propagates to all children. The entire hierarchy improves simultaneously.

### Function Discovery: Four Composition Mechanisms ✓✓

The function discovery layer has the richest composability in the system, with four distinct composition types:

**1. ComposedFunctionFractal** — arithmetic combination of two child functions:
```
mode = add:      f(x) = outer(x) + inner(x)
mode = multiply:  f(x) = outer(x) * inner(x)
mode = subtract:  f(x) = outer(x) - inner(x)
mode = divide:    f(x) = outer(x) / inner(x)
```
Built via residual analysis: fit the dominant pattern first, then fit the remainder with a second function, combine.

**2. NestedComposedFractal** — function-of-function nesting:
```
f(x) = outer(inner(x))
```
Example: `sin(2x + 3)` where outer=SinFractal, inner=LinearFractal.

**3. InvertedCompositionFractal** — arbitrary-depth chains via backward inversion:
```
f(x) = F(G(...poly(x)))
```
Example: `exp(sin(2x + 1))` — chain=['exp','sin'], poly=[2,1]. Fitting works by inverting backward through the chain: apply exp⁻¹ to get sin(poly), apply sin⁻¹ (with branch enumeration) to get poly, then polyfit.

**4. MixedInnerFractal** — composite inner function:
```
f(x) = F(poly(x) + G(inner_poly(x)))
```
Example: `sin(x² + log(x) + 3)` — outer=sin, poly=[1,0,3], inner=log. Discovered via combinatorial unwrap with QR-based validation.

### The Self-Similarity Property

The most important aspect of composability is what it does *not* require: special-purpose code at the Fractal level. A composed fractal runs the same `process()` and `learn()` methods as a leaf fractal. The only difference is that `self.children` is non-empty.

At the FunctionFractal level, all composition types share the same interface: `evaluate(x)`, `fit(x, y)`, `symbolic_repr()`. A `MixedInnerFractal` that represents `sin(x² + log(x) + 3)` is used in exactly the same way as a `LinearFractal` that represents `2x + 3`. The CSVDiscoverer does not need to know which type it is working with — composability is transparent.

### Transfer Learning via seed_from()

Composability extends to knowledge transfer. The `seed_from()` method copies learned state from a library pattern into a fresh candidate:

```python
# function_fractal.py
def seed_from(self, other: "FunctionFractal") -> None:
    if type(self) == type(other) and len(self.coefficients) == len(other.coefficients):
        self.coefficients = other.coefficients.copy()
        if self.dim == other.dim:
            self.prototype = other.prototype.copy()
            self.variance = other.variance.copy()
        self._signature = self.compute_signature()
```

This transfers not just coefficients but also stability knowledge (prototype + variance). A seeded fractal inherits its donor's confidence level — it knows how settled its coefficients are without having to rediscover that information.

### How the Multi-Scale Demo Proves It

At step 200, the demo builds a composed fractal from the best fractal of each scale:

```python
composed = Fractal(dim=16, domain="composed", learning_rate=0.12)
composed.add_child(best_f)   # 8-dim: captures fast oscillation
composed.add_child(best_m)   # 16-dim: captures medium-scale patterns
composed.add_child(best_s)   # 40-dim: captures envelope + regime context
```

```
Composed fitness: 0.476
Composed exposures: 799
Composition mechanism: ACTIVE (fitness > 0, learning through children)
```

The composed fractal achieves positive fitness, proving that it is learning — not just passing data through.

---

## Pillar 4: Exploration

**Creating new patterns when existing ones fail.**

A system that only refines existing patterns will plateau. When the input distribution shifts, the system needs to create new fractals. This is exploration: the mechanism that expands the system's repertoire.

### Base Fractal: Novelty-Triggered Spawning ✓

Exploration is triggered by novelty — the gap between what a fractal expects and what actually arrives.

**Novelty detection**:
```python
# engine.py, lines 73-78
fractal, match_score = self._select_fractal(raw_input)

if fractal is None or match_score < (1.0 - self.novelty_threshold):
    fractal = self._spawn_fractal(raw_input)
```

The engine searches memory for the fractal whose prototype is most similar to the current input (cosine similarity). If the best match falls below the threshold, the input is novel and a new fractal is spawned.

**Spawning**:
```python
# engine.py, lines 125-140
def _spawn_fractal(self, initial_input):
    fractal = Fractal(dim=dim, domain=self.default_domain, learning_rate=self.learning_rate)
    fractal.prototype = initial_input[:dim].copy()
    self.memory.store(fractal)
    return fractal
```

The new fractal is initialized with the novel input as its prototype — it starts with a head start rather than from zeros.

### Function Discovery: Planned, Not Reactive — MODERATE

The CSV discovery pipeline does not spawn new candidates reactively when existing ones fail. Instead, it uses **tiered enumeration** — a planned exploration strategy:

| Tier | What It Tries | When |
|------|--------------|------|
| 0 | Polynomials (degree 1-3) | Always (cheapest) |
| 1 | Single-function wraps: sin(poly), exp(poly), etc. | If tier 0 fails |
| 2 | Gradient trig (warm-start refinement) | If tier 1 fails |
| 3 | Two-level chains: exp(sin(poly)), etc. | If tier 2 fails |
| 4 | Mixed inner: sin(poly + log(inner_poly)) | If tier 3 fails |
| 5 | Three-level chains: exp(sin(cos(poly))) | If tier 4 fails |

Each tier is tried only if the previous tier's best R² falls below a threshold. This is exploration — the system tries progressively more complex hypotheses — but it is *planned* exploration, not *reactive* exploration triggered by novelty detection.

**Where reactive exploration exists:** The `SymbolicEngine` streaming path does use novelty-triggered spawning (same as base Fractal). And the `PatternStore` provides cross-run exploration — patterns discovered in previous runs seed future runs, effectively remembering which corners of function space have been explored.

### How the Demos Prove It

**Sequence demo**: Exploration happens exactly three times — once each for A, B, and C. After that, every input matches one of the three prototypes. The system has exactly the right number of patterns.

```
Active fractals: 3
```

**Multi-scale demo**: Exploration is tested by the regime shift at step 500. The system responds by spawning new fractals:

```
Fast(8)    before=5  after=9   new=+4
Med(16)    before=8  after=13  new=+5
Slow(40)   before=8  after=14  new=+6
Total new fractals across all scales: +15
```

Fifteen new fractals are created in the 50 steps following the regime shift.

### The Explore-Exploit Balance

The `novelty_threshold` parameter controls how aggressively the base system explores:

- **High threshold (0.7)**: Tolerates moderate mismatch. Fewer, more general fractals.
- **Low threshold (0.3)**: Spawns aggressively. More specialized fractals.

In the function discovery pipeline, exploration is controlled by tier thresholds (`_TIER_THRESHOLDS`) and the `_SKIP_STREAM_R2 = 0.999` threshold that skips streaming entirely when a near-perfect fit is found. The system prefers the simplest explanation (Occam's razor via tiered search).

---

## Honest Assessment: The Gap and the Bridge

### The Gap

The Four Pillars were originally defined around the base Fractal's online process()/learn() cycle. The function discovery layer — which now constitutes the majority of the codebase — uses a fundamentally different approach: batch optimization.

| Pillar | Base Fractal | Function Discovery |
|--------|-------------|-------------------|
| Feedback Loops | **STRONG** — predict-observe-correct on every step | **PARTIAL** — selection (best fit wins), not correction |
| Approximability | **STRONG** — mathematically guaranteed EMA convergence | **WEAK** — pipeline-level (tiered search), not fractal-level |
| Composability | **STRONG** — parent-child hierarchies | **STRONG** — 4 composition mechanisms + transfer learning |
| Exploration | **STRONG** — novelty-triggered spawning | **MODERATE** — planned tiered enumeration, not reactive |

### The Bridge

The hybrid fix (`dim=n_coeffs`, coefficient stability tracking) bridges the gap by giving FunctionFractals meaningful inherited state:

- **Prototype** tracks the smoothed average of coefficients across fits → approximability at the fractal level
- **Variance** tracks how much coefficients jitter between fits → confidence signal for feedback
- **seed_from()** transfers both → composability through knowledge transfer

This does not make function discovery fully online — batch `fit()` is still the primary mechanism for coefficient discovery, and that is by design (batch optimization finds exact solutions that online learning cannot). But it means the Fractal's learning state is no longer dead weight. Every FunctionFractal now carries a meaningful measure of its own stability and confidence.

### Why Batch + Online Is the Right Design

Pure online learning (process → learn → update weights) converges slowly and cannot find exact solutions for functions like `sin(x² + log(x) + 3)`. Pure batch optimization (fit once, done) finds exact solutions but carries no memory of stability or confidence.

The hybrid combines both strengths:
- **Batch `fit()`** for coefficient discovery — exact solutions via polyfit, FFT, lstsq, chain inversion
- **Online prototype/variance tracking** for coefficient confidence — how stable are these coefficients across repeated fits?

This is analogous to how a scientist uses both: rigorous experiments (batch) to establish facts, and accumulated experience (online) to develop intuition about which results to trust.

---

## The Four Pillars as a Cycle

The pillars form a single continuous process. In the base Fractal layer, this cycle runs on every step. In the function discovery layer, the cycle runs across tiers and across discovery runs.

```
    Novel input arrives
           |
           v
    +--------------+       No match        +-------------+
    |   Similarity |  ------------------>  | EXPLORATION  |
    |    Search    |                       |  Spawn new   |
    |   (Memory)   |                       |   fractal    |
    +--------------+                       +-------------+
           |                                      |
        Match found                          New fractal
           |                                 initialized
           v                                      |
    +--------------+                               |
    |   FEEDBACK   |  <----------------------------+
    |    LOOP      |
    |  process()   |  Predict next input
    |  learn()     |  Compare prediction to reality
    |              |  Update weights from error
    +--------------+
           |
       Error drops
       with each
       exposure
           |
           v
    +------------------+
    | APPROXIMABILITY  |
    |  Prototype EMA   |
    |  Variance tracks |
    |  Weights converge|
    +------------------+
           |
       Fractal reaches
       high fitness
           |
           v
    +-----------------+
    | COMPOSABILITY   |
    |  Mature fractal |
    |  becomes child  |
    |  of higher-     |
    |  level fractal  |
    +-----------------+
           |
           v
      Hierarchy deepens,
      cycle continues at
      every level
```

In the function discovery layer, the analogous cycle is:

```
    (x, y) data arrives
           |
           v
    +--------------+       Cache miss       +------------------+
    |   Pattern    |  -------------------> | EXPLORATION       |
    |    Store     |                       |  Tiered search:   |
    |   (Cache)    |                       |  poly → trig →    |
    +--------------+                       |  chain → mixed    |
           |                               +------------------+
        Cache hit                                  |
        (instant)                           Best candidate
           |                                 selected
           v                                      |
    +--------------+                               |
    |   FEEDBACK   |  <----------------------------+
    |   (Hybrid)   |
    |  fit() →     |  Batch coefficient discovery
    |  prototype   |  EMA tracks coefficient stability
    |  variance    |  Variance measures confidence
    +--------------+
           |
       Prototype
       converges,
       variance shrinks
           |
           v
    +------------------+
    | APPROXIMABILITY  |
    |  Tiered: simple  |
    |  → complex       |
    |  Progressive deg |
    |  Cache learning  |
    +------------------+
           |
       Best fit found
       (R² > 0.999)
           |
           v
    +-------------------+
    | COMPOSABILITY     |
    |  Chain: F(G(poly)) |
    |  Mixed: F(p+G(q)) |
    |  Arithmetic: f±g  |
    |  Nesting: f(g(x)) |
    |  Transfer: seed() |
    +-------------------+
           |
           v
      Result cached,
      seeds future runs
```

---

## Two Demos, Three Levels of Proof

### Demo 1: Sequence Prediction (`demo_sequence.py`)

A repeating sequence A-B-C-A-B-C, 300 steps.

This demo proves the pillars work in the simplest possible setting using base Fractals:

| Pillar | Evidence |
|--------|----------|
| Feedback Loops | Error drops from 0.11 to 0.00 over 300 steps |
| Approximability | Fitness rises from 0.00 to 1.00 for all 3 fractals |
| Composability | Not exercised (single-level, no hierarchy needed) |
| Exploration | Exactly 3 fractals spawn — one per symbol, no more |

### Demo 2: Multi-Scale Signal Prediction (`demo_multiscale.py`)

A 1000-step time series with patterns at three temporal scales and a regime shift at step 500.

This demo proves the pillars work under realistic stress using base Fractals:

| Pillar | Evidence |
|--------|----------|
| Feedback Loops | Fitness > 0.84 across all engines; post-shift spike recovers within 5 steps |
| Approximability | Error ratio (late/early) below 1.0 for all three scales |
| Composability | Composed fractal achieves fitness 0.476 through 3 children at different scales |
| Exploration | 15 new fractals spawn in the 50 steps after the regime shift |

### Demo 3: Function Discovery (`python3 -m cognitive_fractal data.csv`)

Given any CSV of (x, y) data, discovers the symbolic generating function.

This demo proves the pillars in the function discovery layer:

| Pillar | Evidence |
|--------|----------|
| Feedback Loops | PARTIAL — selection feedback (best R² wins), coefficient stability tracking via prototype EMA |
| Approximability | Tiered search (simple → complex), progressive degree, cache gives instant second-run |
| Composability | **STRONG** — chains (`exp(sin(poly))`), mixed inner (`sin(x²+log(x)+3)`), transfer learning |
| Exploration | 6 tiers × multiple functions × degrees 1-3, PatternStore seeds future runs |

---

## Verification

Run both demos and check the output:

```bash
python3 demos/demo_sequence.py
# Should print: SUCCESS: System learned to predict the sequence.

python3 demos/demo_multiscale.py
# Should print: SUCCESS: All four pillars demonstrated quantitatively.
```

Run the function discovery CLI:

```bash
python3 -m cognitive_fractal your_data.csv --predict 5
# Discovers the generating function and predicts 5 future values
```

Run the test suite to verify the core library:

```bash
python3 -m pytest tests/ -v
# Should print: 228 passed
```

Every claim in this document corresponds to a measurable value in the demo output or test suite. The system either learns or it does not. The demos and tests make the answer unambiguous.
