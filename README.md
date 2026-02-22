# Cognitive Fractal

A recursive learning system built from self-similar fractals, grounded in the Four Pillars of Learning.

```
  Steps   1- 30: avg_error=0.1141  accuracy= 89.7%  fitness=0.3038  fractals=3
  Steps  91-120: avg_error=0.0000  accuracy= 97.5%  fitness=0.9617  fractals=3
  Steps 271-300: avg_error=0.0000  accuracy= 99.0%  fitness=0.9999  fractals=3

  LEARNED FRACTALS:
    e0fb0691: prototype~A  fitness=1.000  exposures=100
    8cf02b2d: prototype~B  fitness=1.000  exposures=100
    bf993183: prototype~C  fitness=1.000  exposures=100

  SUCCESS: System learned to predict the sequence.
```

## The Core Idea

Intelligence is not a collection of preset rules. It is an emergent property of discovering hierarchical patterns in raw data.

This system implements that idea from first principles. The smallest building block — the **Fractal** — is a self-contained learning agent that compares, predicts, and updates itself. The same mechanism operates at every level: a fractal that detects phonemes and a fractal that detects sentences run identical code. Complexity emerges from composition, not from special-purpose modules.

## The Four Pillars of Learning

The theoretical foundation defines four recursive processes that must operate continuously for a system to be self-improving. Every line of code in this project exists to serve one of these pillars.

### Pillar 1: Feedback Loops

*The basic stimulus/response mechanism. Learning from immediate or long-term signals.*

A fractal that cannot close the loop between prediction and outcome cannot learn. The feedback loop is the heartbeat of the system.

**How it works in the code:**

The engine runs a continuous cycle: **process &rarr; predict &rarr; observe outcome &rarr; learn**.

```
engine.step(input_A)  # Fractal processes A, predicts "B will come next"
engine.step(input_B)  # B arrives — this IS the feedback for the previous prediction
                      # The fractal that predicted B now learns from the error
```

At each step, the current input serves as the actual outcome of the previous prediction. This closes the loop naturally without requiring an external teacher.

Inside `Fractal.learn()`, the feedback drives three concrete updates:

| Update | What changes | How |
|--------|-------------|-----|
| Prototype shift | What the fractal "expects" | Exponential moving average toward recent input |
| Variance tracking | How spread out inputs are | Running estimate of per-dimension variance |
| Prediction weights | Input &rarr; next-input mapping | Outer-product gradient rule on prediction error |

The prediction error is tracked as an exponential moving average in `Metrics.prediction_error_ema`, giving a real-time measure of how well the fractal is closing its own loop.

**Key files:** [`fractal.py`](cognitive_fractal/fractal.py) (`process()` + `learn()`), [`engine.py`](cognitive_fractal/engine.py) (`step()`)

---

### Pillar 2: Approximability

*The ability to iterate upon a pattern to improve its accuracy and utility over time.*

A fractal starts ignorant. Its prototype is initialized to the first input it sees, its prediction weights are near-zero, and its fitness is 0.0. Through repeated exposure and feedback, it converges toward an accurate model of its input distribution.

**How it works in the code:**

Every call to `learn()` nudges the fractal closer to the truth:

```python
# Prototype converges via EMA — each exposure sharpens the estimate
self.prototype += learning_rate * (last_input - self.prototype)

# Prediction weights converge via gradient descent on prediction error
pred_error = actual - predicted
self.prediction_weights += learning_rate * outer(pred_error, normalized_input)
self._prediction_bias += learning_rate * pred_error
```

This is not a one-shot adjustment. It is iterative refinement — the same input seen 100 times produces a fractal with fitness approaching 1.0, while a fractal that has seen an input once has fitness near 0.0. The approximation improves with every cycle.

The variance tracker adds a second dimension of refinement. A fractal that has seen many inputs knows not just the center of its distribution (prototype) but also how much each dimension varies. This prevents false novelty signals when input naturally fluctuates.

**Measurable proof:**

```
Step   1: prediction_error=0.11  fitness=0.00
Step  30: prediction_error=0.00  fitness=0.30
Step 120: prediction_error=0.00  fitness=0.96
Step 300: prediction_error=0.00  fitness=1.00
```

**Key files:** [`fractal.py`](cognitive_fractal/fractal.py) (`learn()` — Updates 1, 2, 3), [`types.py`](cognitive_fractal/types.py) (`Metrics.fitness()`)

---

### Pillar 3: Composability

*Reusing learned patterns in new contexts. Abstraction allows old solutions to solve new problems.*

Patterns are features. Once a pattern is learned, it is abstract and domain-agnostic — it can be applied in novel contexts. Composition is recursive: new patterns are formed by leveraging existing sub-patterns, the way bricks compose into walls, walls into buildings, buildings into cities.

**How it works in the code:**

A composed fractal routes input through its children, then operates on their outputs:

```python
# A parent fractal with two children
parent = Fractal(dim=8)
parent.add_child(child_phoneme_detector)   # Detects low-level sound patterns
parent.add_child(child_rhythm_detector)    # Detects timing patterns

# When parent.process() is called:
#   1. Input flows through BOTH children (each runs process())
#   2. Children's predictions are concatenated into a feature vector
#   3. Parent compares THAT to its own prototype
#   4. Parent makes its own prediction in the composed feature space
```

The critical property: **the parent and children run identical code**. `Fractal.process()` does not check whether it is a leaf or a root — it simply processes. If it has children, it delegates first. If not, it operates on raw input. This is structural self-similarity, not a metaphor.

Feedback propagates recursively too. When a parent learns, it calls `child.learn()` for each child, so the entire hierarchy improves simultaneously.

```
Leaf fractals       →  detect edges, phonemes, raw features
Composed fractals   →  detect words, patterns-of-patterns
Deeper composition  →  detect sentences, abstract concepts
```

The output of one level becomes the input for the next. This is the recursive building block hierarchy from the theoretical framework — atomic patterns compose into sub-patterns, sub-patterns into high-level abstractions.

**Key files:** [`fractal.py`](cognitive_fractal/fractal.py) (`add_child()`, `process()` composed branch, `_aggregate_child_outputs()`), [`engine.py`](cognitive_fractal/engine.py) (`compose()`)

---

### Pillar 4: Exploration

*Building new patterns from scratch or by combining existing ones in novel ways.*

A system that only refines existing patterns will never discover new ones. Exploration is the mechanism that creates fresh fractals when the current population cannot explain the input.

**How it works in the code:**

The engine measures how well existing fractals match each incoming input using cosine similarity against their learned prototypes. When nothing matches well, a new fractal is born:

```python
# Inside engine.step():
fractal, match_score = self._select_fractal(raw_input)

if fractal is None or match_score < (1.0 - self.novelty_threshold):
    # Nothing in memory explains this input — EXPLORE
    fractal = self._spawn_fractal(raw_input)
```

The new fractal is initialized with the novel input as its prototype, giving it a head start. From there, the other three pillars take over: feedback loops close around it, approximability refines it, and composability may eventually fold it into a hierarchy.

The `novelty_threshold` parameter controls the explore/exploit tradeoff. A high threshold (0.7) means the system tolerates moderate mismatch before spawning — it tries to reuse what it has. A low threshold (0.3) makes it spawn aggressively, creating more specialized fractals.

In the sequence demo, exploration happens exactly three times — once for A, once for B, once for C. After that, the system exploits its three learned fractals for the remaining 297 steps, refining them from fitness 0.0 to 1.0.

**Key files:** [`engine.py`](cognitive_fractal/engine.py) (`_select_fractal()`, `_spawn_fractal()`), [`compare.py`](cognitive_fractal/compare.py) (`similarity()`)

---

## The Four Pillars Working Together

The pillars are not independent features. They form a continuous cycle, exactly as described in the theoretical framework:

```
         ┌─────────────────────────────────────────┐
         │                                         │
   Input arrives                                   │
         │                                         │
         ▼                                         │
   ┌───────────┐    match    ┌───────────────┐     │
   │  Pattern   │───────────▶│   Reinforce   │     │
   │   Match    │            │  (Feedback +  │     │
   │(Similarity)│            │ Approximation)│     │
   └───────────┘            └───────────────┘     │
         │                         │               │
      no match                  updated            │
         │                      fractal            │
         ▼                         │               │
   ┌───────────┐                   │               │
   │  Explore  │                   ▼               │
   │  (Spawn   │            ┌───────────────┐      │
   │   new)    │            │  Integration  │      │
   └───────────┘            │  (Store in    │      │
         │                  │   Memory)     │      │
         │                  └───────────────┘      │
         │                         │               │
         └─────────────────────────┘───────────────┘
                                         │
                              Compose into hierarchy
                              when patterns co-occur
```

1. Input arrives. The engine searches memory for a matching fractal (**Composability** — reuse what you know).
2. If a match is found, the fractal processes the input and receives feedback (**Feedback Loops**). Its prototype and weights improve (**Approximability**).
3. If no match is found, a new fractal is spawned (**Exploration**) and integrated into memory.
4. Over time, co-occurring fractals can be composed into hierarchies (**Composability** again), creating higher-level abstractions.

## Dual Memory Architecture

The system implements two types of memory, mirroring the theoretical distinction between patterns and state.

**Type A — Patterns (The Code):** Each fractal's learned state — its prototype, variance, and prediction weights — is a compressed, reusable abstraction. This is stored in hot memory as live objects or in cold memory as serialized dicts via `fractal.compress()`.

**Type B — State (The Data):** The engine's context buffer holds a sliding window of recent signals. This is dynamic, temporal context — what happened recently — as opposed to the timeless patterns stored in fractals.

**The Archival Mechanism:** When hot memory reaches capacity, low-fitness fractals are compressed and moved to cold storage. High-fitness fractals and children of active composed fractals are protected. Cold fractals can be promoted back to hot when accessed.

## Project Structure

```
cognitive_fractal/
├── types.py       # Signal, Feedback, Metrics — the data flowing through the system
├── compare.py     # Subtraction (edges), division (ratios), cosine similarity
├── fractal.py     # The atomic learning unit — where all four pillars live
├── memory.py      # Hot/cold tiered storage with learned-prototype similarity
├── engine.py      # Learning loop orchestrator — select, spawn, compose
tests/
├── test_fractal.py       # Atomic fractal: process, learn, convergence
├── test_composition.py   # Hierarchical fractals: children, propagation
├── test_memory.py        # Store, retrieve, evict, promote, similarity
├── test_engine.py        # Full learning loop, sequence learning, spawning
demos/
└── demo_sequence.py      # Learn A-B-C-A-B-C — proof of actual learning
```

## Quick Start

```bash
pip install numpy
python demos/demo_sequence.py
```

Run the tests:

```bash
pip install pytest
python -m pytest tests/ -v
```

## Requirements

- Python 3.9+
- numpy

No frameworks, no GPU, no external services. Pure Python + numpy.
