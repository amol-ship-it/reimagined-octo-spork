# Cognitive Fractal

A recursive learning system built from self-similar fractals, grounded in the Four Pillars of Learning. Streams numeric sequences, discovers their generating functions symbolically, and predicts future values.

```
  === QUADRATIC STREAM: y = 2x^2 - 3x + 7 ===
  Step  30 | pred=1237.00 actual=1237.00 err=0.0000 | formula: 2.00*x^2 + -3.00*x + 7.00
  Step  50 | pred=4707.00 actual=4707.00 err=0.0000 | formula: 2.00*x^2 + -3.00*x + 7.00
  Predict next 5: [5107, 5507, 5917, 6337, 6767]
  STATUS: PASS

  === MANDELBROT ORBIT: z(n+1) = z(n)^2 + c ===
  Predictions stay finite: True
  STATUS: PASS
```

## The Core Idea

Intelligence is not a collection of preset rules. It is an emergent property of discovering hierarchical patterns in raw data.

This system implements that idea from first principles. The smallest building block — the **Fractal** — is a self-contained learning agent that compares, predicts, and updates itself. The same mechanism operates at every level: a fractal that detects phonemes and a fractal that detects sentences run identical code. Complexity emerges from composition, not from special-purpose modules.

## Quick Start

**Predict a sequence in three lines:**

```python
from cognitive_fractal import SequencePredictor

predictor = SequencePredictor()
# Feed a quadratic sequence: y = 2x^2 - 3x + 7
for x in range(100):
    result = predictor.feed(2*x**2 - 3*x + 7)

print(result["formula"])      # 2.00*x^2 + -3.00*x + 7.00
print(predictor.predict(5))   # Next 5 values
print(predictor.accuracy())   # MAE, fitness, formula
```

**Run the demos:**

```bash
pip install numpy
python demos/demo_mvp.py          # Sequence prediction (quadratic + Mandelbrot)
python demos/demo_symbolic.py     # Symbolic function discovery
python demos/demo_transfer.py     # Cross-stream pattern reuse
python demos/demo_improvements.py # Gradient descent, exotic functions
python demos/demo_sequence.py     # Core fractal learning (A-B-C sequence)
```

**Run the tests:**

```bash
pip install pytest
python -m pytest tests/ -v    # 175 tests
```

## Architecture

The system has two layers that build on each other.

### Layer 1: Core Fractal Engine

The foundation. Self-similar fractals that learn to recognize and predict raw signal patterns through the Four Pillars of Learning (see below).

### Layer 2: Symbolic Function Discovery

Built on the core, this layer discovers the mathematical function generating a numeric stream. Instead of matching raw vectors, it maintains a population of **candidate function fractals** — each one a hypothesis about the generating function — and evolves them through competition and composition.

```
Stream: 7, 6, 9, 16, 27, 42, 61, 84, ...
                    |
            SymbolicEngine.step()
                    |
    ┌───────────────┼───────────────┐
    │               │               │
 LinearFractal  QuadraticFractal  SinFractal ...
 y = ax + b     y = ax^2+bx+c    y = a*sin(bx+c)+d
 RMSE: 45.2     RMSE: 0.001      RMSE: 38.7
                    |
              Winner: Quadratic
              Formula: 2.00*x^2 + -3.00*x + 7.00
```

## Function Types

The system discovers functions from a library of 14 fractal types:

| Category | Types | Example |
|----------|-------|---------|
| **Basic** | Constant, Linear, Quadratic, Polynomial | `2.00*x^2 + -3.00*x + 7.00` |
| **Trigonometric** | Sin, Cos, GradientSin, GradientCos | `3.50*sin(0.80*x + 1.20) + 2.00` |
| **Exotic** | Exponential, Log | `2.10*exp(0.05*x) + -1.30` |
| **Composed (arithmetic)** | Add, Subtract, Multiply, Divide | `sin(x) + quadratic(x)` |
| **Composed (nested)** | NestedComposed | `sin(quadratic(x))` — true g(f(x)) |

**Composition is automatic.** The engine periodically attempts to combine existing candidates:
- **Arithmetic:** tries `f(x) + g(x)`, `f(x) - g(x)`, `f(x) * g(x)`, `f(x) / g(x)`
- **Nested:** tries `g(f(x))` for all viable (inner, outer) pairs — e.g. `sin(x^2 + 3)`

## The Four Pillars of Learning

Every line of code serves one of these four recursive processes.

### Pillar 1: Feedback Loops

*Learning from immediate or long-term signals.*

The engine runs a continuous cycle: **process &rarr; predict &rarr; observe outcome &rarr; learn**. Each new value serves as feedback for the previous prediction. In the symbolic layer, every incoming value triggers a refit of all candidate functions, and prediction error drives pruning.

### Pillar 2: Approximability

*Iterative refinement toward accuracy.*

Fractals start ignorant and converge. In the core layer, prototypes shift via EMA and prediction weights via gradient descent. In the symbolic layer, function coefficients are refined via least-squares and warm-start gradient descent. A fractal seen 100 times has fitness near 1.0; one seen once has fitness near 0.0.

### Pillar 3: Composability

*Reusing learned patterns in new contexts.*

Patterns are features. Composition is recursive. In the core layer, child fractals feed into parents. In the symbolic layer, existing function candidates are combined via arithmetic operations and true nested composition g(f(x)). Composed patterns are stored in shared Memory for cross-stream reuse.

### Pillar 4: Exploration

*Discovering new patterns when existing ones fail.*

When no candidate explains the input, the system spawns new fractals. The symbolic engine periodically introduces fresh candidates and exotic function types. Function signatures (32-element shape vectors) enable similarity-based retrieval from Memory, so previously learned patterns from other streams can seed new explorations.

## Transfer Learning

Multiple `SymbolicEngine` instances can share the same `Memory`. When one engine discovers a useful pattern (e.g., a sine function), it stores the pattern's signature. A second engine encountering similar data retrieves that pattern and uses it as a warm start — skipping the discovery phase entirely.

```python
from cognitive_fractal import SymbolicEngine, Memory

shared_memory = Memory(capacity=200)
engine_a = SymbolicEngine(memory=shared_memory)
engine_b = SymbolicEngine(memory=shared_memory)

# Engine A discovers sin pattern
for x in range(200):
    engine_a.step(3.0 * math.sin(0.5 * x))

# Engine B starts with a similar stream — retrieves A's pattern from memory
for x in range(200):
    engine_b.step(3.0 * math.sin(0.5 * x + 1.0))
# Engine B converges faster because it starts from A's discovery
```

## Project Structure

```
cognitive_fractal/
├── types.py             # Signal, Feedback, Metrics
├── compare.py           # Subtraction, division, cosine similarity
├── fractal.py           # The atomic learning unit (core layer)
├── memory.py            # Hot/cold tiered storage with signature similarity
├── engine.py            # Core learning loop orchestrator
├── function_fractal.py  # 14 function fractal types (linear → composed)
├── nested_fractal.py    # True nested composition g(f(x))
├── symbolic_engine.py   # Symbolic function discovery engine
├── predictor.py         # SequencePredictor — clean user-facing API
tests/
├── test_fractal.py      # Core fractal: process, learn, convergence
├── test_composition.py  # Hierarchical fractals: children, propagation
├── test_memory.py       # Store, retrieve, evict, promote, similarity
├── test_engine.py       # Core learning loop, sequence learning
├── test_symbolic.py     # Symbolic engine: discovery, composition, memory
├── test_transfer.py     # Cross-stream transfer learning
├── test_improvements.py # Gradient descent, exotic types, library restore
├── test_predictor.py    # SequencePredictor, nested fractal, MVP streams
demos/
├── demo_mvp.py          # Sequence prediction: quadratic + Mandelbrot
├── demo_symbolic.py     # Symbolic discovery: linear, quadratic, sin, poly
├── demo_transfer.py     # Cross-stream pattern reuse
├── demo_improvements.py # Gradient fitting, exotic functions, composed reuse
├── demo_sequence.py     # Core learning: A-B-C-A-B-C sequence
└── demo_multiscale.py   # Multi-scale signal prediction
```

## Requirements

- Python 3.9+
- numpy

No frameworks, no GPU, no external services. Pure Python + numpy.
