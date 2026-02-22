"""
The FractalEngine — orchestrates the lifecycle of fractals.

IMPORTANT: This engine does NOT contain learning logic.
Learning lives inside each Fractal. The engine's job is:

  1. Route incoming signals to the right fractal(s)
  2. Trigger the forward pass (process)
  3. Trigger the backward pass (learn) when feedback arrives
  4. Manage the fractal hierarchy (create, compose, retire)
  5. Decide when to spawn new fractals (novelty-driven exploration)

The engine implements the Four Pillars through orchestration:
  - Feedback Loops:   process -> feedback -> learn cycle
  - Approximability:  fractals start ignorant and converge via EMA
  - Composability:    engine builds hierarchies by composing fractals
  - Exploration:      high novelty triggers new fractal creation
"""

import numpy as np
import time
from typing import Optional, List, Tuple

from .fractal import Fractal
from .memory import Memory
from .types import Signal, Feedback


class FractalEngine:
    """Orchestrates the learning loop over a population of fractals."""

    def __init__(
        self,
        dim: int = 8,
        domain: str = "default",
        novelty_threshold: float = 0.5,
        learning_rate: float = 0.1,
    ):
        self.default_dim = dim
        self.default_domain = domain
        self.novelty_threshold = novelty_threshold
        self.learning_rate = learning_rate

        self.memory = Memory()

        # Type B memory: contextual, temporal state (sliding window)
        self.context_buffer: List[Signal] = []
        self.context_window: int = 20

        # Tracking
        self._step_count = 0
        self._pending_fractal: Optional[Fractal] = None
        self._pending_prediction: Optional[np.ndarray] = None

    def step(self, raw_input: np.ndarray) -> Tuple[np.ndarray, dict]:
        """One full cycle of the learning loop.

        1. Wrap raw input as Signal
        2. Find the best-matching fractal (or create one)
        3. Process input through fractal -> get prediction + novelty
        4. If we have a previous prediction, compute feedback and learn
        5. Return the prediction and diagnostics

        The current input IS the actual outcome of the previous prediction.
        This closes the feedback loop naturally for sequence learning.
        """
        now = time.time()
        self._step_count += 1

        input_signal = Signal(data=raw_input.copy(), timestamp=now)

        # --- Find best matching fractal ---
        fractal, match_score = self._select_fractal(raw_input)

        # --- If nothing matches well, spawn (Exploration) ---
        if fractal is None or match_score < (1.0 - self.novelty_threshold):
            fractal = self._spawn_fractal(raw_input)
            match_score = 1.0  # Perfect match since we initialized to input

        # --- Forward pass ---
        output_signal, novelty = fractal.process(input_signal)

        # --- Feedback from previous step ---
        error = 0.0
        if self._pending_fractal is not None:
            feedback = Feedback(
                actual=raw_input.copy(),
                reward=0.0,
                timestamp=now,
            )
            error = self._pending_fractal.learn(feedback)

        # --- Remember for next step's feedback ---
        self._pending_fractal = fractal
        self._pending_prediction = output_signal.data.copy()

        # --- Context buffer (Type B memory) ---
        self.context_buffer.append(input_signal)
        if len(self.context_buffer) > self.context_window:
            self.context_buffer.pop(0)

        diagnostics = {
            "step": self._step_count,
            "fractal_id": fractal.id,
            "novelty": novelty,
            "prediction_error": error,
            "match_score": match_score,
            "active_fractals": len(self.memory.hot),
            "fitness": fractal.metrics.fitness(),
        }

        return output_signal.data, diagnostics

    def _select_fractal(
        self, query: np.ndarray
    ) -> Tuple[Optional[Fractal], float]:
        """Find the fractal whose prototype best matches the query."""
        results = self.memory.find_similar(
            query, domain=self.default_domain, top_k=1
        )
        if results:
            return results[0]  # (fractal, similarity_score)
        return None, 0.0

    def _spawn_fractal(self, initial_input: np.ndarray) -> Fractal:
        """Create a new fractal initialized with the input as its prototype.

        This is exploration: the system encounters something novel,
        so it creates a fresh learning unit to model it.
        """
        dim = min(len(initial_input), self.default_dim)
        fractal = Fractal(
            dim=dim,
            domain=self.default_domain,
            learning_rate=self.learning_rate,
        )
        # Initialize prototype to the first observation (not zeros)
        fractal.prototype = initial_input[:dim].copy()
        self.memory.store(fractal)
        return fractal

    def compose(
        self,
        children: List[Fractal],
        parent_domain: Optional[str] = None,
        parent_dim: Optional[int] = None,
    ) -> Fractal:
        """Create a composed fractal over existing children.

        The parent operates in the space of children's outputs.
        Its prototype is over the concatenated/aggregated child output space.
        """
        if parent_dim is None:
            parent_dim = self.default_dim
        if parent_domain is None:
            parent_domain = self.default_domain

        parent = Fractal(
            dim=parent_dim,
            domain=parent_domain,
            learning_rate=self.learning_rate,
        )
        for child in children:
            parent.add_child(child)

        self.memory.store(parent)
        return parent

    def get_stats(self) -> dict:
        """Return engine and memory statistics."""
        return {
            "step_count": self._step_count,
            "memory": self.memory.stats(),
            "context_buffer_len": len(self.context_buffer),
        }
