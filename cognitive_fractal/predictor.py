"""
SequencePredictor — clean wrapper for mathematical sequence prediction.

Wraps SymbolicEngine with a simple API:
  - feed(value) -> dict   : feed one value, get prediction + diagnostics
  - predict(n) -> ndarray : predict next n values
  - accuracy() -> dict    : get accuracy metrics

Works out of the box for any mathematical sequence. Internally maintains
a population of candidate function hypotheses, discovers the generating
function, and extrapolates to predict future values.
"""

import numpy as np
from typing import Optional, List

from .symbolic_engine import SymbolicEngine
from .memory import Memory


class SequencePredictor:
    """Predict mathematical sequences by discovering their generating function.

    Internally maintains a SymbolicEngine that fits a population of
    candidate functions to the observed data stream.

    Usage:
        sp = SequencePredictor()
        for value in my_sequence:
            result = sp.feed(value)
            print(result['formula'], result['prediction'])
        future = sp.predict(10)
    """

    def __init__(
        self,
        predict_ahead: int = 5,
        window_size: int = 50,
        auto_compose: bool = True,
        max_candidates: int = 25,
        memory: Optional[Memory] = None,
    ):
        composition_interval = 40 if auto_compose else 999999
        nested_interval = 80 if auto_compose else 999999

        self._engine = SymbolicEngine(
            window_size=window_size,
            predict_horizon=predict_ahead,
            max_candidates=max_candidates,
            composition_interval=composition_interval,
            composition_threshold=0.3,
            nested_composition_interval=nested_interval,
            memory=memory,
        )
        self._predict_ahead = predict_ahead
        self._step_count = 0
        self._predictions_history: List[dict] = []
        self._error_history: List[float] = []
        self._prev_prediction: Optional[float] = None

    def feed(self, value: float) -> dict:
        """Feed one value from the sequence.

        Returns a dict with:
          - prediction: next predicted value (float)
          - formula: best discovered formula string
          - fitness: fitness of the best candidate [0, 1]
          - step: current step number
          - error: prediction error for this step (0 on first step)
          - num_candidates: active candidate count
        """
        self._step_count += 1

        # Compute error from previous prediction
        error = 0.0
        if self._prev_prediction is not None:
            error = abs(self._prev_prediction - value)
            self._error_history.append(error)

        pred, diag = self._engine.step(float(value))

        self._prev_prediction = pred

        result = {
            "prediction": pred,
            "formula": diag["best_formula"],
            "fitness": diag["best_fitness"],
            "step": diag["step"],
            "error": error,
            "num_candidates": diag["num_candidates"],
        }
        self._predictions_history.append(result)
        return result

    def predict(self, n: int) -> np.ndarray:
        """Predict the next n values using the best discovered function."""
        return self._engine.get_predictions(n)

    def accuracy(self) -> dict:
        """Return accuracy metrics for the predictor.

        Returns:
          - best_formula: symbolic formula of the best candidate
          - best_fitness: fitness score [0, 1]
          - mean_absolute_error: MAE over all predictions so far
          - recent_mae: MAE over last 20 predictions
          - num_candidates: active candidate count
          - steps: total steps processed
        """
        formula, fitness, best = self._engine.get_best()
        errors = self._error_history

        mae = float(np.mean(errors)) if errors else 0.0
        recent = errors[-20:] if len(errors) >= 20 else errors
        recent_mae = float(np.mean(recent)) if recent else 0.0

        return {
            "best_formula": formula,
            "best_fitness": fitness,
            "mean_absolute_error": mae,
            "recent_mae": recent_mae,
            "num_candidates": len(self._engine.candidates),
            "steps": self._step_count,
        }

    def feed_sequence(self, values) -> dict:
        """Feed a batch of values at once.

        Returns the result dict from the last fed value.
        """
        result = {
            "prediction": 0.0, "formula": "none", "fitness": 0.0,
            "step": 0, "error": 0.0, "num_candidates": 0,
        }
        for v in values:
            result = self.feed(float(v))
        return result

    def get_stats(self) -> dict:
        """Return detailed engine statistics."""
        return self._engine.get_stats()
