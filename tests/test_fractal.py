"""Tests for the atomic Fractal learning unit."""

import numpy as np
import time
import pytest

from cognitive_fractal import Fractal, Signal, Feedback


class TestFractalCreation:
    def test_creates_with_correct_dimensions(self):
        f = Fractal(dim=8)
        assert f.dim == 8
        assert f.prototype.shape == (8,)
        assert f.variance.shape == (8,)
        assert f.prediction_weights.shape == (8, 8)

    def test_starts_as_leaf(self):
        f = Fractal(dim=4)
        assert f.is_leaf
        assert not f.is_composed

    def test_unique_ids(self):
        f1 = Fractal(dim=4)
        f2 = Fractal(dim=4)
        assert f1.id != f2.id

    def test_initial_fitness_is_zero(self):
        f = Fractal(dim=4)
        assert f.metrics.fitness() == 0.0


class TestFractalProcess:
    def test_produces_output_signal(self):
        f = Fractal(dim=4)
        signal = Signal(data=np.array([1.0, 0.0, 0.0, 0.0]), timestamp=0.0)
        output, novelty = f.process(signal)
        assert isinstance(output, Signal)
        assert output.data.shape == (4,)
        assert isinstance(novelty, float)

    def test_novelty_is_high_for_new_input(self):
        f = Fractal(dim=4)
        # Prototype is zeros, input is far from it
        signal = Signal(data=np.array([1.0, 1.0, 1.0, 1.0]), timestamp=0.0)
        _, novelty = f.process(signal)
        assert novelty > 0.5

    def test_novelty_is_low_for_matching_input(self):
        f = Fractal(dim=4)
        f.prototype = np.array([1.0, 0.0, 0.0, 0.0])
        signal = Signal(data=np.array([1.0, 0.0, 0.0, 0.0]), timestamp=0.0)
        _, novelty = f.process(signal)
        assert novelty < 0.01

    def test_increments_exposure_count(self):
        f = Fractal(dim=4)
        signal = Signal(data=np.zeros(4), timestamp=0.0)
        f.process(signal)
        assert f.metrics.total_exposures == 1
        f.process(signal)
        assert f.metrics.total_exposures == 2

    def test_handles_input_dimension_mismatch(self):
        f = Fractal(dim=4)
        # Larger input: should truncate
        signal = Signal(data=np.ones(8), timestamp=0.0)
        output, _ = f.process(signal)
        assert output.data.shape == (4,)

        # Smaller input: should pad
        signal = Signal(data=np.ones(2), timestamp=0.0)
        output, _ = f.process(signal)
        assert output.data.shape == (4,)


class TestFractalLearn:
    def test_prototype_moves_toward_input(self):
        f = Fractal(dim=4, learning_rate=0.5)
        signal = Signal(data=np.array([1.0, 0.0, 0.0, 0.0]), timestamp=0.0)
        f.process(signal)

        feedback = Feedback(
            actual=np.array([0.0, 1.0, 0.0, 0.0]), reward=1.0, timestamp=0.1
        )
        f.learn(feedback)

        # Prototype should have moved from [0,0,0,0] toward [1,0,0,0]
        assert f.prototype[0] > 0.0

    def test_prediction_error_decreases_with_repetition(self):
        """Core test: prove the system actually learns."""
        f = Fractal(dim=4, learning_rate=0.15)
        pattern = np.array([1.0, 0.0, 0.0, 0.0])
        next_pattern = np.array([0.0, 1.0, 0.0, 0.0])

        errors = []
        for _ in range(50):
            signal = Signal(data=pattern, timestamp=time.time())
            f.process(signal)
            feedback = Feedback(
                actual=next_pattern, reward=1.0, timestamp=time.time()
            )
            error = f.learn(feedback)
            errors.append(error)

        # Error should decrease over repetitions
        early_error = np.mean(errors[:5])
        late_error = np.mean(errors[-5:])
        assert late_error < early_error, (
            f"Learning failed: early={early_error:.4f}, late={late_error:.4f}"
        )

    def test_returns_zero_error_without_prior_process(self):
        f = Fractal(dim=4)
        feedback = Feedback(
            actual=np.zeros(4), reward=0.0, timestamp=0.0
        )
        error = f.learn(feedback)
        assert error == 0.0

    def test_variance_updates(self):
        f = Fractal(dim=4)
        initial_var = f.variance.copy()

        signal = Signal(data=np.array([2.0, 0.0, 0.0, 0.0]), timestamp=0.0)
        f.process(signal)
        feedback = Feedback(
            actual=np.zeros(4), reward=0.0, timestamp=0.1
        )
        f.learn(feedback)

        # Variance should have changed
        assert not np.array_equal(f.variance, initial_var)

    def test_metrics_accuracy_improves(self):
        f = Fractal(dim=4, learning_rate=0.2)
        pattern = np.array([1.0, 0.0, 0.0, 0.0])

        for _ in range(30):
            signal = Signal(data=pattern, timestamp=time.time())
            f.process(signal)
            feedback = Feedback(
                actual=pattern, reward=1.0, timestamp=time.time()
            )
            f.learn(feedback)

        # After many exposures to the same pattern, accuracy should be high
        assert f.metrics.accuracy_ema > 0.3


class TestFractalSerialization:
    def test_compress_decompress_roundtrip(self):
        f = Fractal(dim=4, domain="test", learning_rate=0.2)
        f.prototype = np.array([1.0, 2.0, 3.0, 4.0])
        f.metrics.total_exposures = 42

        compressed = f.compress()
        restored = Fractal.decompress(compressed)

        assert restored.id == f.id
        assert restored.domain == f.domain
        assert restored.dim == f.dim
        assert restored.learning_rate == f.learning_rate
        np.testing.assert_array_almost_equal(
            restored.prototype, f.prototype
        )
        np.testing.assert_array_almost_equal(
            restored.prediction_weights, f.prediction_weights
        )
        assert restored.metrics.total_exposures == 42

    def test_compressed_form_is_serializable(self):
        """Compressed dict should contain only basic Python types."""
        import json

        f = Fractal(dim=4)
        compressed = f.compress()
        # Should not raise
        json_str = json.dumps(compressed)
        assert isinstance(json_str, str)
