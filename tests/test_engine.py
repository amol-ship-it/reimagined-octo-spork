"""Tests for the FractalEngine learning loop."""

import numpy as np
import pytest

from cognitive_fractal import FractalEngine, Fractal


class TestEngineBasics:
    def test_first_step_spawns_fractal(self):
        engine = FractalEngine(dim=4)
        prediction, diag = engine.step(np.array([1.0, 0.0, 0.0, 0.0]))

        assert prediction.shape == (4,)
        assert diag["active_fractals"] == 1
        assert diag["step"] == 1

    def test_similar_inputs_reuse_fractal(self):
        engine = FractalEngine(dim=4, novelty_threshold=0.5)

        # First input spawns a fractal
        engine.step(np.array([1.0, 0.0, 0.0, 0.0]))
        first_count = engine.memory.stats()["hot_count"]

        # Very similar input should reuse the same fractal
        engine.step(np.array([0.95, 0.05, 0.0, 0.0]))
        assert engine.memory.stats()["hot_count"] == first_count

    def test_novel_input_spawns_new_fractal(self):
        engine = FractalEngine(dim=4, novelty_threshold=0.5)

        engine.step(np.array([1.0, 0.0, 0.0, 0.0]))
        engine.step(np.array([0.0, 0.0, 0.0, 1.0]))

        # Should have spawned 2 fractals
        assert engine.memory.stats()["hot_count"] == 2

    def test_step_count_increments(self):
        engine = FractalEngine(dim=4)
        for i in range(5):
            _, diag = engine.step(np.zeros(4))
            assert diag["step"] == i + 1


class TestEngineLearning:
    def test_sequence_learning(self):
        """Core integration test: engine learns A-B-C-A-B-C pattern."""
        engine = FractalEngine(
            dim=8, novelty_threshold=0.7, learning_rate=0.15
        )

        a = np.zeros(8); a[0] = 1.0
        b = np.zeros(8); b[1] = 1.0
        c = np.zeros(8); c[2] = 1.0
        sequence = [a, b, c] * 50  # 150 steps

        errors = []
        for vec in sequence:
            _, diag = engine.step(vec)
            errors.append(diag["prediction_error"])

        # Compare early vs late error (skip first 3 steps: no feedback yet)
        early = np.mean(errors[3:30])
        late = np.mean(errors[-30:])
        assert late < early, (
            f"Engine didn't learn: early={early:.4f}, late={late:.4f}"
        )

    def test_prediction_improves_over_time(self):
        """Previous step's prediction should get closer to current input."""
        engine = FractalEngine(dim=4, novelty_threshold=0.7, learning_rate=0.2)

        pattern_a = np.array([1.0, 0.0, 0.0, 0.0])
        pattern_b = np.array([0.0, 1.0, 0.0, 0.0])

        # Alternating A-B-A-B...
        sequence = [pattern_a, pattern_b] * 50

        predictions_vs_actual = []
        prev_pred = None
        for i, vec in enumerate(sequence):
            pred, _ = engine.step(vec)
            if prev_pred is not None:
                # How close was the PREVIOUS prediction to THIS actual input?
                error = np.sqrt(np.mean((prev_pred - vec) ** 2))
                predictions_vs_actual.append(error)
            prev_pred = pred

        # Average early vs late
        early = np.mean(predictions_vs_actual[:10])
        late = np.mean(predictions_vs_actual[-10:])
        # Late predictions should be better (lower error)
        assert late < early, (
            f"Predictions didn't improve: early={early:.4f}, late={late:.4f}"
        )

    def test_context_buffer_grows(self):
        engine = FractalEngine(dim=4, domain="test")
        engine.context_window = 5

        for _ in range(10):
            engine.step(np.ones(4))

        assert len(engine.context_buffer) == 5  # Capped at window size


class TestEngineComposition:
    def test_compose_creates_parent(self):
        engine = FractalEngine(dim=4)

        child1 = Fractal(dim=4, learning_rate=0.1)
        child2 = Fractal(dim=4, learning_rate=0.1)
        engine.memory.store(child1)
        engine.memory.store(child2)

        parent = engine.compose([child1, child2], parent_dim=8)

        assert parent.is_composed
        assert len(parent.children) == 2
        assert parent.id in engine.memory.hot

    def test_get_stats(self):
        engine = FractalEngine(dim=4)
        engine.step(np.ones(4))
        stats = engine.get_stats()

        assert stats["step_count"] == 1
        assert "memory" in stats
        assert stats["memory"]["hot_count"] >= 1
