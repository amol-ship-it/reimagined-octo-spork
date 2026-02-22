"""Tests for composed (hierarchical) fractals."""

import numpy as np
import time
import pytest

from cognitive_fractal import Fractal, Signal, Feedback


class TestComposition:
    def test_add_child_makes_composed(self):
        parent = Fractal(dim=8)
        child = Fractal(dim=4)
        parent.add_child(child)

        assert parent.is_composed
        assert not parent.is_leaf
        assert child.parent is parent

    def test_composed_processes_through_children(self):
        child1 = Fractal(dim=4)
        child2 = Fractal(dim=4)
        parent = Fractal(dim=8)

        parent.add_child(child1)
        parent.add_child(child2)

        signal = Signal(data=np.ones(4), timestamp=0.0)
        output, novelty = parent.process(signal)

        # Parent should produce output
        assert output.data.shape == (8,)
        # Children should have been processed
        assert child1.metrics.total_exposures == 1
        assert child2.metrics.total_exposures == 1

    def test_children_exposure_counts_accumulate(self):
        child = Fractal(dim=4)
        parent = Fractal(dim=4)
        parent.add_child(child)

        signal = Signal(data=np.ones(4), timestamp=0.0)
        for _ in range(5):
            parent.process(signal)

        assert child.metrics.total_exposures == 5
        assert parent.metrics.total_exposures == 5

    def test_feedback_propagates_to_children(self):
        child = Fractal(dim=4, learning_rate=0.5)
        parent = Fractal(dim=4, learning_rate=0.5)
        parent.add_child(child)

        initial_child_proto = child.prototype.copy()

        signal = Signal(data=np.array([1.0, 0.0, 0.0, 0.0]), timestamp=0.0)
        parent.process(signal)

        feedback = Feedback(
            actual=np.array([0.0, 1.0, 0.0, 0.0]),
            reward=1.0,
            timestamp=0.1,
        )
        parent.learn(feedback)

        # Child prototype should have moved
        assert not np.array_equal(child.prototype, initial_child_proto)

    def test_composed_fractal_learns(self):
        """A composed fractal should show decreasing error over time."""
        child1 = Fractal(dim=4, learning_rate=0.15)
        child2 = Fractal(dim=4, learning_rate=0.15)
        parent = Fractal(dim=8, learning_rate=0.15)
        parent.add_child(child1)
        parent.add_child(child2)

        pattern = np.array([1.0, 0.5, 0.0, 0.0])
        next_pattern = np.array([0.0, 0.0, 0.5, 1.0])

        errors = []
        for _ in range(50):
            signal = Signal(data=pattern, timestamp=time.time())
            parent.process(signal)
            feedback = Feedback(
                actual=next_pattern, reward=1.0, timestamp=time.time()
            )
            error = parent.learn(feedback)
            errors.append(error)

        early = np.mean(errors[:5])
        late = np.mean(errors[-5:])
        assert late < early, (
            f"Composed fractal didn't learn: early={early:.4f}, late={late:.4f}"
        )

    def test_deep_hierarchy(self):
        """Three-level hierarchy: grandchild -> child -> parent."""
        grandchild = Fractal(dim=4)
        child = Fractal(dim=4)
        parent = Fractal(dim=4)

        child.add_child(grandchild)
        parent.add_child(child)

        signal = Signal(data=np.ones(4), timestamp=0.0)
        output, _ = parent.process(signal)

        assert output.data.shape == (4,)
        assert grandchild.metrics.total_exposures == 1
        assert child.metrics.total_exposures == 1
        assert parent.metrics.total_exposures == 1

    def test_aggregation_handles_dimension_mismatch(self):
        """Children with different output dims still compose correctly."""
        child1 = Fractal(dim=3)  # Outputs 3-dim
        child2 = Fractal(dim=5)  # Outputs 5-dim
        parent = Fractal(dim=4)  # Expects 4-dim feature space

        parent.add_child(child1)
        parent.add_child(child2)

        signal = Signal(data=np.ones(5), timestamp=0.0)
        output, _ = parent.process(signal)
        assert output.data.shape == (4,)
