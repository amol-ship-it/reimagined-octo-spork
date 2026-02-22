"""Tests for the dual-tier memory system."""

import numpy as np
import pytest

from cognitive_fractal import Fractal, Memory


class TestMemoryBasics:
    def test_store_and_retrieve(self):
        mem = Memory()
        f = Fractal(dim=4)
        mem.store(f)

        retrieved = mem.get(f.id)
        assert retrieved is f

    def test_returns_none_for_unknown_id(self):
        mem = Memory()
        assert mem.get("nonexistent") is None

    def test_contains(self):
        mem = Memory()
        f = Fractal(dim=4)
        mem.store(f)
        assert mem.contains(f.id)
        assert not mem.contains("nonexistent")

    def test_stats(self):
        mem = Memory()
        f = Fractal(dim=4)
        mem.store(f)
        stats = mem.stats()
        assert stats["hot_count"] == 1
        assert stats["cold_count"] == 0


class TestSimilaritySearch:
    def test_finds_similar_fractals(self):
        mem = Memory()

        f1 = Fractal(dim=4)
        f1.prototype = np.array([1.0, 0.0, 0.0, 0.0])
        mem.store(f1)

        f2 = Fractal(dim=4)
        f2.prototype = np.array([0.0, 1.0, 0.0, 0.0])
        mem.store(f2)

        # Query close to f1
        results = mem.find_similar(np.array([0.9, 0.1, 0.0, 0.0]))
        assert len(results) == 2
        assert results[0][0] is f1  # f1 should be most similar
        assert results[0][1] > results[1][1]

    def test_filters_by_domain(self):
        mem = Memory()

        f1 = Fractal(dim=4, domain="audio")
        f1.prototype = np.array([1.0, 0.0, 0.0, 0.0])
        mem.store(f1)

        f2 = Fractal(dim=4, domain="vision")
        f2.prototype = np.array([1.0, 0.0, 0.0, 0.0])
        mem.store(f2)

        results = mem.find_similar(
            np.array([1.0, 0.0, 0.0, 0.0]), domain="audio"
        )
        assert len(results) == 1
        assert results[0][0] is f1

    def test_respects_top_k(self):
        mem = Memory()
        for i in range(10):
            f = Fractal(dim=4)
            f.prototype = np.random.randn(4)
            mem.store(f)

        results = mem.find_similar(np.ones(4), top_k=3)
        assert len(results) == 3

    def test_empty_memory_returns_empty(self):
        mem = Memory()
        results = mem.find_similar(np.ones(4))
        assert len(results) == 0


class TestTiering:
    def test_eviction_on_capacity(self):
        mem = Memory(hot_capacity=5)
        fractals = []
        for i in range(7):
            f = Fractal(dim=4)
            f.metrics.accuracy_ema = i * 0.1  # Higher i = higher fitness
            f.metrics.prediction_error_ema = 1.0 - i * 0.1
            mem.store(f)
            fractals.append(f)

        # Should have evicted some to cold
        assert mem.stats()["hot_count"] <= 5
        assert mem.stats()["cold_count"] > 0

    def test_promote_from_cold(self):
        mem = Memory(hot_capacity=3)

        # Fill hot, triggering eviction
        fractals = []
        for i in range(5):
            f = Fractal(dim=4)
            f.metrics.accuracy_ema = i * 0.2
            f.metrics.prediction_error_ema = 1.0 - i * 0.1
            mem.store(f)
            fractals.append(f)

        # Find an ID in cold storage
        cold_ids = list(mem.cold.keys())
        if cold_ids:
            promoted = mem.get(cold_ids[0])
            assert promoted is not None
            assert isinstance(promoted, Fractal)
            assert cold_ids[0] in mem.hot
            assert cold_ids[0] not in mem.cold

    def test_does_not_evict_children_of_active_parents(self):
        mem = Memory(hot_capacity=5)

        parent = Fractal(dim=4)
        child = Fractal(dim=4)
        parent.add_child(child)

        # Give parent high fitness so IT doesn't get evicted
        parent.metrics.accuracy_ema = 0.9
        parent.metrics.prediction_error_ema = 0.1
        # Child has zero fitness but should be protected
        child.metrics.accuracy_ema = 0.0

        mem.store(parent)
        mem.store(child)

        # Fill with low-fitness fractals
        for _ in range(10):
            f = Fractal(dim=4)
            f.metrics.accuracy_ema = 0.0
            mem.store(f)

        # Child should still be in hot (protected by parent)
        assert child.id in mem.hot

    def test_compressed_data_is_smaller(self):
        """Compressed form should be a plain dict, not a live object."""
        mem = Memory(hot_capacity=2)

        f1 = Fractal(dim=4)
        f1.metrics.accuracy_ema = 0.9
        f1.metrics.prediction_error_ema = 0.1
        mem.store(f1)

        f2 = Fractal(dim=4)
        f2.metrics.accuracy_ema = 0.0
        f2.metrics.prediction_error_ema = 1.0
        mem.store(f2)

        f3 = Fractal(dim=4)
        f3.metrics.accuracy_ema = 0.95
        f3.metrics.prediction_error_ema = 0.05
        mem.store(f3)

        # f2 should be in cold (lowest fitness)
        if f2.id in mem.cold:
            assert isinstance(mem.cold[f2.id], dict)
            assert "prototype" in mem.cold[f2.id]
