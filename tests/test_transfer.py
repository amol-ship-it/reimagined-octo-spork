"""Tests for pattern library and transfer learning."""

import numpy as np
import pytest

from cognitive_fractal.function_fractal import (
    FunctionFractal,
    SinFractal,
    CosFractal,
    LinearFractal,
    ConstantFractal,
    ComposedFunctionFractal,
)
from cognitive_fractal.memory import Memory
from cognitive_fractal.symbolic_engine import SymbolicEngine
from cognitive_fractal.types import Signal


# ================================================================
# SIGNATURE TESTS
# ================================================================


class TestFunctionSignature:
    """Test the signature computation mechanism."""

    def test_signature_is_correct_length(self):
        f = SinFractal()
        f.coefficients = np.array([1.0, 1.0, 0.0, 0.0])
        sig = f.compute_signature()
        assert sig.shape == (FunctionFractal.SIGNATURE_POINTS,)

    def test_signature_is_normalized(self):
        f = SinFractal()
        f.coefficients = np.array([5.0, 1.0, 0.0, 0.0])
        sig = f.compute_signature()
        norm = np.linalg.norm(sig)
        assert abs(norm - 1.0) < 1e-6

    def test_similar_functions_have_similar_signatures(self):
        """Same frequency, different amplitude → nearly identical signatures."""
        f1 = SinFractal()
        f1.coefficients = np.array([1.0, 0.5, 0.0, 0.0])
        f2 = SinFractal()
        f2.coefficients = np.array([3.0, 0.5, 0.0, 0.0])
        sig1 = f1.compute_signature()
        sig2 = f2.compute_signature()
        sim = float(np.dot(sig1, sig2))
        assert sim > 0.95, f"Similar functions have low similarity: {sim}"

    def test_different_functions_have_different_signatures(self):
        sin_f = SinFractal()
        sin_f.coefficients = np.array([1.0, 1.0, 0.0, 0.0])
        lin_f = LinearFractal()
        lin_f.coefficients = np.array([1.0, 0.0])
        sig_sin = sin_f.compute_signature()
        sig_lin = lin_f.compute_signature()
        sim = float(np.dot(sig_sin, sig_lin))
        assert sim < 0.9, f"Different functions have high similarity: {sim}"

    def test_zero_coefficient_signature(self):
        f = ConstantFractal()
        f.coefficients = np.array([0.0])
        sig = f.compute_signature()
        assert np.allclose(sig, 0)

    def test_signature_updated_after_process(self):
        f = SinFractal()
        initial_sig = f._signature.copy()
        x = np.arange(50, dtype=float)
        y = 2.0 * np.sin(0.5 * x) + 1.0
        signal = Signal(data=y, timestamp=0.0)
        f._x_offset = 0
        f.predict_horizon = 1
        f.process(signal)
        assert not np.allclose(f._signature, initial_sig)

    def test_signature_in_compress(self):
        f = LinearFractal()
        f.coefficients = np.array([2.0, 3.0])
        f._signature = f.compute_signature()
        d = f.compress()
        assert "signature" in d
        assert len(d["signature"]) == FunctionFractal.SIGNATURE_POINTS


# ================================================================
# MEMORY SIGNATURE SEARCH TESTS
# ================================================================


class TestMemorySignatureSearch:
    """Test the signature-based similarity search in Memory."""

    def test_find_similar_by_signature_exact_match(self):
        mem = Memory()
        f1 = SinFractal()
        f1.coefficients = np.array([1.0, 0.5, 0.0, 0.0])
        f1._signature = f1.compute_signature()
        f1.metrics.accuracy_ema = 0.8
        f1.metrics.prediction_error_ema = 0.2
        mem.store(f1)

        results = mem.find_similar_by_signature(f1._signature, domain="symbolic")
        assert len(results) >= 1
        assert results[0][0] is f1
        assert results[0][1] > 0.99

    def test_signature_search_respects_min_fitness(self):
        mem = Memory()
        f1 = SinFractal()
        f1.coefficients = np.array([1.0, 0.5, 0.0, 0.0])
        f1._signature = f1.compute_signature()
        # Zero fitness (default)
        mem.store(f1)

        results = mem.find_similar_by_signature(f1._signature, min_fitness=0.1)
        assert len(results) == 0

    def test_signature_search_ranks_correctly(self):
        mem = Memory()

        f_sin = SinFractal()
        f_sin.coefficients = np.array([1.0, 0.5, 0.0, 0.0])
        f_sin._signature = f_sin.compute_signature()
        f_sin.metrics.accuracy_ema = 0.8
        f_sin.metrics.prediction_error_ema = 0.2
        mem.store(f_sin)

        f_lin = LinearFractal()
        f_lin.coefficients = np.array([1.0, 0.0])
        f_lin._signature = f_lin.compute_signature()
        f_lin.metrics.accuracy_ema = 0.8
        f_lin.metrics.prediction_error_ema = 0.2
        mem.store(f_lin)

        results = mem.find_similar_by_signature(f_sin._signature, top_k=2)
        assert results[0][0] is f_sin

    def test_update_signature(self):
        mem = Memory()
        f = SinFractal()
        mem.store(f)
        new_sig = np.random.randn(FunctionFractal.SIGNATURE_POINTS)
        mem.update_signature(f.id, new_sig)
        assert np.allclose(mem._signatures[f.id], new_sig)

    def test_stats_includes_signature_count(self):
        mem = Memory()
        f = SinFractal()
        f.coefficients = np.array([1.0, 1.0, 0.0, 0.0])
        f._signature = f.compute_signature()
        mem.store(f)
        stats = mem.stats()
        assert "signature_count" in stats
        assert stats["signature_count"] >= 1

    def test_domain_filtering(self):
        mem = Memory()
        f = SinFractal()
        f.coefficients = np.array([1.0, 0.5, 0.0, 0.0])
        f._signature = f.compute_signature()
        f.metrics.accuracy_ema = 0.8
        f.metrics.prediction_error_ema = 0.2
        mem.store(f)

        # Search with wrong domain should return nothing
        results = mem.find_similar_by_signature(
            f._signature, domain="wrong_domain"
        )
        assert len(results) == 0

        # Search with correct domain should find it
        results = mem.find_similar_by_signature(
            f._signature, domain="symbolic"
        )
        assert len(results) == 1


# ================================================================
# SEED FROM LIBRARY TESTS
# ================================================================


class TestSeedFromLibrary:
    """Test coefficient seeding (transfer learning)."""

    def test_seed_from_copies_coefficients(self):
        source = SinFractal()
        source.coefficients = np.array([3.0, 0.5, 1.0, 2.0])

        target = SinFractal()
        assert np.allclose(target.coefficients, [1.0, 1.0, 0.0, 0.0])

        target.seed_from(source)
        np.testing.assert_allclose(target.coefficients, source.coefficients)

    def test_seed_from_wrong_type_does_nothing(self):
        source = LinearFractal()
        source.coefficients = np.array([2.0, 3.0])

        target = SinFractal()
        original = target.coefficients.copy()
        target.seed_from(source)
        np.testing.assert_allclose(target.coefficients, original)

    def test_seed_does_not_alias(self):
        """Seeded coefficients should be a copy, not a reference."""
        source = SinFractal()
        source.coefficients = np.array([3.0, 0.5, 1.0, 2.0])

        target = SinFractal()
        target.seed_from(source)

        source.coefficients[0] = 999.0
        assert target.coefficients[0] != 999.0

    def test_seed_updates_signature(self):
        source = SinFractal()
        source.coefficients = np.array([3.0, 0.5, 1.0, 2.0])

        target = SinFractal()
        target.seed_from(source)

        # Signature should reflect the new coefficients
        expected_sig = source.compute_signature()
        np.testing.assert_allclose(target._signature, expected_sig, atol=1e-10)


# ================================================================
# SHARED MEMORY TRANSFER TESTS
# ================================================================


class TestSharedMemoryTransfer:
    """Test transfer learning via shared Memory."""

    def test_shared_memory_constructor(self):
        shared = Memory()
        e1 = SymbolicEngine(memory=shared)
        e2 = SymbolicEngine(memory=shared)
        assert e1.memory is e2.memory

    def test_default_memory_is_private(self):
        e1 = SymbolicEngine()
        e2 = SymbolicEngine()
        assert e1.memory is not e2.memory

    def test_patterns_from_stream1_visible_to_stream2(self):
        shared = Memory()

        # Stream 1 learns a linear pattern
        e1 = SymbolicEngine(window_size=15, memory=shared)
        for t in range(100):
            e1.step(2.0 * t + 3.0)

        hot_count_after_e1 = shared.stats()["hot_count"]

        # Stream 2 starts — its candidates also go into shared memory
        e2 = SymbolicEngine(window_size=15, memory=shared)
        e2.step(0.0)

        # Shared memory should have patterns from both engines
        assert shared.stats()["hot_count"] > hot_count_after_e1

    def test_engine_seeds_from_library(self):
        """Second engine should pick up learned coefficients."""
        shared = Memory()

        # Stream 1: learn linear y = 2x + 3
        e1 = SymbolicEngine(window_size=15, memory=shared)
        for t in range(100):
            e1.step(2.0 * t + 3.0)

        # Find the linear candidate in e1 and check it learned
        e1_linear = None
        for c in e1.candidates:
            if isinstance(c, LinearFractal):
                e1_linear = c
                break
        assert e1_linear is not None
        assert e1_linear.metrics.fitness() > 0.1

        # Stream 2: initialize — should seed linear from e1's learned coefficients
        e2 = SymbolicEngine(window_size=15, memory=shared)
        e2.step(0.0)

        e2_linear = None
        for c in e2.candidates:
            if isinstance(c, LinearFractal):
                e2_linear = c
                break
        assert e2_linear is not None
        # e2's linear should have been seeded (not zeros)
        assert not np.allclose(e2_linear.coefficients, [0.0, 0.0])

    def test_transfer_learning_on_similar_streams(self):
        """Stream 2 with transfer should converge at least as well as without."""
        shared = Memory()

        # Stream 1: learn sin(0.5x)
        e1 = SymbolicEngine(window_size=30, memory=shared)
        for t in range(300):
            e1.step(3.0 * np.sin(0.5 * t) + 1.0)

        _, fitness1, _ = e1.get_best()
        assert fitness1 > 0.05, f"Stream 1 fitness too low: {fitness1}"

        # Stream 2 with transfer: similar pattern
        e2_transfer = SymbolicEngine(window_size=30, memory=shared)
        transfer_errors = []
        prev_pred = None
        for t in range(100):
            y = 2.0 * np.sin(0.5 * t) + 5.0
            pred, diag = e2_transfer.step(y)
            if prev_pred is not None and t > 5:
                transfer_errors.append(abs(prev_pred - y))
            prev_pred = pred

        # Stream 2 without transfer
        e2_isolated = SymbolicEngine(window_size=30)
        isolated_errors = []
        prev_pred = None
        for t in range(100):
            y = 2.0 * np.sin(0.5 * t) + 5.0
            pred, diag = e2_isolated.step(y)
            if prev_pred is not None and t > 5:
                isolated_errors.append(abs(prev_pred - y))
            prev_pred = pred

        early_transfer = np.mean(transfer_errors[:20]) if transfer_errors else float("inf")
        early_isolated = np.mean(isolated_errors[:20]) if isolated_errors else float("inf")

        # Transfer should not make things significantly worse
        assert early_transfer < early_isolated * 1.5, (
            f"Transfer hurt: {early_transfer:.4f} vs {early_isolated:.4f}"
        )

    def test_signatures_populated_after_steps(self):
        engine = SymbolicEngine(window_size=10)
        for t in range(20):
            engine.step(float(t) * 2 + 1)
        assert engine.memory.stats()["signature_count"] > 0

    def test_pruning_preserves_high_fitness_in_memory(self):
        """High-fitness patterns should stay in memory after pruning."""
        engine = SymbolicEngine(
            window_size=10,
            max_candidates=8,
            composition_interval=10,
            composition_threshold=0.01,
        )
        for t in range(200):
            y = float(t) * 2 + np.sin(float(t))
            engine.step(y)

        # After pruning, memory should retain more patterns than candidates
        assert engine.memory.stats()["hot_count"] >= len(engine.candidates)
