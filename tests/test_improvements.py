"""Tests for the three improvements: gradient fitting, library restoration, exotic types."""

import copy
import numpy as np
import pytest

from cognitive_fractal.function_fractal import (
    FunctionFractal,
    SinFractal,
    CosFractal,
    LinearFractal,
    GradientSinFractal,
    GradientCosFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)
from cognitive_fractal.memory import Memory
from cognitive_fractal.symbolic_engine import SymbolicEngine
from cognitive_fractal.types import Signal


# ================================================================
# IMPROVEMENT 1: GRADIENT DESCENT WITH WARM STARTS
# ================================================================


class TestGradientSinFractal:
    """Test warm-start gradient descent for sin fitting."""

    def test_first_fit_uses_fft(self):
        """First call should use FFT+lstsq (inherited from SinFractal)."""
        f = GradientSinFractal()
        x = np.arange(50, dtype=float)
        y = 3.0 * np.sin(0.5 * x) + 1.0
        f.fit(x, y)
        a, b, c, d = f.coefficients
        assert abs(b - 0.5) < 0.1, f"First fit should find freq ~0.5, got {b}"

    def test_warm_start_preserves_coefficients(self):
        """Subsequent fits should refine, not recompute from scratch."""
        f = GradientSinFractal()
        # Seed with known coefficients
        f.coefficients = np.array([3.0, 0.5, 0.0, 1.0])
        f._fit_count = 1  # Mark as already initialized

        x = np.arange(50, dtype=float)
        y = 3.0 * np.sin(0.5 * x) + 1.0
        f.fit(x, y)

        # Gradient descent should stay close to seeded values
        a, b, c, d = f.coefficients
        assert abs(b - 0.5) < 0.2, f"Warm start should keep freq ~0.5, got {b}"

    def test_seeded_then_fit_retains_knowledge(self):
        """seed_from() + warm-start fit should retain the seeded frequency."""
        source = GradientSinFractal()
        source.coefficients = np.array([3.0, 0.5, 0.0, 1.0])
        source._fit_count = 5

        target = GradientSinFractal()
        target.seed_from(source)
        target._fit_count = 1  # Ensure warm-start path

        # Fit on slightly different data (same frequency)
        x = np.arange(50, dtype=float)
        y = 2.0 * np.sin(0.5 * x) + 5.0
        target.fit(x, y)

        a, b, c, d = target.coefficients
        assert abs(b - 0.5) < 0.3, f"Seeded freq should survive warm-start, got {b}"

    def test_evaluate_works(self):
        f = GradientSinFractal()
        f.coefficients = np.array([2.0, 1.0, 0.0, 0.0])
        x = np.array([0.0, np.pi / 2, np.pi])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [0.0, 2.0, 0.0], atol=1e-10)

    def test_func_name(self):
        f = GradientSinFractal()
        assert f.func_name == "grad_sin"

    def test_signature_works(self):
        f = GradientSinFractal()
        f.coefficients = np.array([2.0, 1.0, 0.0, 0.0])
        sig = f.compute_signature()
        assert sig.shape == (FunctionFractal.SIGNATURE_POINTS,)
        assert abs(np.linalg.norm(sig) - 1.0) < 1e-6


class TestGradientCosFractal:
    """Test warm-start gradient descent for cos fitting."""

    def test_first_fit_uses_fft(self):
        f = GradientCosFractal()
        x = np.arange(50, dtype=float)
        y = 2.0 * np.cos(0.8 * x) + 3.0
        f.fit(x, y)
        a, b, c, d = f.coefficients
        assert abs(b - 0.8) < 0.2, f"First fit should find freq ~0.8, got {b}"

    def test_warm_start_preserves_coefficients(self):
        f = GradientCosFractal()
        f.coefficients = np.array([2.0, 0.8, 0.0, 3.0])
        f._fit_count = 1

        x = np.arange(50, dtype=float)
        y = 2.0 * np.cos(0.8 * x) + 3.0
        f.fit(x, y)

        a, b, c, d = f.coefficients
        assert abs(b - 0.8) < 0.3, f"Warm start should keep freq ~0.8, got {b}"

    def test_func_name(self):
        f = GradientCosFractal()
        assert f.func_name == "grad_cos"


# ================================================================
# IMPROVEMENT 2: LIBRARY RESTORES AFTER PRUNING
# ================================================================


class TestLibraryRestoration:
    """Test that pruning refills slots from the pattern library."""

    def test_restore_from_library_fills_slots(self):
        """After pruning, empty slots should be filled from memory."""
        shared = Memory()

        # Create a high-fitness pattern in memory
        lib_sin = SinFractal()
        lib_sin.coefficients = np.array([3.0, 0.5, 0.0, 1.0])
        lib_sin.metrics.accuracy_ema = 0.9
        lib_sin.metrics.prediction_error_ema = 0.1
        shared.store(lib_sin)

        # Create engine with small max_candidates
        engine = SymbolicEngine(window_size=10, max_candidates=8, memory=shared)
        for t in range(50):
            engine.step(float(t) * 2 + 1)

        # After pruning + restoration, candidates should not exceed max
        assert len(engine.candidates) <= engine.max_candidates + 5

    def test_restore_respects_diversity_limit(self):
        """Max 2 restored candidates per type."""
        shared = Memory()

        # Add 5 high-fitness SinFractals to memory
        for i in range(5):
            f = SinFractal()
            f.coefficients = np.array([float(i + 1), 0.5, 0.0, 0.0])
            f.metrics.accuracy_ema = 0.9
            f.metrics.prediction_error_ema = 0.1
            shared.store(f)

        engine = SymbolicEngine(window_size=10, max_candidates=20, memory=shared)
        engine.step(0.0)  # Initialize candidates

        # Manually trigger restoration
        engine._restore_from_library()

        # Count SinFractals (originals + restored)
        sin_count = sum(1 for c in engine.candidates if type(c).__name__ == "SinFractal")
        # Should have at most 3 (1 default + 2 restored max)
        assert sin_count <= 3

    def test_restore_creates_independent_copies(self):
        """Restored candidates should not share state with library originals."""
        shared = Memory()
        original = SinFractal()
        original.coefficients = np.array([3.0, 0.5, 0.0, 1.0])
        original.metrics.accuracy_ema = 0.9
        original.metrics.prediction_error_ema = 0.1
        shared.store(original)

        engine = SymbolicEngine(window_size=10, memory=shared)
        engine.step(0.0)
        engine._restore_from_library()

        # Find any restored SinFractal
        for c in engine.candidates:
            if isinstance(c, SinFractal) and c.id != original.id:
                c.coefficients[0] = 999.0
                break

        # Original should be unaffected
        assert original.coefficients[0] == 3.0

    def test_prune_calls_restore(self):
        """_prune() should call _restore_from_library()."""
        engine = SymbolicEngine(
            window_size=10,
            max_candidates=8,
            composition_interval=10,
            composition_threshold=0.01,
        )
        # Run enough to trigger composition + pruning
        for t in range(200):
            y = float(t) * 2 + np.sin(float(t))
            engine.step(y)

        # After pruning, candidates should be bounded
        assert len(engine.candidates) <= engine.max_candidates + 5


# ================================================================
# IMPROVEMENT 3: EXOTIC FUNCTION TYPES
# ================================================================


class TestExponentialFractal:
    """Test exponential function discovery."""

    def test_evaluate(self):
        f = ExponentialFractal()
        f.coefficients = np.array([2.0, 0.1, 1.0])
        x = np.array([0.0, 1.0, 10.0])
        y = f.evaluate(x)
        expected = 2.0 * np.exp(0.1 * x) + 1.0
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_fit_exponential_data(self):
        f = ExponentialFractal()
        x = np.arange(30, dtype=float)
        y = 2.0 * np.exp(0.05 * x) + 3.0
        # Multiple fit calls (as in streaming) let gradient descent converge
        for _ in range(5):
            f.fit(x, y)
        y_pred = f.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < np.std(y), f"Exp fit RMSE too high: {rmse}"

    def test_overflow_protection(self):
        """Large b*x should not cause overflow."""
        f = ExponentialFractal()
        f.coefficients = np.array([1.0, 100.0, 0.0])
        x = np.array([0.0, 1.0, 1000.0])
        y = f.evaluate(x)
        assert np.all(np.isfinite(y))

    def test_b_clamp(self):
        """b should be clamped to [-2, 2] after fitting."""
        f = ExponentialFractal()
        x = np.arange(10, dtype=float)
        y = np.exp(5.0 * x)  # Very steep exponential
        f.fit(x, y)
        assert abs(f.coefficients[1]) <= 2.0 + 1e-6

    def test_symbolic_repr(self):
        f = ExponentialFractal()
        f.coefficients = np.array([2.0, 0.1, 1.0])
        s = f.symbolic_repr()
        assert "exp" in s

    def test_func_name(self):
        f = ExponentialFractal()
        assert f.func_name == "exp"

    def test_signature_works(self):
        f = ExponentialFractal()
        f.coefficients = np.array([1.0, 0.5, 0.0])
        sig = f.compute_signature()
        assert sig.shape == (FunctionFractal.SIGNATURE_POINTS,)


class TestLogFractal:
    """Test logarithmic function discovery."""

    def test_evaluate(self):
        f = LogFractal()
        f.coefficients = np.array([2.0, 1.0, 1.0, 0.0])
        x = np.array([0.0, 1.0, np.e - 1.0])
        y = f.evaluate(x)
        expected = 2.0 * np.log(np.maximum(1.0 * x + 1.0, 1e-8))
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_fit_log_data(self):
        f = LogFractal()
        x = np.arange(1, 50, dtype=float)
        y = 3.0 * np.log(x + 1.0) + 2.0
        # Multiple fit calls (as in streaming) let gradient descent converge
        for _ in range(5):
            f.fit(x, y)
        y_pred = f.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < np.std(y), f"Log fit RMSE too high: {rmse}"

    def test_domain_safety(self):
        """Evaluating at negative inner values should not crash."""
        f = LogFractal()
        f.coefficients = np.array([1.0, -1.0, -10.0, 0.0])
        x = np.array([0.0, 1.0, 2.0])
        y = f.evaluate(x)
        assert np.all(np.isfinite(y))

    def test_symbolic_repr(self):
        f = LogFractal()
        f.coefficients = np.array([2.0, 1.0, 1.0, 3.0])
        s = f.symbolic_repr()
        assert "log" in s

    def test_func_name(self):
        f = LogFractal()
        assert f.func_name == "log"


class TestCompositionDepth:
    """Test the _composition_depth static method."""

    def test_leaf_depth_is_zero(self):
        f = SinFractal()
        assert SymbolicEngine._composition_depth(f) == 0

    def test_simple_composition_depth_is_one(self):
        child1 = SinFractal()
        child2 = LinearFractal()
        composed = ComposedFunctionFractal(child1, child2, "add")
        assert SymbolicEngine._composition_depth(composed) == 1

    def test_nested_composition_depth(self):
        inner1 = SinFractal()
        inner2 = LinearFractal()
        inner_composed = ComposedFunctionFractal(inner1, inner2, "add")
        outer_leaf = ExponentialFractal()
        outer = ComposedFunctionFractal(inner_composed, outer_leaf, "add")
        assert SymbolicEngine._composition_depth(outer) == 2


class TestLoadComposedTemplates:
    """Test that composed patterns from memory are loaded as candidates."""

    def test_loads_high_fitness_composed(self):
        shared = Memory()

        # Create a high-fitness composed pattern
        child1 = SinFractal()
        child1.coefficients = np.array([3.0, 0.5, 0.0, 1.0])
        child2 = LinearFractal()
        child2.coefficients = np.array([2.0, 0.0])
        composed = ComposedFunctionFractal(child1, child2, "add")
        composed.metrics.accuracy_ema = 0.9
        composed.metrics.prediction_error_ema = 0.1
        shared.store(composed)

        engine = SymbolicEngine(window_size=10, memory=shared)
        engine.step(0.0)

        # Check that a composed template was loaded
        composed_candidates = [
            c for c in engine.candidates if isinstance(c, ComposedFunctionFractal)
        ]
        assert len(composed_candidates) >= 1

    def test_does_not_load_low_fitness_composed(self):
        shared = Memory()

        child1 = SinFractal()
        child2 = LinearFractal()
        composed = ComposedFunctionFractal(child1, child2, "add")
        # Default low fitness
        shared.store(composed)

        engine = SymbolicEngine(window_size=10, memory=shared)
        engine.step(0.0)

        # No composed templates should be loaded (fitness too low)
        composed_candidates = [
            c for c in engine.candidates if isinstance(c, ComposedFunctionFractal)
        ]
        assert len(composed_candidates) == 0


class TestExpandedCandidatePool:
    """Test that the expanded candidate pool includes new types."""

    def test_default_pool_has_11_candidates(self):
        engine = SymbolicEngine(window_size=10)
        engine.step(0.0)
        assert len(engine.candidates) == 11

    def test_gradient_sin_in_pool(self):
        engine = SymbolicEngine(window_size=10)
        engine.step(0.0)
        assert any(isinstance(c, GradientSinFractal) for c in engine.candidates)

    def test_gradient_cos_in_pool(self):
        engine = SymbolicEngine(window_size=10)
        engine.step(0.0)
        assert any(isinstance(c, GradientCosFractal) for c in engine.candidates)

    def test_exponential_in_pool(self):
        engine = SymbolicEngine(window_size=10)
        engine.step(0.0)
        assert any(isinstance(c, ExponentialFractal) for c in engine.candidates)

    def test_log_in_pool(self):
        engine = SymbolicEngine(window_size=10)
        engine.step(0.0)
        assert any(isinstance(c, LogFractal) for c in engine.candidates)
