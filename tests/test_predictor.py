"""Tests for SequencePredictor MVP, nested composition, and subtract/divide."""

import numpy as np
import pytest

from cognitive_fractal.predictor import SequencePredictor
from cognitive_fractal.nested_fractal import NestedComposedFractal
from cognitive_fractal.function_fractal import (
    FunctionFractal,
    LinearFractal,
    QuadraticFractal,
    SinFractal,
    CosFractal,
    ConstantFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)
from cognitive_fractal.symbolic_engine import SymbolicEngine
from cognitive_fractal.memory import Memory


# ================================================================
# NESTED COMPOSED FRACTAL
# ================================================================


class TestNestedComposedFractalCreation:

    def test_creates_with_inner_outer(self):
        inner = LinearFractal()
        outer = SinFractal()
        nested = NestedComposedFractal(inner, outer)
        assert nested.inner is inner
        assert nested.outer is outer

    def test_is_fractal_subclass(self):
        inner = LinearFractal()
        outer = QuadraticFractal()
        nested = NestedComposedFractal(inner, outer)
        assert isinstance(nested, FunctionFractal)
        assert nested.is_composed

    def test_func_name_reflects_nesting(self):
        inner = LinearFractal()
        outer = SinFractal()
        nested = NestedComposedFractal(inner, outer)
        assert "sin" in nested.func_name
        assert "linear" in nested.func_name

    def test_children_registered(self):
        inner = LinearFractal()
        outer = SinFractal()
        nested = NestedComposedFractal(inner, outer)
        assert len(nested.children) == 2


class TestNestedComposedFractalEvaluation:

    def test_sin_of_linear(self):
        """sin(2x + 1) — outer=sin, inner=linear."""
        inner = LinearFractal()
        inner.coefficients = np.array([2.0, 1.0])
        outer = SinFractal()
        outer.coefficients = np.array([1.0, 1.0, 0.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        x = np.array([0.0, 1.0, 2.0])
        y = nested.evaluate(x)
        expected = np.sin(2.0 * x + 1.0)
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_quadratic_of_linear(self):
        """(3x)^2 = 9x^2 — outer=quadratic(z^2), inner=linear(3x)."""
        inner = LinearFractal()
        inner.coefficients = np.array([3.0, 0.0])
        outer = QuadraticFractal()
        outer.coefficients = np.array([1.0, 0.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = nested.evaluate(x)
        expected = (3.0 * x) ** 2
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_exp_of_linear(self):
        """exp(0.1x) — outer=exp(z), inner=linear(0.1x)."""
        inner = LinearFractal()
        inner.coefficients = np.array([0.1, 0.0])
        outer = ExponentialFractal()
        outer.coefficients = np.array([1.0, 1.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        x = np.array([0.0, 1.0, 5.0, 10.0])
        z = 0.1 * x
        expected = np.exp(np.clip(z, -50, 50))
        y = nested.evaluate(x)
        np.testing.assert_allclose(y, expected, atol=1e-8)

    def test_handles_inf_from_inner(self):
        """Inner producing extreme values should not crash outer."""
        inner = ExponentialFractal()
        inner.coefficients = np.array([1.0, 2.0, 0.0])  # exp(2x) grows fast
        outer = LinearFractal()
        outer.coefficients = np.array([1.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        x = np.array([0.0, 100.0, 1000.0])
        y = nested.evaluate(x)
        assert np.all(np.isfinite(y))


class TestNestedComposedFractalFit:

    def test_fit_refits_outer_only(self):
        """Fitting should modify outer coefficients but not inner."""
        inner = LinearFractal()
        inner.coefficients = np.array([2.0, 0.0])
        inner_original = inner.coefficients.copy()

        outer = QuadraticFractal()
        outer.coefficients = np.array([0.0, 0.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        x = np.arange(20, dtype=float)
        y = (2.0 * x) ** 2 + 3.0

        nested.fit(x, y)

        np.testing.assert_allclose(nested.inner.coefficients, inner_original)
        assert not np.allclose(nested.outer.coefficients, [0.0, 0.0, 0.0])

    def test_fit_skips_degenerate_inner(self):
        """If inner produces constant output, fit should not crash."""
        inner = ConstantFractal()
        inner.coefficients = np.array([5.0])
        outer = LinearFractal()

        nested = NestedComposedFractal(inner, outer)
        x = np.arange(20, dtype=float)
        y = np.arange(20, dtype=float)
        nested.fit(x, y)  # Should not raise

    def test_symbolic_repr(self):
        inner = LinearFractal()
        inner.coefficients = np.array([2.0, 1.0])
        outer = SinFractal()
        outer.coefficients = np.array([1.0, 1.0, 0.0, 0.0])

        nested = NestedComposedFractal(inner, outer)
        s = nested.symbolic_repr()
        assert "Nested" in s or "sin" in s

    def test_compress_includes_composition_type(self):
        inner = LinearFractal()
        outer = SinFractal()
        nested = NestedComposedFractal(inner, outer)
        d = nested.compress()
        assert d["composition_type"] == "nested"
        assert "inner_data" in d
        assert "outer_data" in d


# ================================================================
# SUBTRACT / DIVIDE COMPOSITION
# ================================================================


class TestSubtractDivideComposition:

    def test_subtract_evaluate(self):
        lin = LinearFractal()
        lin.coefficients = np.array([3.0, 0.0])
        const = ConstantFractal()
        const.coefficients = np.array([1.0])

        composed = ComposedFunctionFractal(lin, const, operation="subtract")
        x = np.array([0.0, 1.0, 2.0])
        y = composed.evaluate(x)
        np.testing.assert_allclose(y, [-1.0, 2.0, 5.0])

    def test_subtract_fit(self):
        child1 = LinearFractal()
        child2 = ConstantFractal()
        composed = ComposedFunctionFractal(child1, child2, operation="subtract")

        x = np.arange(20, dtype=float)
        y = 2.0 * x - 5.0
        composed.fit(x, y)
        y_pred = composed.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 1.0, f"Subtract RMSE too high: {rmse}"

    def test_subtract_symbolic_repr(self):
        lin = LinearFractal()
        const = ConstantFractal()
        composed = ComposedFunctionFractal(lin, const, operation="subtract")
        assert " - " in composed.symbolic_repr()

    def test_divide_evaluate(self):
        lin = LinearFractal()
        lin.coefficients = np.array([6.0, 0.0])
        const = ConstantFractal()
        const.coefficients = np.array([2.0])

        composed = ComposedFunctionFractal(lin, const, operation="divide")
        x = np.array([1.0, 2.0, 3.0])
        y = composed.evaluate(x)
        np.testing.assert_allclose(y, [3.0, 6.0, 9.0])

    def test_divide_safe_near_zero(self):
        lin = LinearFractal()
        lin.coefficients = np.array([1.0, 0.0])
        zero_const = ConstantFractal()
        zero_const.coefficients = np.array([0.0])

        composed = ComposedFunctionFractal(lin, zero_const, operation="divide")
        x = np.array([1.0, 2.0, 3.0])
        y = composed.evaluate(x)
        assert np.all(np.isfinite(y))

    def test_divide_symbolic_repr(self):
        lin = LinearFractal()
        const = ConstantFractal()
        composed = ComposedFunctionFractal(lin, const, operation="divide")
        assert " / " in composed.symbolic_repr()


# ================================================================
# SEQUENCE PREDICTOR WRAPPER
# ================================================================


class TestSequencePredictorCreation:

    def test_default_creation(self):
        sp = SequencePredictor()
        assert sp._step_count == 0

    def test_custom_parameters(self):
        sp = SequencePredictor(
            predict_ahead=10,
            window_size=30,
            auto_compose=False,
        )
        assert sp._predict_ahead == 10


class TestSequencePredictorFeed:

    def test_feed_returns_dict(self):
        sp = SequencePredictor(window_size=10)
        result = sp.feed(1.0)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "formula" in result
        assert "fitness" in result
        assert "step" in result
        assert "error" in result
        assert "num_candidates" in result

    def test_feed_increments_step(self):
        sp = SequencePredictor(window_size=10)
        sp.feed(1.0)
        sp.feed(2.0)
        assert sp._step_count == 2

    def test_first_feed_has_zero_error(self):
        sp = SequencePredictor(window_size=10)
        result = sp.feed(5.0)
        assert result["error"] == 0.0

    def test_feed_tracks_error(self):
        sp = SequencePredictor(window_size=10)
        sp.feed(1.0)
        result = sp.feed(2.0)
        assert isinstance(result["error"], float)


class TestSequencePredictorPredict:

    def test_predict_returns_array(self):
        sp = SequencePredictor(window_size=10)
        for i in range(20):
            sp.feed(float(i))
        preds = sp.predict(5)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5,)

    def test_predict_linear_sequence(self):
        sp = SequencePredictor(window_size=15, predict_ahead=1)
        for t in range(100):
            sp.feed(2.0 * t + 3.0)
        preds = sp.predict(5)
        actuals = np.array([2.0 * t + 3.0 for t in range(100, 105)])
        errors = np.abs(preds - actuals)
        assert np.mean(errors) < 1.0, f"Linear prediction MAE too high: {np.mean(errors)}"


class TestSequencePredictorAccuracy:

    def test_accuracy_returns_dict(self):
        sp = SequencePredictor(window_size=10)
        for i in range(20):
            sp.feed(float(i))
        acc = sp.accuracy()
        assert "best_formula" in acc
        assert "best_fitness" in acc
        assert "mean_absolute_error" in acc
        assert "recent_mae" in acc
        assert "num_candidates" in acc
        assert "steps" in acc

    def test_accuracy_improves_over_time(self):
        sp = SequencePredictor(window_size=15)
        for t in range(100):
            sp.feed(2.0 * t + 3.0)
        acc = sp.accuracy()
        assert acc["best_fitness"] > 0.1


class TestSequencePredictorBatch:

    def test_feed_sequence(self):
        sp = SequencePredictor(window_size=10)
        values = [float(i) for i in range(30)]
        result = sp.feed_sequence(values)
        assert result["step"] == 30
        assert sp._step_count == 30


# ================================================================
# NESTED COMPOSITION DISCOVERY
# ================================================================


class TestNestedCompositionDiscovery:

    def test_engine_has_nested_composition_method(self):
        engine = SymbolicEngine()
        assert hasattr(engine, "_attempt_nested_composition")

    def test_nested_composition_does_not_crash(self):
        engine = SymbolicEngine(
            window_size=15,
            nested_composition_interval=20,
        )
        for t in range(50):
            y = np.sin(float(t) ** 2)
            engine.step(y)

    def test_composition_depth_handles_nested(self):
        inner = LinearFractal()
        outer = SinFractal()
        nested = NestedComposedFractal(inner, outer)
        assert SymbolicEngine._composition_depth(nested) == 1

    def test_nested_depth_two(self):
        inner1 = LinearFractal()
        outer1 = QuadraticFractal()
        nested1 = NestedComposedFractal(inner1, outer1)
        outer2 = SinFractal()
        nested2 = NestedComposedFractal(nested1, outer2)
        assert SymbolicEngine._composition_depth(nested2) == 2


# ================================================================
# QUADRATIC STREAM
# ================================================================


class TestQuadraticStream:

    def test_discovers_quadratic(self):
        sp = SequencePredictor(window_size=20, predict_ahead=1)
        for t in range(150):
            y = 0.5 * t ** 2 - 3.0 * t + 10.0
            sp.feed(y)
        acc = sp.accuracy()
        assert acc["best_fitness"] > 0.2, f"Quadratic fitness too low: {acc['best_fitness']}"

    def test_quadratic_future_predictions(self):
        sp = SequencePredictor(window_size=20, predict_ahead=1)
        for t in range(150):
            y = 0.5 * t ** 2 - 3.0 * t + 10.0
            sp.feed(y)
        preds = sp.predict(5)
        actuals = np.array([0.5 * t ** 2 - 3.0 * t + 10.0 for t in range(150, 155)])
        errors = np.abs(preds - actuals)
        # Relative error: predictions on scale of ~10000+
        rel_errors = errors / np.abs(actuals)
        assert np.mean(rel_errors) < 0.01, f"Quadratic relative error too high: {np.mean(rel_errors)}"


# ================================================================
# MANDELBROT STREAM
# ================================================================


def mandelbrot_sequence(c: complex, n_terms: int) -> list:
    """Generate |z_n| for z_{n+1} = z_n^2 + c, z_0 = 0."""
    z = complex(0, 0)
    magnitudes = [abs(z)]
    for _ in range(n_terms - 1):
        z = z * z + c
        magnitudes.append(abs(z))
        if abs(z) > 1e10:
            break
    return magnitudes


class TestMandelbrotStream:

    def test_mandelbrot_sequence_generation(self):
        seq = mandelbrot_sequence(c=-0.75, n_terms=50)
        assert len(seq) == 50
        assert seq[0] == 0.0
        assert all(isinstance(v, float) for v in seq)

    def test_bounded_mandelbrot_runs(self):
        """c = -0.75 is bounded; predictor should handle it."""
        seq = mandelbrot_sequence(c=-0.75, n_terms=100)
        sp = SequencePredictor(window_size=30, predict_ahead=5)
        for v in seq:
            sp.feed(v)
        acc = sp.accuracy()
        assert acc["steps"] == 100
        assert acc["best_fitness"] >= 0.0

    def test_divergent_mandelbrot_handles_growth(self):
        """c = 0.3 + 0.5j diverges; predictor should not crash."""
        seq = mandelbrot_sequence(c=complex(0.3, 0.5), n_terms=30)
        sp = SequencePredictor(window_size=15, predict_ahead=1)
        for v in seq:
            sp.feed(v)
        assert sp._step_count == len(seq)

    def test_mandelbrot_predictions_are_finite(self):
        """Predictions on bounded Mandelbrot should be finite."""
        seq = mandelbrot_sequence(c=-0.75, n_terms=100)
        sp = SequencePredictor(window_size=30, predict_ahead=5)
        for v in seq:
            sp.feed(v)
        preds = sp.predict(5)
        assert np.all(np.isfinite(preds))
