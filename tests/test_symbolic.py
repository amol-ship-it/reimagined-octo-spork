"""Tests for the symbolic function discovery system."""

import numpy as np
import pytest
import copy

from cognitive_fractal.fractal import Fractal
from cognitive_fractal.function_fractal import (
    FunctionFractal,
    ConstantFractal,
    LinearFractal,
    QuadraticFractal,
    PolynomialFractal,
    SinFractal,
    CosFractal,
    ComposedFunctionFractal,
)
from cognitive_fractal.symbolic_engine import SymbolicEngine


# ================================================================
# CREATION TESTS
# ================================================================

class TestFunctionFractalCreation:

    def test_constant_has_one_coefficient(self):
        f = ConstantFractal()
        assert f.coefficients.shape == (1,)

    def test_linear_has_two_coefficients(self):
        f = LinearFractal()
        assert f.coefficients.shape == (2,)

    def test_quadratic_has_three_coefficients(self):
        f = QuadraticFractal()
        assert f.coefficients.shape == (3,)

    def test_polynomial_degree_4_has_five_coefficients(self):
        f = PolynomialFractal(degree=4)
        assert f.coefficients.shape == (5,)

    def test_sin_has_four_coefficients(self):
        f = SinFractal()
        assert f.coefficients.shape == (4,)

    def test_is_fractal_subclass(self):
        assert isinstance(ConstantFractal(), Fractal)
        assert isinstance(LinearFractal(), Fractal)
        assert isinstance(SinFractal(), Fractal)

    def test_initial_fitness_is_zero(self):
        f = LinearFractal()
        assert f.metrics.fitness() == 0.0

    def test_unique_ids(self):
        f1 = LinearFractal()
        f2 = LinearFractal()
        assert f1.id != f2.id


# ================================================================
# EVALUATION TESTS
# ================================================================

class TestFunctionEvaluation:

    def test_constant_evaluate(self):
        f = ConstantFractal()
        f.coefficients[0] = 5.0
        x = np.array([0.0, 1.0, 2.0, 10.0])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [5.0, 5.0, 5.0, 5.0])

    def test_linear_evaluate(self):
        f = LinearFractal()
        f.coefficients = np.array([2.0, 3.0])  # 2x + 3
        x = np.array([0.0, 1.0, 2.0, 5.0])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [3.0, 5.0, 7.0, 13.0])

    def test_quadratic_evaluate(self):
        f = QuadraticFractal()
        f.coefficients = np.array([1.0, 0.0, 0.0])  # x^2
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [0.0, 1.0, 4.0, 9.0])

    def test_sin_evaluate(self):
        f = SinFractal()
        f.coefficients = np.array([1.0, 1.0, 0.0, 0.0])  # sin(x)
        x = np.array([0.0, np.pi / 2, np.pi])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [0.0, 1.0, 0.0], atol=1e-10)

    def test_polynomial_evaluate(self):
        f = PolynomialFractal(degree=3)
        f.coefficients = np.array([1.0, 0.0, 0.0, 0.0])  # x^3
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = f.evaluate(x)
        np.testing.assert_allclose(y, [0.0, 1.0, 8.0, 27.0])


# ================================================================
# FITTING TESTS
# ================================================================

class TestFunctionFitting:

    def test_constant_fit(self):
        f = ConstantFractal()
        x = np.arange(10, dtype=float)
        y = np.full(10, 7.0)
        f.fit(x, y)
        assert abs(f.coefficients[0] - 7.0) < 1e-10

    def test_linear_fit_exact(self):
        f = LinearFractal()
        x = np.arange(20, dtype=float)
        y = 2.0 * x + 3.0
        f.fit(x, y)
        np.testing.assert_allclose(f.coefficients, [2.0, 3.0], atol=1e-8)

    def test_quadratic_fit_exact(self):
        f = QuadraticFractal()
        x = np.arange(20, dtype=float)
        y = 0.5 * x ** 2 - 3.0 * x + 10.0
        f.fit(x, y)
        np.testing.assert_allclose(f.coefficients, [0.5, -3.0, 10.0], atol=1e-6)

    def test_sin_fit_converges(self):
        """Sin fitting should converge to reasonable RMSE."""
        f = SinFractal(learning_rate=0.01)
        x = np.arange(50, dtype=float)
        y = 2.0 * np.sin(x)
        f.fit(x, y)
        y_pred = f.evaluate(x)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        # Should achieve < 50% of the signal range
        assert rmse < 2.0, f"Sin RMSE too high: {rmse}"

    def test_cos_fit_converges(self):
        f = CosFractal(learning_rate=0.01)
        x = np.arange(50, dtype=float)
        y = 3.0 * np.cos(0.5 * x) + 1.0
        f.fit(x, y)
        y_pred = f.evaluate(x)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rmse < 3.0, f"Cos RMSE too high: {rmse}"


# ================================================================
# COMPOSITION TESTS
# ================================================================

class TestComposedFunctionFractal:

    def test_additive_composition_evaluate(self):
        lin = LinearFractal()
        lin.coefficients = np.array([1.0, 0.0])  # x
        const = ConstantFractal()
        const.coefficients = np.array([5.0])  # 5

        composed = ComposedFunctionFractal(lin, const, operation="add")
        x = np.array([0.0, 1.0, 2.0])
        y = composed.evaluate(x)
        np.testing.assert_allclose(y, [5.0, 6.0, 7.0])

    def test_multiplicative_composition_evaluate(self):
        lin = LinearFractal()
        lin.coefficients = np.array([2.0, 0.0])  # 2x
        const = ConstantFractal()
        const.coefficients = np.array([3.0])  # 3

        composed = ComposedFunctionFractal(lin, const, operation="multiply")
        x = np.array([0.0, 1.0, 2.0])
        y = composed.evaluate(x)
        np.testing.assert_allclose(y, [0.0, 6.0, 12.0])

    def test_symbolic_repr_contains_children(self):
        lin = LinearFractal()
        lin.coefficients = np.array([2.0, 3.0])
        sin = SinFractal()
        composed = ComposedFunctionFractal(lin, sin, operation="add")
        repr_str = composed.symbolic_repr()
        assert "+" in repr_str
        assert "sin" in repr_str

    def test_composed_is_fractal_subclass(self):
        lin = LinearFractal()
        const = ConstantFractal()
        composed = ComposedFunctionFractal(lin, const, "add")
        assert isinstance(composed, Fractal)
        assert composed.is_composed


# ================================================================
# ENGINE TESTS
# ================================================================

class TestSymbolicEngine:

    def test_creates_candidates_on_first_step(self):
        engine = SymbolicEngine()
        engine.step(1.0)
        assert len(engine.candidates) >= 7

    def test_step_returns_prediction_and_diagnostics(self):
        engine = SymbolicEngine(window_size=5)
        for i in range(10):
            pred, diag = engine.step(float(i))
        assert isinstance(pred, float)
        assert "step" in diag
        assert "best_name" in diag
        assert "best_fitness" in diag
        assert "num_candidates" in diag

    def test_learns_linear_function(self):
        engine = SymbolicEngine(window_size=15)
        for t in range(100):
            y = 2.0 * t + 3.0
            engine.step(y)
        formula, fitness, best = engine.get_best()
        assert fitness > 0.3, f"Linear fitness too low: {fitness}"
        assert "linear" in best.func_name or "poly" in best.func_name

    def test_learns_quadratic_function(self):
        engine = SymbolicEngine(window_size=20)
        for t in range(150):
            y = 0.5 * t ** 2 - 3.0 * t + 10.0
            engine.step(y)
        formula, fitness, best = engine.get_best()
        assert fitness > 0.2, f"Quadratic fitness too low: {fitness}"

    def test_learns_sin_function(self):
        engine = SymbolicEngine(window_size=30)
        for t in range(300):
            y = 3.0 * np.sin(0.5 * t) + 1.0
            engine.step(y)
        formula, fitness, best = engine.get_best()
        assert fitness > 0.05, f"Sin fitness too low: {fitness}"

    def test_predictions_improve_over_time(self):
        engine = SymbolicEngine(window_size=10)
        early_errors = []
        late_errors = []
        prev_pred = None
        for t in range(100):
            y = 2.0 * t + 3.0
            pred, diag = engine.step(y)
            # Compare PREVIOUS prediction to CURRENT actual
            if prev_pred is not None:
                err = abs(prev_pred - y)
                if 10 <= t < 30:
                    early_errors.append(err)
                elif 80 <= t < 100:
                    late_errors.append(err)
            prev_pred = pred
        avg_early = np.mean(early_errors) if early_errors else float("inf")
        avg_late = np.mean(late_errors) if late_errors else float("inf")
        assert avg_late <= avg_early + 1e-6, f"Late error {avg_late} > early {avg_early}"

    def test_get_predictions_shape(self):
        engine = SymbolicEngine(window_size=10)
        for t in range(20):
            engine.step(float(t))
        preds = engine.get_predictions(5)
        assert preds.shape == (5,)

    def test_engine_uses_memory(self):
        engine = SymbolicEngine()
        engine.step(1.0)
        assert engine.memory.stats()["hot_count"] > 0

    def test_pruning_bounds_candidates(self):
        engine = SymbolicEngine(
            window_size=10,
            max_candidates=10,
            composition_interval=10,
            composition_threshold=0.01,
        )
        for t in range(200):
            y = float(t) * 2 + np.sin(float(t))
            engine.step(y)
        assert len(engine.candidates) <= engine.max_candidates + 5


# ================================================================
# METRICS INTEGRATION TESTS
# ================================================================

class TestMetricsIntegration:

    def test_fitness_increases_with_exposure(self):
        engine = SymbolicEngine(window_size=10)
        fitnesses = []
        for t in range(60):
            y = 2.0 * t + 3.0
            engine.step(y)
            if t % 20 == 19:
                _, fitness, _ = engine.get_best()
                fitnesses.append(fitness)
        # Fitness should increase over time
        assert fitnesses[-1] >= fitnesses[0], (
            f"Fitness did not increase: {fitnesses}"
        )

    def test_function_fractal_serialization(self):
        f = LinearFractal()
        f.coefficients = np.array([2.0, 3.0])
        d = f.compress()
        assert "coefficients" in d
        assert "func_name" in d
        assert d["func_name"] == "linear"
        np.testing.assert_allclose(d["coefficients"], [2.0, 3.0])
