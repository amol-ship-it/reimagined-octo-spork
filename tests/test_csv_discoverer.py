"""Tests for CSVFunctionDiscoverer, PatternStore, new log bases, tan, chain compositions, and mixed inner."""

import os
import tempfile
import numpy as np
import pytest

from cognitive_fractal.csv_discoverer import CSVFunctionDiscoverer, DiscoveryResult
from cognitive_fractal.pattern_store import PatternStore
from cognitive_fractal.base_functions import BASE_FUNCTIONS
from cognitive_fractal.inverted_composition import InvertedCompositionFractal
from cognitive_fractal.mixed_inner import MixedInnerFractal


# ================================================================
# CSV parsing
# ================================================================


class TestCSVParsing:
    """Test that various CSV formats are handled correctly."""

    def test_two_column_no_header(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("1,10\n2,20\n3,30\n4,40\n5,50\n")
        d = CSVFunctionDiscoverer(str(csv_file), verbose=False, db=None)
        np.testing.assert_array_equal(d._x, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(d._y, [10, 20, 30, 40, 50])

    def test_two_column_with_header(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n0,5\n1,10\n2,15\n")
        d = CSVFunctionDiscoverer(str(csv_file), verbose=False, db=None)
        np.testing.assert_array_equal(d._x, [0, 1, 2])
        np.testing.assert_array_equal(d._y, [5, 10, 15])

    def test_single_column(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("100\n200\n300\n")
        d = CSVFunctionDiscoverer(str(csv_file), verbose=False, db=None)
        np.testing.assert_array_equal(d._x, [0, 1, 2])
        np.testing.assert_array_equal(d._y, [100, 200, 300])

    def test_from_arrays(self):
        x = np.arange(10, dtype=float)
        y = x * 2 + 1
        d = CSVFunctionDiscoverer((x, y), verbose=False, db=None)
        np.testing.assert_array_equal(d._x, x)
        np.testing.assert_array_equal(d._y, y)

    def test_from_list(self):
        d = CSVFunctionDiscoverer([1.0, 2.0, 3.0], verbose=False, db=None)
        np.testing.assert_array_equal(d._x, [0, 1, 2])
        np.testing.assert_array_equal(d._y, [1, 2, 3])

    def test_empty_csv_raises(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        with pytest.raises(ValueError):
            CSVFunctionDiscoverer(str(csv_file), verbose=False, db=None)


# ================================================================
# Discovery on known functions
# ================================================================


class TestLinearDiscovery:
    """Test that a simple linear function is discovered."""

    def test_discovers_linear(self):
        x = np.arange(50, dtype=float)
        y = 3.0 * x + 7.0
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 1e-6
        assert result.r_squared > 0.999

    def test_evaluate_works(self):
        x = np.arange(50, dtype=float)
        y = 3.0 * x + 7.0
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        y_pred = result.evaluate(np.array([100.0]))
        assert abs(y_pred[0] - 307.0) < 1.0

    def test_predict_works(self):
        x = np.arange(50, dtype=float)
        y = 3.0 * x + 7.0
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        future = result.predict(5)
        assert len(future) == 5
        assert all(np.isfinite(future))


class TestQuadraticDiscovery:
    """Test quadratic function discovery."""

    def test_discovers_quadratic(self):
        x = np.arange(30, dtype=float)
        y = 0.5 * x ** 2 - 3.0 * x + 10.0
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 1e-4
        assert result.r_squared > 0.999


class TestSinPolyDiscovery:
    """Test that sin(polynomial(x)) functions are discovered."""

    def test_discovers_sin_linear(self):
        x = np.arange(1, 51, dtype=float)
        y = np.sin(2.0 * x + 3.0)
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 0.01, f"RMSE too high: {result.rmse}"

    def test_discovers_cos_quadratic(self):
        x = np.arange(1, 31, dtype=float)
        y = np.cos(0.1 * x ** 2 + 0.5 * x + 1.0)
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, max_degree=2, db=None)
        result = d.run()
        assert result.rmse < 0.01, f"RMSE too high: {result.rmse}"


class TestExpLogDiscovery:
    """Test exponential and logarithmic function discovery."""

    def test_discovers_exp_linear(self):
        x = np.arange(1, 40, dtype=float)
        y = np.exp(0.05 * x + 1.0)
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 0.1 * np.std(y), f"RMSE too high: {result.rmse}"

    def test_discovers_log_linear(self):
        x = np.arange(1, 50, dtype=float)
        y = np.log(3.0 * x + 2.0)
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 0.01, f"RMSE too high: {result.rmse}"


# ================================================================
# New log bases
# ================================================================


class TestLogBaseDiscovery:
    """Test log2 and log10 discovery via the base function dictionary."""

    def test_log2_in_dictionary(self):
        assert "log2" in BASE_FUNCTIONS
        bf = BASE_FUNCTIONS["log2"]
        # forward(8) = log2(8) = 3
        np.testing.assert_almost_equal(bf.forward(np.array([8.0])), [3.0])

    def test_log10_in_dictionary(self):
        assert "log10" in BASE_FUNCTIONS
        bf = BASE_FUNCTIONS["log10"]
        # forward(1000) = log10(1000) = 3
        np.testing.assert_almost_equal(bf.forward(np.array([1000.0])), [3.0])

    def test_log2_inverse_roundtrip(self):
        bf = BASE_FUNCTIONS["log2"]
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        y = bf.forward(x)
        x_back = bf.inverse(y)
        np.testing.assert_array_almost_equal(x, x_back)

    def test_log10_inverse_roundtrip(self):
        bf = BASE_FUNCTIONS["log10"]
        x = np.array([1.0, 10.0, 100.0, 1000.0])
        y = bf.forward(x)
        x_back = bf.inverse(y)
        np.testing.assert_array_almost_equal(x, x_back)

    def test_discovers_log2_linear(self):
        x = np.arange(1, 50, dtype=float)
        y = np.log2(2.0 * x + 5.0)
        c = InvertedCompositionFractal(BASE_FUNCTIONS["log2"], degree=1)
        c.fit(x, y)
        y_pred = c.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"log2 RMSE too high: {rmse}"

    def test_discovers_log10_linear(self):
        x = np.arange(1, 50, dtype=float)
        y = np.log10(3.0 * x + 1.0)
        c = InvertedCompositionFractal(BASE_FUNCTIONS["log10"], degree=1)
        c.fit(x, y)
        y_pred = c.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"log10 RMSE too high: {rmse}"


# ================================================================
# Pattern Store persistence
# ================================================================


class TestPatternStore:
    """Test save/load of patterns across runs."""

    def test_save_and_load(self, tmp_path):
        db_path = str(tmp_path / "patterns.json")
        store = PatternStore(path=db_path)

        # Create a candidate and save it
        x = np.arange(1, 30, dtype=float)
        y = np.log(3.0 * x + 2.0)
        c = InvertedCompositionFractal(BASE_FUNCTIONS["log"], degree=1)
        c.fit(x, y)

        n_saved = store.save([c], x, y, source_hint="test")
        assert n_saved >= 1
        assert os.path.exists(db_path)

        # Load and verify
        loaded = store.load()
        assert len(loaded) >= 1
        # Loaded candidate should produce similar output
        y_pred = loaded[0].evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01

    def test_deduplication(self, tmp_path):
        db_path = str(tmp_path / "patterns.json")
        store = PatternStore(path=db_path)

        x = np.arange(50, dtype=float)
        y = 2.0 * x + 1.0
        from cognitive_fractal.function_fractal import LinearFractal
        c = LinearFractal()
        c.fit(x, y)

        # Save twice
        store.save([c], x, y)
        store.save([c], x, y)

        # Should de-duplicate by formula
        loaded = store.load()
        formulas = [l.symbolic_repr() for l in loaded]
        # No exact duplicates
        assert len(formulas) == len(set(formulas))

    def test_clear(self, tmp_path):
        db_path = str(tmp_path / "patterns.json")
        store = PatternStore(path=db_path)

        x = np.arange(20, dtype=float)
        y = x * 5.0
        from cognitive_fractal.function_fractal import LinearFractal
        c = LinearFractal()
        c.fit(x, y)
        store.save([c], x, y)
        assert store.count() >= 1

        store.clear()
        assert store.count() == 0

    def test_empty_load_returns_empty(self, tmp_path):
        db_path = str(tmp_path / "nonexistent.json")
        store = PatternStore(path=db_path)
        assert store.load() == []

    def test_cross_run_learning(self, tmp_path):
        """Simulate two runs: run 1 discovers, run 2 starts with prior knowledge."""
        db_path = str(tmp_path / "patterns.json")

        # Run 1: discover log(3x+2)
        x = np.arange(1, 50, dtype=float)
        y = np.log(3.0 * x + 2.0)
        d1 = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=db_path)
        r1 = d1.run()
        assert r1.rmse < 0.01

        # Patterns should be saved
        store = PatternStore(path=db_path)
        assert store.count() > 0

        # Run 2: same data — should load prior patterns
        d2 = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=db_path)
        r2 = d2.run()
        assert r2.rmse < 0.01


# ================================================================
# Result object
# ================================================================


class TestDiscoveryResult:
    """Test the DiscoveryResult dataclass."""

    def test_summary_returns_string(self):
        r = DiscoveryResult(
            formula="2*x + 1",
            rmse=0.001,
            r_squared=0.999,
            fractal=None,
            n_points=100,
            candidates_tried=50,
        )
        s = r.summary()
        assert "2*x + 1" in s
        assert "0.001" in s

    def test_evaluate_without_fractal_returns_nan(self):
        r = DiscoveryResult(formula="none", rmse=float("inf"),
                            r_squared=0.0, fractal=None)
        y = r.evaluate(5.0)
        assert np.isnan(y[0])


# ================================================================
# CSV file round-trip
# ================================================================


class TestCSVFileRoundTrip:
    """Test discovery from an actual CSV file."""

    def test_csv_file_discovery(self, tmp_path):
        x = np.arange(1, 61, dtype=float)
        y = 5.0 * x + 2.0
        csv_file = tmp_path / "linear.csv"
        with open(csv_file, "w") as f:
            for xi, yi in zip(x, y):
                f.write(f"{xi},{yi}\n")

        d = CSVFunctionDiscoverer(str(csv_file), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 1e-4
        assert result.n_points == 60


# ================================================================
# Tan base function
# ================================================================


class TestTanBaseFunction:
    """Test tan entry in the base function dictionary."""

    def test_tan_in_dictionary(self):
        assert "tan" in BASE_FUNCTIONS
        bf = BASE_FUNCTIONS["tan"]
        assert bf.has_branches is True

    def test_tan_forward(self):
        bf = BASE_FUNCTIONS["tan"]
        np.testing.assert_almost_equal(
            bf.forward(np.array([0.0])), [0.0], decimal=10
        )
        np.testing.assert_almost_equal(
            bf.forward(np.array([np.pi / 4])), [1.0], decimal=10
        )

    def test_tan_inverse_roundtrip(self):
        bf = BASE_FUNCTIONS["tan"]
        # Values in (-pi/2, pi/2) so arctan round-trips
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        y = bf.forward(x)
        x_back = bf.inverse(y)
        np.testing.assert_array_almost_equal(x, x_back)

    def test_tan_branch_generator(self):
        bf = BASE_FUNCTIONS["tan"]
        branches = bf.branch_generator(1.0, 2)
        assert len(branches) == 5  # K=2: k in [-2, -1, 0, 1, 2]
        # Principal branch should be arctan(1) = pi/4
        assert any(abs(b - np.pi / 4) < 1e-10 for b in branches)

    def test_tan_poly_fit(self):
        """Test that InvertedCompositionFractal can fit tan(linear(x))."""
        x = np.linspace(0.1, 1.0, 50)
        y = np.tan(2.0 * x + 0.5)
        c = InvertedCompositionFractal(BASE_FUNCTIONS["tan"], degree=1)
        c.fit(x, y)
        y_pred = c.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"tan(poly) RMSE too high: {rmse}"


# ================================================================
# Chain Composition (was Double Composition Fractal)
# ================================================================


class TestChainComposition:
    """Test multi-level compositions via InvertedCompositionFractal chains: F(G(poly(x)))."""

    def test_construction(self):
        """Test that a two-function chain can be constructed."""
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["sin"], BASE_FUNCTIONS["cos"]], degree=1
        )
        assert len(ic.chain) == 2
        assert ic.chain[0].name == "sin"
        assert ic.chain[1].name == "cos"
        assert ic.degree == 1

    def test_evaluate_basic(self):
        """Test evaluation with known coefficients."""
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["exp"], BASE_FUNCTIONS["sin"]], degree=1
        )
        ic.coefficients = np.array([2.0, 1.0])  # poly = 2x + 1
        x = np.array([0.0, 0.5, 1.0])
        expected = np.exp(np.sin(2.0 * x + 1.0))
        y_pred = ic.evaluate(x)
        np.testing.assert_array_almost_equal(y_pred, expected)

    def test_exp_sin_linear(self):
        """Discover y = exp(sin(ax+b)) — non-branching outer, branching inner."""
        x = np.arange(1, 50, dtype=float)
        y = np.exp(np.sin(0.1 * x + 0.5))
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["exp"], BASE_FUNCTIONS["sin"]], degree=1
        )
        ic.fit(x, y)
        y_pred = ic.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"exp(sin(linear)) RMSE too high: {rmse}"

    def test_log_cos_linear(self):
        """Discover y = log(cos(ax+b)) — non-branching outer, branching inner.
        cos(ax+b) must stay positive for log to work."""
        x = np.linspace(0.1, 1.0, 50)
        y = np.log(np.cos(0.3 * x + 0.1))
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["log"], BASE_FUNCTIONS["cos"]], degree=1
        )
        ic.fit(x, y)
        y_pred = ic.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"log(cos(linear)) RMSE too high: {rmse}"

    def test_tanh_sin_linear(self):
        """Discover y = tanh(sin(ax+b)) — non-branching outer, branching inner."""
        x = np.arange(1, 50, dtype=float)
        y = np.tanh(np.sin(0.2 * x + 1.0))
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["tanh"], BASE_FUNCTIONS["sin"]], degree=1
        )
        ic.fit(x, y)
        y_pred = ic.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"tanh(sin(linear)) RMSE too high: {rmse}"

    def test_sin_cos_linear(self):
        """Discover y = sin(cos(ax+b)) — branching outer, branching inner."""
        x = np.linspace(0, 5, 50)
        y = np.sin(np.cos(0.5 * x + 0.3))
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["sin"], BASE_FUNCTIONS["cos"]], degree=1
        )
        ic.fit(x, y)
        y_pred = ic.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.05, f"sin(cos(linear)) RMSE too high: {rmse}"

    def test_exp_log_linear(self):
        """Discover y = exp(log(ax+b)) = ax+b — both non-branching.
        This is essentially the identity composition, should work perfectly."""
        x = np.arange(1, 50, dtype=float)
        y = np.exp(np.log(3.0 * x + 2.0))  # = 3x + 2
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["exp"], BASE_FUNCTIONS["log"]], degree=1
        )
        ic.fit(x, y)
        y_pred = ic.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01, f"exp(log(linear)) RMSE too high: {rmse}"

    def test_symbolic_repr(self):
        """Test symbolic representation format."""
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["sin"], BASE_FUNCTIONS["cos"]], degree=1
        )
        ic.coefficients = np.array([2.0, 1.0])
        s = ic.symbolic_repr()
        assert "sin" in s
        assert "cos" in s

    def test_serialization_roundtrip(self, tmp_path):
        """Test that chain-based InvertedCompositionFractal survives save/load cycle."""
        db_path = str(tmp_path / "patterns.json")
        store = PatternStore(path=db_path)

        x = np.arange(1, 50, dtype=float)
        y = np.exp(np.sin(0.1 * x + 0.5))
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["exp"], BASE_FUNCTIONS["sin"]], degree=1
        )
        ic.fit(x, y)

        n_saved = store.save([ic], x, y, source_hint="test")
        assert n_saved >= 1

        loaded = store.load()
        assert len(loaded) >= 1
        # Find the chain composition (chain length > 1)
        chain_loaded = [
            c for c in loaded
            if isinstance(c, InvertedCompositionFractal) and len(c.chain) > 1
        ]
        assert len(chain_loaded) >= 1
        y_pred = chain_loaded[0].evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.01

    def test_discoverer_finds_chain_composition(self):
        """Test that CSVFunctionDiscoverer can find exp(sin(linear))."""
        x = np.arange(1, 60, dtype=float)
        y = np.exp(np.sin(0.1 * x + 0.5))
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 0.05, f"Discoverer RMSE too high: {result.rmse}"

    def test_triple_chain_construction(self):
        """Test that a three-function chain can be constructed and evaluated."""
        ic = InvertedCompositionFractal(
            [BASE_FUNCTIONS["exp"], BASE_FUNCTIONS["sin"], BASE_FUNCTIONS["cos"]],
            degree=1,
        )
        assert len(ic.chain) == 3
        ic.coefficients = np.array([1.0, 0.5])  # poly = x + 0.5
        x = np.array([0.0, 0.5, 1.0])
        expected = np.exp(np.sin(np.cos(1.0 * x + 0.5)))
        y_pred = ic.evaluate(x)
        np.testing.assert_array_almost_equal(y_pred, expected)


# ================================================================
# Mixed Inner Fractal
# ================================================================


class TestMixedInnerFractal:
    """Test MixedInnerFractal: F(poly(x) + G(inner_poly(x)))."""

    def test_construction(self):
        """Test that MixedInnerFractal can be constructed with correct fields."""
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=2)
        assert mif.outer_func.name == "sin"
        assert mif.poly_degree == 2
        assert mif.inner_func is None
        assert mif.inner_coeffs is None
        # n_coeffs = poly_degree + 1 + 2 = 5
        assert len(mif.coefficients) == 5

    def test_evaluate_without_inner(self):
        """Without inner func, degrades to F(poly(x))."""
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=1)
        mif.coefficients = np.array([2.0, 1.0, 0.0])  # poly=2x+1, inner unused
        x = np.array([0.0, 0.5, 1.0])
        expected = np.sin(2.0 * x + 1.0)
        y_pred = mif.evaluate(x)
        np.testing.assert_array_almost_equal(y_pred, expected)

    def test_evaluate_with_inner(self):
        """Test evaluation with manually set inner function."""
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=2)
        # Poly: 1.0*x^2 + 0.0*x + 3.0
        mif.coefficients = np.array([1.0, 0.0, 3.0, 1.0, 0.0])
        mif.inner_func = BASE_FUNCTIONS["log"]
        mif.inner_coeffs = np.array([1.0, 0.0])  # inner_poly = 1*x + 0 = x
        mif.inner_scale = 1.0
        x = np.array([1.0, 2.0, 3.0])
        expected = np.sin(x ** 2 + 3.0 + 1.0 * np.log(x))
        y_pred = mif.evaluate(x)
        np.testing.assert_array_almost_equal(y_pred, expected)

    def test_discovers_sin_x2_plus_log_x(self):
        """The motivating case: sin(x² + log(x) + 3)."""
        x = np.arange(1, 61, dtype=float) * 0.1
        y = np.sin(x ** 2 + np.log(x) + 3.0)
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=2)
        mif.fit(x, y)
        y_pred = mif.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.05, f"sin(x²+log(x)+3) RMSE too high: {rmse}"
        assert mif.inner_func is not None, "No inner function discovered"
        # Simplicity scoring should prefer log (scale~1) over log10 (scale~2.3)
        assert mif.inner_func.name == "log", (
            f"Expected 'log', got '{mif.inner_func.name}' "
            f"(scale={mif.inner_scale:.4f})"
        )
        assert abs(mif.inner_scale - 1.0) < 0.01, (
            f"Expected scale ~1.0, got {mif.inner_scale:.6f}"
        )

    def test_discovers_cos_x_plus_exp_x(self):
        """Discover cos(x + exp(0.1*x))."""
        x = np.arange(1, 51, dtype=float) * 0.1
        y = np.cos(x + np.exp(0.1 * x))
        mif = MixedInnerFractal(BASE_FUNCTIONS["cos"], poly_degree=1)
        mif.fit(x, y)
        y_pred = mif.evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.05, f"cos(x+exp(0.1x)) RMSE too high: {rmse}"
        assert mif.inner_func is not None, "No inner function discovered"

    def test_symbolic_repr_with_inner(self):
        """Test formula string includes both components."""
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=2)
        mif.coefficients = np.array([1.0, 0.0, 3.0, 1.0, 0.0])
        mif.inner_func = BASE_FUNCTIONS["log"]
        mif.inner_coeffs = np.array([1.0, 0.0])
        s = mif.symbolic_repr()
        assert "sin" in s
        assert "log" in s

    def test_symbolic_repr_without_inner(self):
        """Test formula shows just F(poly) when no inner discovered."""
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=1)
        mif.coefficients = np.array([2.0, 1.0, 0.0])
        s = mif.symbolic_repr()
        assert "sin" in s
        # Should not mention any inner function
        assert "log" not in s
        assert "exp" not in s

    def test_serialization_roundtrip(self, tmp_path):
        """Test save/load via PatternStore."""
        db_path = str(tmp_path / "patterns.json")
        store = PatternStore(path=db_path)

        x = np.arange(1, 61, dtype=float) * 0.1
        y = np.sin(x ** 2 + np.log(x) + 3.0)
        mif = MixedInnerFractal(BASE_FUNCTIONS["sin"], poly_degree=2)
        mif.fit(x, y)

        n_saved = store.save([mif], x, y, source_hint="test")
        assert n_saved >= 1

        loaded = store.load()
        mixed_loaded = [c for c in loaded if isinstance(c, MixedInnerFractal)]
        assert len(mixed_loaded) >= 1
        y_pred = mixed_loaded[0].evaluate(x)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        assert rmse < 0.05, f"Loaded MixedInner RMSE too high: {rmse}"

    def test_discoverer_finds_mixed_inner(self):
        """Full CSVFunctionDiscoverer pipeline discovers sin(x²+log(x)+3)."""
        x = np.arange(1, 61, dtype=float) * 0.1
        y = np.sin(x ** 2 + np.log(x) + 3.0)
        d = CSVFunctionDiscoverer((x, y), verbose=False, passes=1, db=None)
        result = d.run()
        assert result.rmse < 0.05, f"Discoverer RMSE too high: {result.rmse}"
        assert "log" in result.formula, f"Expected 'log' in formula: {result.formula}"
