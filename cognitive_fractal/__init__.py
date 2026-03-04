"""
cognitive_fractal — A recursive learning system built from self-similar fractals.

Each Fractal is an active learning agent that compares, predicts, and updates.
The same mechanism operates at every level of the hierarchy.
"""

from .fractal import Fractal
from .engine import FractalEngine
from .memory import Memory
from .types import Signal, Feedback, Metrics
from .compare import deviation, deviation_magnitude, ratio, similarity
from .function_fractal import (
    FunctionFractal,
    ConstantFractal,
    LinearFractal,
    QuadraticFractal,
    PolynomialFractal,
    SinFractal,
    CosFractal,
    GradientSinFractal,
    GradientCosFractal,
    ExponentialFractal,
    LogFractal,
    ComposedFunctionFractal,
)
from .nested_fractal import NestedComposedFractal
from .base_functions import BASE_FUNCTIONS, BaseFunction
from .inverted_composition import InvertedCompositionFractal
from .mixed_inner import MixedInnerFractal
from .symbolic_engine import SymbolicEngine
from .predictor import SequencePredictor
from .csv_discoverer import CSVFunctionDiscoverer, DiscoveryResult
from .pattern_store import PatternStore

__all__ = [
    "Fractal",
    "FractalEngine",
    "Memory",
    "Signal",
    "Feedback",
    "Metrics",
    "deviation",
    "deviation_magnitude",
    "ratio",
    "similarity",
    "FunctionFractal",
    "ConstantFractal",
    "LinearFractal",
    "QuadraticFractal",
    "PolynomialFractal",
    "SinFractal",
    "CosFractal",
    "GradientSinFractal",
    "GradientCosFractal",
    "ExponentialFractal",
    "LogFractal",
    "ComposedFunctionFractal",
    "NestedComposedFractal",
    "BASE_FUNCTIONS",
    "BaseFunction",
    "InvertedCompositionFractal",
    "MixedInnerFractal",
    "SymbolicEngine",
    "SequencePredictor",
    "CSVFunctionDiscoverer",
    "DiscoveryResult",
    "PatternStore",
]
