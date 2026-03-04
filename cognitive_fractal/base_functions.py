"""
Base function dictionary for compositional function discovery.

Each entry defines a mathematical building block with:
  - forward:    callable(z) -> y          (the function itself)
  - inverse:    callable(y) -> z          (principal inverse)
  - domain_check: callable(y) -> bool_array (where the inverse is valid)
  - branch_generator: callable(y_scalar, K) -> list (candidate inverse values)
  - has_branches: bool                     (whether inverse is multi-valued)

The SymbolicEngine uses this dictionary to generate InvertedCompositionFractal
candidates that discover y = F(poly(x)) via the inversion approach:
  1. Apply F⁻¹ to y  →  target inner values z
  2. Fit polynomial to (x, z)
  3. Validate: F(poly(x)) ≈ y
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class BaseFunction:
    """A dictionary entry describing a single mathematical building block."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    domain_check: Callable[[np.ndarray], np.ndarray]
    branch_generator: Optional[Callable[[float, int], List[float]]]
    has_branches: bool


# ================================================================
# Branch generator factories
# ================================================================

def _make_periodic_branch_gen(
    principal_inv: Callable[[float], float],
    period: float,
    complement: Callable[[float], float],
) -> Callable[[float, int], List[float]]:
    """Factory for periodic function branch generators.

    For sin: principal_inv=arcsin, period=2π, complement=λv: π-v
    For cos: principal_inv=arccos, period=2π, complement=λv: -v

    Returns a function (y_scalar, K) -> list of all candidate inverse values.
    Each period contributes two branches (principal and complement).
    """

    def branch_gen(y_scalar: float, K: int) -> List[float]:
        base = principal_inv(y_scalar)
        comp = complement(base)
        results = []
        for k in range(-K, K + 1):
            results.append(base + period * k)
            results.append(comp + period * k)
        return results

    return branch_gen


def _make_single_periodic_branch_gen(
    principal_inv: Callable[[float], float],
    period: float,
) -> Callable[[float, int], List[float]]:
    """Factory for functions with one branch per period (e.g. tan).

    tan is bijective within each period, so arctan(y) + k*π gives
    all solutions — no complement needed.
    """

    def branch_gen(y_scalar: float, K: int) -> List[float]:
        base = principal_inv(y_scalar)
        return [base + period * k for k in range(-K, K + 1)]

    return branch_gen


def _make_sign_branch_gen() -> Callable[[float, int], List[float]]:
    """Branch generator for sqrt: two branches (±sqrt), no periodicity."""

    def branch_gen(y_scalar: float, _K: int) -> List[float]:
        if y_scalar < 0:
            return []
        s = np.sqrt(y_scalar)
        return [s, -s]

    return branch_gen


# ================================================================
# The dictionary
# ================================================================

BASE_FUNCTIONS = {
    "sin": BaseFunction(
        name="sin",
        forward=np.sin,
        inverse=lambda y: np.arcsin(np.clip(y, -0.9999, 0.9999)),
        domain_check=lambda y: np.abs(y) <= 1.05,
        branch_generator=_make_periodic_branch_gen(
            principal_inv=lambda v: float(np.arcsin(np.clip(v, -0.9999, 0.9999))),
            period=2 * np.pi,
            complement=lambda v: np.pi - v,
        ),
        has_branches=True,
    ),
    "cos": BaseFunction(
        name="cos",
        forward=np.cos,
        inverse=lambda y: np.arccos(np.clip(y, -0.9999, 0.9999)),
        domain_check=lambda y: np.abs(y) <= 1.05,
        branch_generator=_make_periodic_branch_gen(
            principal_inv=lambda v: float(np.arccos(np.clip(v, -0.9999, 0.9999))),
            period=2 * np.pi,
            complement=lambda v: -v,
        ),
        has_branches=True,
    ),
    "exp": BaseFunction(
        name="exp",
        forward=lambda z: np.exp(np.clip(z, -50, 50)),
        inverse=lambda y: np.log(np.maximum(y, 1e-12)),
        domain_check=lambda y: y > 0,
        branch_generator=None,
        has_branches=False,
    ),
    "log": BaseFunction(
        name="log",
        forward=lambda z: np.log(np.maximum(z, 1e-12)),
        inverse=lambda y: np.exp(np.clip(y, -50, 50)),
        domain_check=lambda y: np.ones_like(y, dtype=bool),
        branch_generator=None,
        has_branches=False,
    ),
    "tanh": BaseFunction(
        name="tanh",
        forward=np.tanh,
        inverse=lambda y: np.arctanh(np.clip(y, -0.9999, 0.9999)),
        domain_check=lambda y: np.abs(y) < 1.0,
        branch_generator=None,
        has_branches=False,
    ),
    "sqrt": BaseFunction(
        name="sqrt",
        forward=lambda z: np.sqrt(np.maximum(z, 0)),
        inverse=lambda y: y ** 2,
        domain_check=lambda y: y >= 0,
        branch_generator=_make_sign_branch_gen(),
        has_branches=True,
    ),
    "log2": BaseFunction(
        name="log2",
        forward=lambda z: np.log2(np.maximum(z, 1e-12)),
        inverse=lambda y: np.power(2.0, np.clip(y, -50, 50)),
        domain_check=lambda y: np.ones_like(y, dtype=bool),
        branch_generator=None,
        has_branches=False,
    ),
    "log10": BaseFunction(
        name="log10",
        forward=lambda z: np.log10(np.maximum(z, 1e-12)),
        inverse=lambda y: np.power(10.0, np.clip(y, -20, 20)),
        domain_check=lambda y: np.ones_like(y, dtype=bool),
        branch_generator=None,
        has_branches=False,
    ),
    "tan": BaseFunction(
        name="tan",
        forward=np.tan,
        inverse=lambda y: np.arctan(y),
        domain_check=lambda y: np.ones_like(y, dtype=bool),
        branch_generator=_make_single_periodic_branch_gen(
            principal_inv=lambda v: float(np.arctan(v)),
            period=np.pi,
        ),
        has_branches=True,
    ),
}
