"""The numerical search over complete Bernstein functions (Appendix G.1).

For a complete Bernstein function φ with Lévy density m and no drift, the smallest
contraction factor across dimensions at discount λ (ρ = λ²) is

    τ⋆(λ) = sup_{s>0} m(s/ρ) / (ρ m(s)).

The search writes s = e^u and maximizes ℓ_λ(u) = log m(e^u/ρ) − log m(e^u) − log ρ on one
window [−U, U]: a grid, then a bounded local refinement around the best grid points. It also
reports whether the best point is an end of the window, where the supremum may lie beyond.
The densities are evaluated with mpmath; the grid and the refinement work in float64.

The 43 entries are those of Tables 3–6, by their number in Schilling, Song and Vondraček's
table of complete Bernstein functions.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import mpmath as mp  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
    OptimizeResult,
    minimize_scalar,  # pyright: ignore[reportUnknownVariableType]
)

# An mpmath real number; mpmath has no type information, so its numbers are typed as Any
type Number = Any
type ParameterValues = Mapping[str, Number]


@dataclass(frozen=True)
class CompleteBernsteinEntry:
    """One complete Bernstein function of Tables 3–6 (Appendix G.1)."""

    schilling_number: int
    """Its number in Schilling, Song and Vondraček's table; Figures 6–11 call it "Entry N"."""
    table_key: int
    """Its key, 1 to 43, in the paper's tables."""
    bernstein_function: Callable[[Number, ParameterValues], Number]
    """φ(t), at the given parameters."""
    levy_density: Callable[[Number, ParameterValues], Number]
    """m(s), the density of its Lévy measure, at the given parameters."""
    default_parameters: Mapping[str, float]
    """The default parameters of Tables 3–6, for example {"alpha": 0.5} for entry 84."""


@dataclass(frozen=True)
class SupremumSearchSettings:
    """The constants of the search (Table 7)."""

    half_width: float
    """U, the half width of the window [−U, U]."""
    grid_size: int
    """n_grid, the number of grid points on the window."""
    refined_point_count: int
    """K, the number of best grid points refined locally."""
    refinement_half_width: float
    """Half width of each local refinement interval, in grid steps."""
    refinement_tolerance: float
    """Tolerance on u of the bounded scalar optimization."""
    decimal_digits: int
    """Decimal digits of the mpmath evaluations of the density."""


@dataclass(frozen=True)
class RatioSupremum:
    """The result of the search at one discount λ."""

    discount: float
    """λ."""
    estimate: float
    """τ̂⋆(λ), the estimate of τ⋆(λ): the largest ratio m(s/λ²)/(λ² m(s)) found."""
    best_log_argument: float
    """u⋆ = log s⋆, where it was found."""
    at_window_edge: bool
    """Whether u⋆ is an end of the window; the supremum may then lie beyond it."""


def _build_log_ratio(
    entry: CompleteBernsteinEntry, parameters: Mapping[str, float], discount: float
) -> Callable[[float], float]:
    """Build ℓ_λ(u) = log m(e^u/ρ) − log m(e^u) − log ρ, with ρ = λ², as a float function.

    Raises:
        ValueError: If the density is not positive where it is evaluated.
    """
    # Squared discount ρ = λ²
    squared_discount = discount * discount
    # log ρ
    log_squared_discount = math.log(squared_discount)
    # Parameters as mpmath numbers
    parameter_values = {name: mp.mpf(value) for name, value in parameters.items()}

    def compute_log_density(argument: float) -> float:
        # Density m(s) at the float argument s
        density = entry.levy_density(mp.mpf(argument), parameter_values)
        if not density > 0:
            raise ValueError(f"entry {entry.schilling_number}: density {density} at s = {argument}")
        # log m(s), rounded to float64
        return float(mp.log(density))

    def compute_log_ratio(log_argument: float) -> float:
        # Argument s = e^u
        argument = math.exp(log_argument)
        # ℓ_λ(u) = log m(s/ρ) − log m(s) − log ρ
        return (
            compute_log_density(argument / squared_discount)
            - compute_log_density(argument)
            - log_squared_discount
        )

    return compute_log_ratio


def _maximize_on_window(
    log_ratio: Callable[[float], float], settings: SupremumSearchSettings
) -> tuple[float, float]:
    """Maximize ℓ_λ on [−U, U]: the grid, then a bounded refinement around its K best points.

    Returns:
        The best point u⋆ and its value ℓ_λ(u⋆).
    """
    half_width, grid_size = settings.half_width, settings.grid_size
    # Grid on [−U, U], shape (n,)
    grid = np.linspace(-half_width, half_width, grid_size)
    # ℓ_λ on the grid, shape (n,)
    values = np.array([log_ratio(float(point)) for point in grid])
    # Indices of the K largest values, largest first, shape (K,)
    count = min(settings.refined_point_count, grid_size)
    best_indices = np.argpartition(values, -count)[-count:]
    best_indices = best_indices[np.argsort(-values[best_indices])]
    # Best grid point and its value, the starting answer
    best_point, best_value = float(grid[best_indices[0]]), float(values[best_indices[0]])
    # Grid step Δu
    step = 2.0 * half_width / max(1, grid_size - 1)
    for index in best_indices:
        # Local interval around the grid point, inside [−U, U]
        center = float(grid[int(index)])
        lower = max(-half_width, center - settings.refinement_half_width * step)
        upper = min(half_width, center + settings.refinement_half_width * step)
        if not lower < upper:
            continue
        # Bounded scalar maximization of ℓ_λ on the local interval
        refined = cast(
            OptimizeResult,
            minimize_scalar(
                lambda point: -log_ratio(float(point)),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": settings.refinement_tolerance},
            ),
        )
        # Its value ℓ_λ, kept when it beats the best so far
        refined_value = -float(refined["fun"])
        if np.isfinite(refined_value) and refined_value > best_value:
            best_point, best_value = float(refined["x"]), refined_value
    return best_point, best_value


def find_ratio_supremum(
    entry: CompleteBernsteinEntry,
    parameters: Mapping[str, float],
    discount: float,
    settings: SupremumSearchSettings,
) -> RatioSupremum:
    """Estimate τ⋆(λ) = sup_{s>0} m(s/λ²)/(λ² m(s)) by the search of Algorithm 2.

    Args:
        entry: The complete Bernstein function.
        parameters: Its parameters.
        discount: λ, in (0, 1).
        settings: The constants of the search.

    Returns:
        The estimate, where it was found, and whether that is an end of the window.

    Raises:
        ValueError: If λ is not in (0, 1), or if the density is not positive where it is
            evaluated.
    """
    if not 0.0 < discount < 1.0:
        raise ValueError(f"the discount must be in (0, 1), got {discount}")
    with mp.workdps(settings.decimal_digits):
        log_ratio = _build_log_ratio(entry, parameters, discount)
        # Best point u⋆ and its value ℓ_λ(u⋆) on the window
        best_point, best_value = _maximize_on_window(log_ratio, settings)
    return RatioSupremum(
        discount=discount,
        estimate=math.exp(best_value),
        best_log_argument=best_point,
        at_window_edge=abs(best_point) >= settings.half_width,
    )


def compute_ratio_at_log_argument(
    entry: CompleteBernsteinEntry,
    parameters: Mapping[str, float],
    discount: float,
    log_argument: float,
    decimal_digits: int,
) -> float:
    """Compute R_λ(e^u) = m(e^u/λ²) / (λ² m(e^u)) entirely in mpmath, for any u.

    The search computes s = e^u in double precision, which rounds to 0 for u below about −745.
    Here e^u is computed in mpmath, so u can lie far outside the search window, for example to
    see where the ratio goes as s → 0⁺.

    Args:
        entry: The complete Bernstein function.
        parameters: Its parameters.
        discount: λ, in (0, 1).
        log_argument: u = log s.
        decimal_digits: Decimal digits of the mpmath evaluations.

    Returns:
        R_λ(e^u), rounded to double precision.

    Raises:
        ValueError: If λ is not in (0, 1), or if the density is not positive at e^u or e^u/λ².
    """
    if not 0.0 < discount < 1.0:
        raise ValueError(f"the discount must be in (0, 1), got {discount}")
    with mp.workdps(decimal_digits):
        # Squared discount ρ = λ²
        squared_discount = mp.mpf(discount) ** 2
        # Parameters as mpmath numbers
        parameter_values = {name: mp.mpf(value) for name, value in parameters.items()}
        # Argument s = e^u
        argument = mp.exp(mp.mpf(log_argument))
        # m(s/ρ) and m(s)
        scaled_density = entry.levy_density(argument / squared_discount, parameter_values)
        density = entry.levy_density(argument, parameter_values)
        if not (scaled_density > 0 and density > 0):
            raise ValueError(
                f"entry {entry.schilling_number}: density not positive at u = {log_argument}"
            )
        # R_λ(s) = m(s/ρ) / (ρ m(s))
        return float(scaled_density / (squared_discount * density))


def classify_estimate_curve(
    estimates: Sequence[float], at_window_edge: Sequence[bool], tolerance: float
) -> Literal["accepted", "borderline", "rejected"]:
    """Classify a curve τ̂⋆(λ) over the discounts by its largest value and where its best points lie.

    A value above 1 rejects the entry, because each estimate is the ratio at one point and so
    never exceeds τ⋆(λ). A curve below 1 is accepted only if no best point is an end of the
    window: there, the supremum may lie beyond the window (Entry 78).

    Args:
        estimates: τ̂⋆(λ) at each discount.
        at_window_edge: Whether the best point is an end of the last window, at each discount.
        tolerance: Values within this distance of 1 count as 1.

    Returns:
        "rejected" if the largest value is above 1 + tolerance; "accepted" if it is below
        1 − tolerance and no best point is an end of the window; "borderline" otherwise.
    """
    # Largest value over the discounts
    largest = max(estimates)
    if largest > 1.0 + tolerance:
        return "rejected"
    if largest < 1.0 - tolerance and not any(at_window_edge):
        return "accepted"
    return "borderline"


def _subtract_from_one_exp(value: Number) -> Number:
    """1 − e^(−x), accurate for small x."""
    return -mp.expm1(-value)


def _evaluate_small_argument_series(
    value: Number, coefficients: Sequence[str], cutoff: str, formula: Callable[[Number], Number]
) -> Number:
    """Evaluate a density by its Taylor series below the cutoff, by its formula above.

    The formulas of entries 45 and 46 cancel to many digits near 0; their series do not.
    """
    if abs(value) < mp.mpf(cutoff):
        return sum((mp.mpf(c) * value**k for k, c in enumerate(coefficients)), mp.mpf(0))
    return formula(value)


# The entries, in the order of the tables. Each pair of functions takes (t or s, parameters).
# Densities in the tables written with e^(x) − 1 or e^(2as) − 1 are rewritten with 1 − e^(−x),
# which is the same value and does not overflow.
_ENTRY_DEFINITIONS: tuple[
    tuple[
        int,
        int,
        Callable[[Number, ParameterValues], Number],
        Callable[[Number, ParameterValues], Number],
        dict[str, float],
    ],
    ...,
] = (
    (
        1,
        1,
        lambda t, p: t ** p["alpha"],
        lambda s, p: p["alpha"] / mp.gamma(1 - p["alpha"]) * s ** (-1 - p["alpha"]),
        {"alpha": 0.6},
    ),
    (
        2,
        2,
        lambda t, p: (t + 1) ** p["alpha"] - 1,
        lambda s, p: p["alpha"] / mp.gamma(1 - p["alpha"]) * mp.exp(-s) * s ** (-1 - p["alpha"]),
        {"alpha": 0.6},
    ),
    (
        3,
        3,
        lambda t, p: 1 - (1 + t) ** (p["alpha"] - 1),
        lambda s, p: mp.exp(-s) * s ** (-p["alpha"]) / mp.gamma(1 - p["alpha"]),
        {"alpha": 0.6},
    ),
    (
        4,
        4,
        lambda t, p: t / (t + p["a"]),
        lambda s, p: p["a"] * mp.exp(-p["a"] * s),
        {"a": 1.0},
    ),
    (
        5,
        5,
        lambda t, p: t / mp.sqrt(t + p["a"]),
        lambda s, p: (
            mp.exp(-p["a"] * s) * (2 * p["a"] * s + 1) / (2 * mp.sqrt(mp.pi) * s ** mp.mpf(1.5))
        ),
        {"a": 1.0},
    ),
    (
        6,
        6,
        lambda t, p: t / (t + p["a"]) ** p["alpha"],
        lambda s, p: (
            mp.sin(p["alpha"] * mp.pi)
            * mp.gamma(1 - p["alpha"])
            / mp.pi
            * mp.exp(-p["a"] * s)
            * s ** (p["alpha"] - 2)
            * (p["a"] * s + 1 - p["alpha"])
        ),
        {"a": 1.0, "alpha": 0.5},
    ),
    (
        7,
        7,
        lambda t, p: mp.sqrt(t / (t + p["a"])),
        lambda s, p: (
            p["a"]
            / 2
            * mp.exp(-p["a"] * s / 2)
            * (mp.besseli(0, p["a"] * s / 2) - mp.besseli(1, p["a"] * s / 2))
        ),
        {"a": 1.0},
    ),
    (
        14,
        8,
        lambda t, p: (
            mp.pi / mp.sin(p["alpha"] * mp.pi) * p["alpha"] * p["c"] ** p["alpha"]
            if t == p["c"]
            else mp.pi
            / mp.sin(p["alpha"] * mp.pi)
            * t
            * (p["c"] ** p["alpha"] - t ** p["alpha"])
            / (p["c"] - t)
        ),
        lambda s, p: (
            mp.gamma(p["alpha"] + 2)
            * p["c"] ** (p["alpha"] + 1)
            * mp.exp(p["c"] * s)
            * mp.gammainc(-p["alpha"] - 1, p["c"] * s, mp.inf)
        ),
        {"alpha": 0.5, "c": 2.0},
    ),
    (
        18,
        9,
        lambda t, p: mp.sqrt(t) * (1 - mp.exp(-2 * p["a"] * mp.sqrt(t))),
        lambda s, p: (
            ((2 * p["a"] ** 2 - s) * mp.exp(-(p["a"] ** 2) / s) + s)
            / (2 * mp.sqrt(mp.pi) * s ** mp.mpf(2.5))
        ),
        {"a": 1.0},
    ),
    (
        19,
        10,
        lambda t, p: mp.sqrt(t) * (1 + mp.exp(-2 * p["a"] * mp.sqrt(t))),
        lambda s, p: (
            ((s - 2 * p["a"] ** 2) * mp.exp(-(p["a"] ** 2) / s) + s)
            / (2 * mp.sqrt(mp.pi) * s ** mp.mpf(2.5))
        ),
        {"a": 1.0},
    ),
    (
        20,
        11,
        lambda t, p: t * (1 - mp.exp(-2 * mp.sqrt(t + p["a"]))) / mp.sqrt(t + p["a"]),
        # e^(−1/s − as) (2 + s (e^(1/s) − 1)(1 + 2as)), with the e^(1/s) term multiplied out
        lambda s, p: (
            (
                2 * mp.exp(-1 / s - p["a"] * s)
                + s * mp.exp(-p["a"] * s) * _subtract_from_one_exp(1 / s) * (1 + 2 * p["a"] * s)
            )
            / (2 * mp.sqrt(mp.pi) * s ** mp.mpf(2.5))
        ),
        {"a": 1.0},
    ),
    (
        26,
        12,
        lambda t, p: mp.log(1 + t / p["a"]),
        lambda s, p: mp.exp(-p["a"] * s) / s,
        {"a": 1.0},
    ),
    (
        27,
        13,
        lambda t, p: t * mp.log(1 + p["a"] / t),
        # 1 − e^(−as)(1 + as) = e^(−as)(e^(as) − 1 − as)
        lambda s, p: mp.exp(-p["a"] * s) * (mp.expm1(p["a"] * s) - p["a"] * s) / s**2,
        {"a": 1.0},
    ),
    (
        28,
        14,
        lambda t, p: mp.log(p["b"] * (t + p["a"]) / (p["a"] * (t + p["b"]))),
        lambda s, p: mp.exp(-p["a"] * s) * _subtract_from_one_exp((p["b"] - p["a"]) * s) / s,
        {"a": 1.0, "b": 3.0},
    ),
    (
        29,
        15,
        lambda t, p: 1 / p["a"] - mp.log(1 + t / p["a"]) / t,
        lambda s, p: -mp.ei(-p["a"] * s),
        {"a": 1.0},
    ),
    (
        30,
        16,
        lambda t, p: (p["a"] + t) / (p["a"] * t) * mp.log(1 + t / p["a"]) - 1 / p["a"],
        lambda s, p: mp.exp(-p["a"] * s) / (p["a"] * s) + mp.ei(-p["a"] * s),
        {"a": 1.0},
    ),
    (
        31,
        17,
        lambda t, p: (t + p["a"]) * mp.log(t + p["a"]) - t * mp.log(t) - p["a"] * mp.log(p["a"]),
        lambda s, p: _subtract_from_one_exp(p["a"] * s) / s**2,
        {"a": 1.0},
    ),
    (
        32,
        18,
        lambda t, p: (
            (t + p["b"]) * mp.log(t + p["b"])
            - p["b"] * mp.log(p["b"])
            - (t + p["a"]) * mp.log(t + p["a"])
            + p["a"] * mp.log(p["a"])
        ),
        lambda s, p: mp.exp(-p["a"] * s) * _subtract_from_one_exp((p["b"] - p["a"]) * s) / s**2,
        {"a": 1.0, "b": 3.0},
    ),
    (
        33,
        19,
        lambda t, p: mp.mpf(1) if t == p["a"] else t / (p["a"] - t) * mp.log(p["a"] / t),
        lambda s, p: p["a"] * mp.exp(p["a"] * s) * mp.ei(-p["a"] * s) + 1 / s,
        {"a": 1.0},
    ),
    (
        38,
        20,
        lambda t, p: (
            (p["a"] ** 2 + p["a"] * mp.sqrt(p["a"] ** 2 + 2 * t)) / 2
            - p["a"] ** 2
            + t * mp.log(mp.sqrt(1 + p["a"] ** 2 / (2 * t)) + p["a"] / mp.sqrt(2 * t))
        ),
        lambda s, p: mp.erf(p["a"] * mp.sqrt(s / 2)) / (2 * s**2),
        {"a": 1.0},
    ),
    (
        45,
        21,
        lambda t, p: t * (1 + t) * mp.log(1 + 1 / t) - t,
        lambda s, p: _evaluate_small_argument_series(
            s,
            ("1/6", "-1/12", "1/40", "-1/180", "1/1008"),
            "1e-4",
            lambda x: (mp.exp(-x) * (2 + x) + x - 2) / x**3,
        ),
        {},
    ),
    (
        46,
        22,
        lambda t, p: t * (1 - t * mp.log(1 + 1 / t)),
        lambda s, p: _evaluate_small_argument_series(
            s,
            ("1/3", "-1/4", "1/10", "-1/36", "1/168"),
            "1e-4",
            lambda x: (2 - mp.exp(-x) * (2 + 2 * x + x**2)) / x**3,
        ),
        {},
    ),
    (
        57,
        23,
        lambda t, p: mp.log(1 + 2 * t + 2 * mp.sqrt(t * (1 + t))),
        lambda s, p: mp.exp(-s / 2) * mp.besseli(0, s / 2) / s,
        {},
    ),
    (
        63,
        24,
        lambda t, p: mp.sqrt(t) * mp.atan(mp.sqrt(p["a"] / t)),
        lambda s, p: mp.gammainc(mp.mpf(1.5), 0, p["a"] * s) / (2 * s ** mp.mpf(1.5)),
        {"a": 1.0},
    ),
    (
        64,
        25,
        lambda t, p: mp.sqrt(t) * mp.atan(mp.sqrt(t / p["a"])),
        lambda s, p: mp.gammainc(mp.mpf(1.5), p["a"] * s, mp.inf) / (2 * s ** mp.mpf(1.5)),
        {"a": 1.0},
    ),
    (
        77,
        26,
        lambda t, p: mp.acosh(1 + t),
        lambda s, p: mp.exp(-s) * mp.besseli(0, s) / s,
        {},
    ),
    (
        78,
        27,
        lambda t, p: mp.acosh(1 + t) ** 2 / 2,
        lambda s, p: mp.exp(-s) * mp.besselk(0, s) / s,
        {},
    ),
    (
        82,
        28,
        lambda t, p: mp.gamma((t + p["a"]) / (2 * p["a"])) / mp.gamma(t / (2 * p["a"])),
        lambda s, p: (
            p["a"]
            * mp.exp(-p["a"] * s)
            / (mp.sqrt(mp.pi) * _subtract_from_one_exp(2 * p["a"] * s) ** mp.mpf(1.5))
        ),
        {"a": 1.0},
    ),
    (
        83,
        29,
        lambda t, p: mp.gamma(p["alpha"] * t + 1) / mp.gamma(p["alpha"] * t + 1 - p["alpha"]),
        lambda s, p: (
            mp.exp(-s / p["alpha"])
            / (
                mp.gamma(1 - p["alpha"])
                * _subtract_from_one_exp(s / p["alpha"]) ** (1 + p["alpha"])
            )
        ),
        {"alpha": 0.5},
    ),
    (
        84,
        30,
        lambda t, p: t * mp.gamma(p["alpha"] * t + 1 - p["alpha"]) / mp.gamma(p["alpha"] * t + 1),
        lambda s, p: (
            (1 - p["alpha"])
            / (p["alpha"] * mp.gamma(1 + p["alpha"]))
            * mp.exp(-(1 - p["alpha"]) * s / p["alpha"])
            / _subtract_from_one_exp(s / p["alpha"]) ** (2 - p["alpha"])
        ),
        {"alpha": 0.5},
    ),
    (
        85,
        31,
        lambda t, p: t ** (1 - p["nu"]) * mp.exp(t) * mp.gammainc(p["nu"], t, mp.inf),
        lambda s, p: mp.gamma(2 - p["nu"]) / mp.gamma(1 - p["nu"]) * (1 + s) ** (p["nu"] - 2),
        {"nu": 0.5},
    ),
    (
        86,
        32,
        lambda t, p: (
            t ** (1 - p["nu"]) * mp.exp(p["a"] * t) * mp.gammainc(p["nu"], p["a"] * t, mp.inf)
        ),
        lambda s, p: mp.gamma(2 - p["nu"]) / mp.gamma(1 - p["nu"]) * (p["a"] + s) ** (p["nu"] - 2),
        {"a": 1.0, "nu": 0.5},
    ),
    (
        88,
        33,
        lambda t, p: (
            mp.loggamma(t + p["a"] + p["b"])
            - mp.loggamma(t + p["a"])
            + mp.loggamma(p["a"])
            - mp.loggamma(p["a"] + p["b"])
        ),
        lambda s, p: (
            mp.exp(-p["a"] * s)
            * _subtract_from_one_exp(p["b"] * s)
            / (s * _subtract_from_one_exp(s))
        ),
        {"a": 1.0, "b": 1.0},
    ),
    (
        99,
        34,
        lambda t, p: t * mp.exp(t) * mp.expint(int(p["n"]), t),
        lambda s, p: p["n"] / (1 + s) ** (p["n"] + 1),
        {"n": 1.0},
    ),
    (
        100,
        35,
        lambda t, p: -t * mp.exp(p["a"] * t) * mp.ei(-p["a"] * t),
        lambda s, p: 1 / (p["a"] + s) ** 2,
        {"a": 1.0},
    ),
    (
        101,
        36,
        lambda t, p: t * mp.exp(p["a"] * t) * (mp.ei(-p["a"] * (p["b"] + t)) - mp.ei(-p["a"] * t)),
        lambda s, p: (
            (1 - (p["b"] * (s + p["a"]) + 1) * mp.exp(-(s + p["a"]) * p["b"])) / (s + p["a"]) ** 2
        ),
        {"a": 1.0, "b": 1.0},
    ),
    (
        102,
        37,
        lambda t, p: -t * mp.exp(p["a"] * t) * mp.ei(-p["a"] * (p["b"] + t)),
        lambda s, p: (
            (p["b"] * (s + p["a"]) + 1) * mp.exp(-(s + p["a"]) * p["b"]) / (s + p["a"]) ** 2
        ),
        {"a": 1.0, "b": 1.0},
    ),
    (
        103,
        38,
        lambda t, p: (
            (-1) ** (int(p["n"]) + 1) * t ** (p["n"] + 1) * mp.exp(p["a"] * t) * mp.ei(-p["a"] * t)
            + sum(
                (
                    (-1) ** (int(p["n"]) - k)
                    * mp.factorial(k - 1)
                    * p["a"] ** (-k)
                    * t ** (p["n"] - k + 1)
                    for k in range(1, int(p["n"]) + 1)
                ),
                mp.mpf(0),
            )
        ),
        lambda s, p: mp.gamma(p["n"] + 2) / (p["a"] + s) ** (p["n"] + 2),
        {"a": 1.0, "n": 1.0},
    ),
    (
        104,
        39,
        lambda t, p: mp.euler + mp.log(p["a"] * t) - mp.exp(p["a"] * t) * mp.ei(-p["a"] * t),
        lambda s, p: p["a"] / (s * (p["a"] + s)),
        {"a": 1.0},
    ),
    (
        108,
        40,
        lambda t, p: mp.sqrt(mp.pi * t) * mp.exp(p["a"] * t) * mp.erfc(mp.sqrt(p["a"] * t)),
        lambda s, p: 1 / (2 * (p["a"] + s) ** mp.mpf(1.5)),
        {"a": 1.0},
    ),
    (
        109,
        41,
        lambda t, p: (
            mp.sqrt(mp.pi / p["a"]) * t
            - mp.pi * t ** mp.mpf(1.5) * mp.exp(p["a"] * t) * mp.erfc(mp.sqrt(p["a"] * t))
        ),
        lambda s, p: mp.gamma(mp.mpf(2.5)) / (p["a"] + s) ** mp.mpf(2.5),
        {"a": 1.0},
    ),
    (
        131,
        42,
        lambda t, p: (
            -((2 * p["a"]) ** p["nu"])
            * mp.gamma(-p["nu"])
            * t
            * mp.gamma(1 + p["nu"] + t / (2 * p["a"]))
            / ((2 * p["a"] * p["nu"] + t) * mp.gamma(1 + t / (2 * p["a"])))
        ),
        # (a / sinh(as))^(1+ν) e^((1−ν)as) = (2a)^(1+ν) e^(−2ν as) / (1 − e^(−2as))^(1+ν)
        lambda s, p: (
            (2 * p["a"]) ** (1 + p["nu"])
            * mp.exp(-2 * p["nu"] * p["a"] * s)
            / _subtract_from_one_exp(2 * p["a"] * s) ** (1 + p["nu"])
        ),
        {"a": 1.0, "nu": 0.5},
    ),
    (
        132,
        43,
        lambda t, p: (
            (2 * p["a"]) ** p["nu"]
            * mp.gamma(-p["nu"])
            * (
                mp.gamma((1 + p["nu"]) / 2) / mp.gamma(1 - (1 + p["nu"]) / 2)
                - mp.gamma((1 + p["nu"]) / 2 + t / (2 * p["a"]))
                / mp.gamma(1 - (1 + p["nu"]) / 2 + t / (2 * p["a"]))
            )
        ),
        # (a / sinh(as))^(1+ν) = (2a)^(1+ν) e^(−(1+ν) as) / (1 − e^(−2as))^(1+ν)
        lambda s, p: (
            (2 * p["a"]) ** (1 + p["nu"])
            * mp.exp(-(1 + p["nu"]) * p["a"] * s)
            / _subtract_from_one_exp(2 * p["a"] * s) ** (1 + p["nu"])
        ),
        {"a": 1.0, "nu": 0.5},
    ),
)

ENTRIES: dict[int, CompleteBernsteinEntry] = {
    schilling_number: CompleteBernsteinEntry(
        schilling_number, table_key, bernstein_function, levy_density, default_parameters
    )
    for (
        schilling_number,
        table_key,
        bernstein_function,
        levy_density,
        default_parameters,
    ) in _ENTRY_DEFINITIONS
}
"""The 43 entries of Tables 3–6, by Schilling number."""
