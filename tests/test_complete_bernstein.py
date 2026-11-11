"""Tests of the search over complete Bernstein functions and of the 43 entries."""

import math

import mpmath as mp  # pyright: ignore[reportMissingTypeStubs]
import pytest

from mmd_contraction.complete_bernstein import (
    ENTRIES,
    Number,
    SupremumSearchSettings,
    classify_estimate_curve,
    compute_ratio_at_log_argument,
    find_ratio_supremum,
)

PAPER_SETTINGS = SupremumSearchSettings(
    half_width=80.0,
    grid_size=8500,
    refined_point_count=12,
    refinement_half_width=120.0,
    refinement_tolerance=1e-10,
    decimal_digits=80,
)


@pytest.mark.parametrize("discount", [0.3, 0.9])
def test_power_function_has_the_constant_ratio_discount_to_the_two_alpha(discount: float) -> None:
    """For φ(t) = t^α (entry 1), m(s/ρ)/(ρ m(s)) = ρ^α at every s, so τ⋆(λ) = λ^(2α)."""
    result = find_ratio_supremum(ENTRIES[1], {"alpha": 0.6}, discount, PAPER_SETTINGS)

    assert result.estimate == pytest.approx(discount ** (2 * 0.6), rel=1e-10)


@pytest.mark.parametrize("log_argument", [-8000.0, 8000.0])
def test_power_function_ratio_far_outside_the_window_is_discount_to_the_two_alpha(
    log_argument: float,
) -> None:
    """For φ(t) = t^α (entry 1), R_λ(e^u) = λ^(2α) at every u, also far beyond u = ±745."""
    ratio = compute_ratio_at_log_argument(ENTRIES[1], {"alpha": 0.6}, 0.9, log_argument, 80)

    assert ratio == pytest.approx(0.9 ** (2 * 0.6), rel=1e-14)


@pytest.mark.parametrize("discount", [0.3, 0.9])
def test_shifted_power_reaches_discount_to_the_two_alpha_as_s_goes_to_zero(
    discount: float,
) -> None:
    """For φ(t) = (1 + t)^α − 1 (entry 2), the ratio is ρ^α e^(−s(1/ρ − 1)).

    It increases to ρ^α as s → 0, so τ⋆(λ) = λ^(2α), approached but not attained; the search
    finds it to 10⁻⁸ inside the window.
    """
    result = find_ratio_supremum(ENTRIES[2], {"alpha": 0.6}, discount, PAPER_SETTINGS)

    assert result.estimate == pytest.approx(discount ** (2 * 0.6), rel=1e-8)
    assert not result.at_window_edge


@pytest.mark.parametrize("schilling_number", sorted(ENTRIES))
def test_bernstein_function_and_levy_density_agree(schilling_number: int) -> None:
    """φ(t₂) − φ(t₁) = b (t₂ − t₁) + ∫₀^∞ (e^(−t₁ s) − e^(−t₂ s)) m(s) ds, with drift b ≥ 0.

    This is the Lévy–Khintchine representation of φ between two arguments, so φ(0) and
    the drift need not be known: the drift is read from (t₁, t₂) = (1, 2) and must be the
    same, to 10⁻⁸, for (2, 4), and nonnegative. It checks each entry's φ against its m.
    """
    entry = ENTRIES[schilling_number]
    with mp.workdps(30):
        parameters = {name: mp.mpf(value) for name, value in entry.default_parameters.items()}

        def compute_remainder(first: int, second: int) -> Number:
            # ∫₀^∞ (e^(−t₁ s) − e^(−t₂ s)) m(s) ds, split where the density changes scale
            integral = mp.quad(
                lambda s: (
                    (mp.exp(-first * s) - mp.exp(-second * s)) * entry.levy_density(s, parameters)
                ),
                [0, mp.mpf("0.01"), 1, 10, mp.inf],
            )
            # φ(t₂) − φ(t₁) minus the integral: b (t₂ − t₁)
            return (
                entry.bernstein_function(mp.mpf(second), parameters)
                - entry.bernstein_function(mp.mpf(first), parameters)
                - integral
            )

        # Drift b read from each pair of arguments
        first_drift = float(compute_remainder(1, 2)) / 1.0
        second_drift = float(compute_remainder(2, 4)) / 2.0

    assert math.isclose(first_drift, second_drift, rel_tol=0.0, abs_tol=1e-8)
    assert first_drift >= -1e-8


def test_curves_are_classified_by_their_largest_value() -> None:
    """Largest values 0.99, 1 and 1.01 give accepted, borderline and rejected."""
    inside = [False, False]
    assert classify_estimate_curve([0.5, 0.99], inside, tolerance=1e-10) == "accepted"
    assert classify_estimate_curve([0.5, 1.0], inside, tolerance=1e-10) == "borderline"
    assert classify_estimate_curve([0.5, 1.01], inside, tolerance=1e-10) == "rejected"


def test_curve_below_one_with_a_best_point_at_the_window_end_is_not_accepted() -> None:
    """Below 1 but with a best point at an end of the window (Entry 78): borderline.

    Above 1, an end of the window does not matter: the value already rejects the entry.
    """
    at_end = [False, True]
    assert classify_estimate_curve([0.5, 0.99], at_end, tolerance=1e-10) == "borderline"
    assert classify_estimate_curve([0.5, 1.01], at_end, tolerance=1e-10) == "rejected"
