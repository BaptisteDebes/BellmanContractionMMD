"""Tests of the finite-support contraction threshold against identities known in closed form."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mmd_contraction.finite_support import (
    build_zero_sum_basis,
    compute_contraction_threshold,
    draw_contraction_thresholds,
)
from mmd_contraction.kernels import PowerDistanceKernel


@pytest.mark.parametrize("support_size", [2, 3, 17])
def test_zero_sum_basis_is_orthonormal_and_has_zero_sums(support_size: int) -> None:
    """Q's columns are orthonormal (QᵀQ = I) and each sums to zero (1ᵀQ = 0)."""
    basis = np.asarray(build_zero_sum_basis(support_size))

    assert basis.shape == (support_size, support_size - 1)
    np.testing.assert_allclose(basis.T @ basis, np.eye(support_size - 1), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(basis.sum(axis=0), 0.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("discount", [0.3, 0.7, 0.95])
@pytest.mark.parametrize("dimension", [1, 2])
def test_power_distance_threshold_is_discount_to_the_exponent(
    exponent: float, discount: float, dimension: int
) -> None:
    """For k(x, y) = −‖x − y‖^α, K_γ = γ^α K on every support, so τ⋆_S(γ) = γ^α."""
    support_size = 8
    support = jax.random.normal(jax.random.key(0), (support_size, dimension), dtype=jnp.float64)
    kernel = PowerDistanceKernel(exponent=exponent)

    threshold = compute_contraction_threshold(
        support,
        kernel,
        jnp.asarray(discount),
        build_zero_sum_basis(support_size),
        relative_eigenvalue_tolerance=1e-9,
    )

    assert abs(float(threshold) - discount**exponent) <= 1e-10


def test_drawn_power_distance_thresholds_are_discount_to_the_exponent() -> None:
    """Every drawn support gives γ^α, with one row per discount and one column per support.

    Batches of 2 do not divide the 3 supports or the 5 discounts, so the last, shorter
    batch is covered too.
    """
    exponent = 1.5
    discounts = jnp.asarray([0.1, 0.4, 0.6, 0.9, 0.99])
    kernel = PowerDistanceKernel(exponent=exponent)

    thresholds = draw_contraction_thresholds(
        jax.random.key(321),
        kernel,
        discounts,
        radius=2.0,
        dimension=2,
        support_size=6,
        support_count=3,
        relative_eigenvalue_tolerance=1e-9,
        supports_per_batch=2,
        discounts_per_batch=2,
        direction_norm_floor=1e-12,
    )

    assert thresholds.shape == (5, 3)
    expected = np.broadcast_to(np.asarray(discounts)[:, None] ** exponent, (5, 3))
    np.testing.assert_allclose(np.asarray(thresholds), expected, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("dimension", [1, 2])
def test_drawn_power_distance_thresholds_are_exact_at_a_small_exponent(dimension: int) -> None:
    """At α = 0.2 on supports of 512 points, every drawn support still gives γ^α.

    A small exponent turns a tiny error in a squared distance into a large one, for example
    (10⁻³²)^0.1 ≈ 6·10⁻⁴, so a point's distance to itself must come out exactly 0 even though
    the points are drawn and their distances computed in one compiled program.
    """
    exponent = 0.2
    discounts = jnp.asarray([0.01, 0.5, 1.0])
    kernel = PowerDistanceKernel(exponent=exponent)

    thresholds = draw_contraction_thresholds(
        jax.random.key(20260550),
        kernel,
        discounts,
        radius=1.4,
        dimension=dimension,
        support_size=512,
        support_count=2,
        relative_eigenvalue_tolerance=1e-9,
        supports_per_batch=2,
        discounts_per_batch=3,
        direction_norm_floor=1e-12,
    )

    expected = np.broadcast_to(np.asarray(discounts)[:, None] ** exponent, (3, 2))
    np.testing.assert_allclose(np.asarray(thresholds), expected, rtol=0.0, atol=1e-9)
