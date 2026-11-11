"""Tests of the permutation test against a closed form and against its guaranteed level."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64, PRNGKeyArray

from mmd_contraction.kernels import GaussianRBFKernel, PowerDistanceKernel
from mmd_contraction.permutation_test import estimate_test_power, run_permutation_test


@jax.tree_util.register_static
@dataclass(frozen=True)
class NormalSamplesWithShiftedMean:
    """P = N(0, I_d) and Q = N(s e_1, I_d): P = Q when the shift s is 0."""

    first_sample_size: int
    second_sample_size: int
    dimension: int
    mean_shift: float

    def draw_samples(
        self, key: PRNGKeyArray
    ) -> tuple[Float64[Array, "first dimension"], Float64[Array, "second dimension"]]:
        """Args and returns: see `TwoSampleProblem.draw_samples`."""
        first_key, second_key = jax.random.split(key)
        first = jax.random.normal(
            first_key, (self.first_sample_size, self.dimension), dtype=jnp.float64
        )
        second = jax.random.normal(
            second_key, (self.second_sample_size, self.dimension), dtype=jnp.float64
        )
        shift = jnp.zeros(self.dimension, dtype=jnp.float64).at[0].set(self.mean_shift)
        return first, second + shift


def test_statistic_of_negative_squared_distance_is_twice_the_squared_mean_difference() -> None:
    """For k(x, y) = −‖x − y‖², ℓᵀKℓ = 2‖x̄ − ȳ‖², because ℓ sums to zero."""
    first_key, second_key = jax.random.split(jax.random.key(0))
    first = jax.random.normal(first_key, (5, 2), dtype=jnp.float64)
    second = jax.random.normal(second_key, (7, 2), dtype=jnp.float64) + 0.3

    result = run_permutation_test(
        jax.random.key(1), PowerDistanceKernel(exponent=2.0), first, second, permutation_count=19
    )

    mean_difference = np.asarray(first).mean(axis=0) - np.asarray(second).mean(axis=0)
    expected = 2.0 * float(mean_difference @ mean_difference)
    np.testing.assert_allclose(float(result.statistic), expected, rtol=1e-12, atol=0.0)


def test_p_value_is_one_of_the_values_a_permutation_test_can_take() -> None:
    """p̂ (B_perm + 1) is a whole number from 1 to B_perm + 1, and it counts the T_π ≥ T."""
    permutation_count = 19
    first_key, second_key = jax.random.split(jax.random.key(2))
    first = jax.random.normal(first_key, (6, 1), dtype=jnp.float64)
    second = jax.random.normal(second_key, (6, 1), dtype=jnp.float64)

    result = run_permutation_test(
        jax.random.key(3),
        GaussianRBFKernel(lengthscale=1.0),
        first,
        second,
        permutation_count=permutation_count,
    )

    assert result.null_statistics.shape == (permutation_count,)
    scaled = float(result.p_value) * (permutation_count + 1)
    assert abs(scaled - round(scaled)) < 1e-9
    exceed_count = int(np.sum(np.asarray(result.null_statistics) >= float(result.statistic)))
    assert round(scaled) == 1 + exceed_count


def test_power_is_zero_when_the_kernel_is_zero_between_distinct_points() -> None:
    """With ℓ = 10⁻¹², k(x, y) is exactly 0 in float64 for x ≠ y, although P ≠ Q.

    The Gram matrix is then the identity, every shuffle gives exactly the statistic of the
    data, T_π ≥ T for all shuffles, p̂ = 1 and the test never rejects: the power is exactly
    0, for equal and unequal sample sizes.
    """
    kernel = GaussianRBFKernel(lengthscale=1e-12)

    for first_size, second_size in [(100, 100), (70, 70), (60, 140)]:
        power = estimate_test_power(
            jax.random.key(5),
            kernel,
            NormalSamplesWithShiftedMean(first_size, second_size, 2, 1.0),
            dataset_count=20,
            permutation_count=99,
            level=0.05,
            datasets_per_batch=10,
        )

        assert float(power) == 0.0, (first_size, second_size)


def test_rejection_rate_is_the_level_when_both_samples_come_from_one_distribution() -> None:
    """With P = Q, the permutation test rejects with probability α (here exactly 5/100).

    With B_perm = 99, p̂ ≤ 0.05 means at most 4 of the 99 shuffles reach T, which happens
    with probability 5/100 when the labels are exchangeable. Over 400 datasets the rejection
    rate has standard deviation √(0.05 · 0.95 / 400) ≈ 0.011; the test allows three of them.
    """
    problem = NormalSamplesWithShiftedMean(15, 20, 1, 0.0)

    rejection_rate = estimate_test_power(
        jax.random.key(4),
        GaussianRBFKernel(lengthscale=1.0),
        problem,
        dataset_count=400,
        permutation_count=99,
        level=0.05,
        datasets_per_batch=20,
    )

    assert abs(float(rejection_rate) - 0.05) <= 3.0 * np.sqrt(0.05 * 0.95 / 400)
