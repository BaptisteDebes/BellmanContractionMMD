"""Tests of the radial kernels against values known in closed form."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from mmd_contraction.kernels import GaussianRBFKernel, TruncatedChainedRBFKernel


def test_gaussian_rbf_kernel_is_one_over_e_at_distance_lengthscale() -> None:
    """k(x, y) = exp(−‖x − y‖² / ℓ²) is 1 at distance 0 and e⁻¹ at distance ℓ."""
    lengthscale = 0.7
    kernel = GaussianRBFKernel(lengthscale=lengthscale)

    values = kernel.evaluate_on_squared_distances(jnp.asarray([0.0, lengthscale**2]))

    np.testing.assert_allclose(np.asarray(values), [1.0, np.exp(-1.0)], rtol=1e-15, atol=0.0)


SQUARED_DISTANCES = [0.0, 1e-6, 0.01, 0.5, 4.0]


def test_truncated_chain_with_one_term_is_one_scaled_rbf_kernel() -> None:
    """With K = 1 the chain is its first term, c (exp(−a t) − 1)."""
    scale, weight = 0.7, 1.3
    chain = TruncatedChainedRBFKernel(
        discount=0.95, target_contraction_factor=0.99275, scale=scale, weight=weight, term_count=1
    )
    squared_distances = np.asarray(SQUARED_DISTANCES)

    values = chain.evaluate_on_squared_distances(jnp.asarray(squared_distances))

    expected = weight * np.expm1(-scale * squared_distances)
    np.testing.assert_allclose(np.asarray(values), expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("target_contraction_factor", [0.99275, 0.911525])
@pytest.mark.parametrize("term_count", [1, 2, 17, 172, 2500])
def test_truncated_chain_turns_into_one_more_term_when_distances_shrink_by_the_discount(
    target_contraction_factor: float, term_count: int
) -> None:
    """Scaling distances by γ shifts the chain by one term: the identity behind Theorem 6.1.

    Term n at squared distance γ² t is τ times term n + 1 at t, so
    k_K(γ² t) = τ (k_{K+1}(t) − c (exp(−a t) − 1)).
    """
    discount, scale, weight = 0.95, 0.7, 1.3

    def build_chain(count: int) -> TruncatedChainedRBFKernel:
        return TruncatedChainedRBFKernel(
            discount=discount,
            target_contraction_factor=target_contraction_factor,
            scale=scale,
            weight=weight,
            term_count=count,
        )

    squared_distances = np.asarray(SQUARED_DISTANCES)

    shrunk = build_chain(term_count).evaluate_on_squared_distances(
        jnp.asarray(discount**2 * squared_distances)
    )
    one_more = build_chain(term_count + 1).evaluate_on_squared_distances(
        jnp.asarray(squared_distances)
    )

    first_term = weight * np.expm1(-scale * squared_distances)
    expected = target_contraction_factor * (np.asarray(one_more) - first_term)
    np.testing.assert_allclose(np.asarray(shrunk), expected, rtol=1e-12, atol=0.0)


def build_chain_for_shares() -> TruncatedChainedRBFKernel:
    """Build the first chain of Table 9 (γ = 0.95, τ = 0.99275, a = c = 1) with N = 200."""
    return TruncatedChainedRBFKernel(
        discount=0.95, target_contraction_factor=0.99275, scale=1.0, weight=1.0, term_count=201
    )


# Squared distances t > 0 at which the shares are checked, from 10⁻⁶ to 4
SHARE_SQUARED_DISTANCES = np.geomspace(1e-6, 4.0, 50)


@pytest.mark.parametrize("truncation_index", [1, 13, 80, 200])
def test_omitted_shares_equal_the_ratio_of_two_truncated_chains(truncation_index: int) -> None:
    """R_K^(N)(t) is the chain of terms K, …, N over the chain of all N + 1 terms.

    Terms K, …, N form a chain of their own, with weight c τ^{−K}, scale a γ^{2K} and N + 1 − K
    terms. The shares, computed at once for every K by summing from the last term down, must
    equal this ratio of two kernel evaluations, which add their terms from the first one up.
    """
    chain = build_chain_for_shares()
    squared_distances = jnp.asarray(SHARE_SQUARED_DISTANCES)

    shares = chain.compute_omitted_shares(squared_distances)

    omitted_terms = replace(
        chain,
        weight=chain.weight * chain.target_contraction_factor ** (-truncation_index),
        scale=chain.scale * chain.discount ** (2 * truncation_index),
        term_count=chain.term_count - truncation_index,
    )
    expected = np.asarray(omitted_terms.evaluate_on_squared_distances(squared_distances)) / (
        np.asarray(chain.evaluate_on_squared_distances(squared_distances))
    )
    np.testing.assert_allclose(np.asarray(shares[truncation_index]), expected, rtol=1e-12, atol=0.0)


def test_omitted_shares_start_at_one_and_never_increase_with_the_truncation_index() -> None:
    """Every term is positive for t > 0, so the omitted sum can only shrink as K grows.

    Both properties hold exactly in floating point: s / s = 1, and adding a positive number
    never makes a sum smaller.
    """
    shares = np.asarray(
        build_chain_for_shares().compute_omitted_shares(jnp.asarray(SHARE_SQUARED_DISTANCES))
    )

    assert shares.shape == (201, SHARE_SQUARED_DISTANCES.size)
    np.testing.assert_array_equal(shares[0], 1.0)
    assert np.all(np.diff(shares, axis=0) <= 0.0)
