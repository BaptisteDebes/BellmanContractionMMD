"""Radial kernels of the paper, each evaluated on squared distances ‖x − y‖².

Each kernel is a small record of plain-number parameters. The parameters are fixed when a
function that receives the kernel is compiled, so each new parameter value compiles anew.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64, Int

from mmd_contraction.precision import raise_unless_float64_enabled


@runtime_checkable
class RadialKernel(Protocol):
    """A kernel k(x, y) that depends on x and y only through ‖x − y‖²."""

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Evaluate the kernel on squared distances.

        Args:
            squared_distances: Values ‖x − y‖² ≥ 0, any shape.

        Returns:
            k(x, y) for each entry: same shape.
        """
        ...


@jax.tree_util.register_static
@dataclass(frozen=True)
class GaussianRBFKernel:
    """Gaussian RBF kernel k(x, y) = exp(−‖x − y‖² / ℓ²), with lengthscale ℓ > 0 (Theorem 6.1)."""

    lengthscale: float

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Args and returns: see `RadialKernel.evaluate_on_squared_distances`."""
        # Lengthscale ℓ
        lengthscale = jnp.asarray(self.lengthscale, dtype=jnp.float64)
        # Kernel values exp(−‖x − y‖²/ℓ²), same shape as the squared distances
        return jnp.exp(-squared_distances / (lengthscale * lengthscale))


@jax.tree_util.register_static
@dataclass(frozen=True)
class LogarithmicKernel:
    """Logarithmic kernel k(x, y) = −log(1 + ‖x − y‖² / a), with scale a > 0."""

    scale: float

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Args and returns: see `RadialKernel.evaluate_on_squared_distances`."""
        # Scale a
        scale = jnp.asarray(self.scale, dtype=jnp.float64)
        # Kernel values −log(1 + ‖x − y‖²/a), same shape as the squared distances
        return -jnp.log1p(squared_distances / scale)


@jax.tree_util.register_static
@dataclass(frozen=True)
class GeneralizedMultiquadricKernel:
    """Generalized multiquadric kernel k(x, y) = −(1 + h²‖x − y‖²)^β.

    The scale is h > 0 and the exponent is β ∈ (0, 1].
    """

    scale: float
    exponent: float

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Args and returns: see `RadialKernel.evaluate_on_squared_distances`."""
        # Scale h
        scale = jnp.asarray(self.scale, dtype=jnp.float64)
        # Exponent β
        exponent = jnp.asarray(self.exponent, dtype=jnp.float64)
        # Kernel values −(1 + h²‖x − y‖²)^β, same shape as the squared distances
        return -jnp.power(1.0 + (scale * scale) * squared_distances, exponent)


@jax.tree_util.register_static
@dataclass(frozen=True)
class PowerDistanceKernel:
    """Power-distance kernel k(x, y) = −‖x − y‖^α, with exponent α ∈ (0, 2]."""

    exponent: float

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Args and returns: see `RadialKernel.evaluate_on_squared_distances`."""
        # Half exponent α/2, since ‖x − y‖^α = (‖x − y‖²)^(α/2)
        half_exponent = jnp.asarray(0.5 * self.exponent, dtype=jnp.float64)
        # Kernel values −‖x − y‖^α, same shape as the squared distances
        return -jnp.power(squared_distances, half_exponent)


@jax.tree_util.register_static
@dataclass(frozen=True)
class TruncatedChainedRBFKernel:
    """First K terms of the chained RBF kernel of Theorem 6.1.

    k(x, y) = Σ_{n=0}^{K−1} c τ^{−n} (exp(−a γ^{2n} ‖x − y‖²) − 1),

    the sum of Theorem 6.1 with ℓ_n = a^{−1/2} γ^{−n}, stopped after K terms. Its Bernstein
    function is the φ of Appendix G.2 with s₀ = a and ρ = γ², truncated at K.
    """

    discount: float
    """γ ∈ (0, 1)."""
    target_contraction_factor: float
    """τ ∈ (γ², 1)."""
    scale: float
    """a > 0."""
    weight: float
    """c > 0."""
    term_count: int
    """K ≥ 1, the truncation index of Appendix G.2."""

    def evaluate_on_squared_distances(
        self, squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Args and returns: see `RadialKernel.evaluate_on_squared_distances`.

        The K terms are added one at a time in a compiled loop, so memory stays that of one
        array of squared distances, whatever K.
        """

        def add_term(
            term_index: Int[Array, ""], kernel_sum: Float64[Array, "*shape"]
        ) -> Float64[Array, "*shape"]:
            """Add term n, c τ^{−n} (exp(−a γ^{2n} t) − 1).

            Args:
                term_index: n.
                kernel_sum: The sum of terms 0, …, n − 1: same shape as the distances.

            Returns:
                The sum of terms 0, …, n: same shape.
            """
            # Sum of terms 0, …, n, same shape as the squared distances
            return kernel_sum + self.evaluate_term_on_squared_distances(
                term_index, squared_distances
            )

        # Kernel values: the K terms added one at a time, same shape as the squared distances
        return jax.lax.fori_loop(0, self.term_count, add_term, jnp.zeros_like(squared_distances))

    def evaluate_term_on_squared_distances(
        self, term_index: Int[Array, ""], squared_distances: Float64[Array, "*shape"]
    ) -> Float64[Array, "*shape"]:
        """Evaluate term n of the chain, c τ^{−n} (exp(−a γ^{2n} t) − 1), on squared distances.

        The term is −φ_n(t), minus term n of the Bernstein function φ of Appendix G.2. It is
        computed with expm1, which stays accurate when a γ^{2n} t is small.

        Args:
            term_index: n, from 0 to K − 1.
            squared_distances: Values t = ‖x − y‖² ≥ 0, any shape.

        Returns:
            Term n at each squared distance: same shape.
        """
        # Squared discount γ²
        squared_discount = jnp.asarray(self.discount**2, dtype=jnp.float64)
        # Target contraction factor τ
        target = jnp.asarray(self.target_contraction_factor, dtype=jnp.float64)
        # Scale a
        scale = jnp.asarray(self.scale, dtype=jnp.float64)
        # Weight c
        weight = jnp.asarray(self.weight, dtype=jnp.float64)
        # Rate of term n, a γ^{2n}, shape ()
        rate = scale * jnp.power(squared_discount, term_index)
        # Weight of term n, c τ^{−n}, shape ()
        term_weight = weight * jnp.power(target, -term_index)
        # Term n at each squared distance, same shape as the squared distances
        return term_weight * jnp.expm1(-rate * squared_distances)

    def compute_omitted_shares(
        self, squared_distances: Float64[Array, " distances"]
    ) -> Float64[Array, "indices distances"]:
        """Compute the share R_K^(N)(t) of the omitted terms, for every truncation index K.

        With the N + 1 = term_count terms φ_n(t) = c τ^{−n} (1 − exp(−a γ^{2n} t)),
        n = 0, …, N, the share of Appendix G.2 is

            R_K^(N)(t) = Σ_{K ≤ n ≤ N} φ_n(t) / Σ_{0 ≤ n ≤ N} φ_n(t),

        the part of the kernel's profile that truncating at K would leave out.

        Args:
            squared_distances: Values t = ‖x − y‖² > 0; at t = 0 every term is 0: shape (T,).

        Returns:
            R_K^(N)(t) at row K and column t: shape (N + 1, T).

        Raises:
            RuntimeError: If float64 is off; τ^{−n} would overflow.
        """
        raise_unless_float64_enabled()
        # Term indices n = 0, …, N, shape (N + 1,)
        term_indices: Int[Array, " indices"] = jnp.arange(self.term_count)
        # Terms −φ_n(t), one row per n, one column per t, shape (N + 1, T)
        terms: Float64[Array, "indices distances"] = jax.vmap(
            self.evaluate_term_on_squared_distances, in_axes=(0, None)
        )(term_indices, squared_distances)
        # Omitted sums −Σ_{K ≤ n ≤ N} φ_n(t), added from n = N down, one row per K,
        # shape (N + 1, T)
        omitted_sums: Float64[Array, "indices distances"] = jnp.flip(
            jnp.cumsum(jnp.flip(terms, axis=0), axis=0), axis=0
        )
        # Shares R_K^(N)(t): each omitted sum over the sum of all terms (row K = 0); the minus
        # signs cancel, shape (N + 1, T)
        return omitted_sums / omitted_sums[0]
