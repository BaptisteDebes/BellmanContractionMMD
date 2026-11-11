"""The two-sample tasks used to study the tradeoff between power and contraction.

The tasks of Table 12 draw their two samples from Gaussian mixtures P and Q, then move every
point that falls outside the ball B(0, R) radially onto its boundary. Their sample sizes are
those of the experiments behind Figures 3 and 17. Two more pairs exist in any dimension d: the
parity checkerboard on the corners of a cube and a mean shift between two Gaussians.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64, PRNGKeyArray


@jax.tree_util.register_static
@dataclass(frozen=True)
class GaussianMixture:
    """The law Σ_c w_c N(μ_c, Σ_c) on R^d."""

    weights: tuple[float, ...]
    """w_c, one per component, summing to 1."""
    means: tuple[tuple[float, ...], ...]
    """μ_c, one per component, each of length d."""
    covariances: tuple[tuple[tuple[float, ...], ...], ...]
    """Σ_c, one d × d matrix per component."""

    def draw_points(
        self, key: PRNGKeyArray, point_count: int
    ) -> Float64[Array, "points dimension"]:
        """Draw independent points from the mixture.

        Each point picks its component c with probability w_c, then is drawn from N(μ_c, Σ_c).

        Args:
            key: Random key for the components and the Gaussian draws.
            point_count: Number of points.

        Returns:
            The points: shape (point_count, d).
        """
        # Key for the components and key for the Gaussian draws
        component_key, normal_key = jax.random.split(key)
        # Component weights w_c, shape (C,)
        weights = jnp.asarray(self.weights, dtype=jnp.float64)
        # Component means μ_c, shape (C, d)
        means = jnp.asarray(self.means, dtype=jnp.float64)
        # Cholesky factors L_c with L_c L_cᵀ = Σ_c, computed while compiling, shape (C, d, d)
        cholesky_factors = jnp.asarray(
            np.linalg.cholesky(np.asarray(self.covariances, dtype=np.float64))
        )
        # Component c of each point, drawn with probabilities w_c, shape (point_count,)
        components = jax.random.choice(component_key, len(self.weights), (point_count,), p=weights)
        # Standard Gaussian draws ε ~ N(0, I_d), shape (point_count, d)
        standard_normals = jax.random.normal(
            normal_key, (point_count, means.shape[1]), dtype=jnp.float64
        )
        # Points μ_c + L_c ε, with c the component of each point, shape (point_count, d)
        return means[components] + jnp.einsum(
            "pij,pj->pi", cholesky_factors[components], standard_normals
        )


def _move_onto_ball(
    points: Float64[Array, "points dimension"], radius: float
) -> Float64[Array, "points dimension"]:
    """Move each point outside B(0, R) radially onto its boundary; leave the others unchanged.

    Args:
        points: The points: shape (n, d).
        radius: R.

    Returns:
        x R / max(‖x‖, R) for each point x: shape (n, d).
    """
    # Norms ‖x‖ of the points, shape (n, 1)
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    # Points moved onto the ball: x R / max(‖x‖, R), shape (n, d)
    return points * (radius / jnp.maximum(norms, radius))


@jax.tree_util.register_static
@dataclass(frozen=True)
class GaussianMixturePairOnBall:
    """A task of Table 12: samples from P and Q, each point moved onto B(0, R) if outside it."""

    first_mixture: GaussianMixture
    """P."""
    second_mixture: GaussianMixture
    """Q."""
    first_sample_size: int
    """n, the number of points drawn from P."""
    second_sample_size: int
    """m, the number of points drawn from Q."""
    radius: float
    """R."""

    def draw_samples(
        self, key: PRNGKeyArray
    ) -> tuple[Float64[Array, "first dimension"], Float64[Array, "second dimension"]]:
        """Args and returns: see `TwoSampleProblem.draw_samples`.

        Both samples are drawn from their mixtures, then moved onto the ball.
        """
        # Key for each sample
        first_key, second_key = jax.random.split(key)
        # Points drawn from P, shape (n, d)
        first = self.first_mixture.draw_points(first_key, self.first_sample_size)
        # Points drawn from Q, shape (m, d)
        second = self.second_mixture.draw_points(second_key, self.second_sample_size)
        # Both samples, each point outside B(0, R) moved onto its boundary
        return _move_onto_ball(first, self.radius), _move_onto_ball(second, self.radius)


@jax.tree_util.register_static
@dataclass(frozen=True)
class ParityCheckerboardPair:
    """P and Q uniform on the corners (R/√d) s, s ∈ {−1, +1}^d, of a cube in R^d.

    P is uniform on the corners with an even number of −1 in s, Q on those with an odd number.
    """

    dimension: int
    """d ≥ 2."""
    first_sample_size: int
    """n, the number of points drawn from P."""
    second_sample_size: int
    """m, the number of points drawn from Q."""
    radius: float
    """R, the distance of every corner from the origin."""

    def draw_samples(
        self, key: PRNGKeyArray
    ) -> tuple[Float64[Array, "first dimension"], Float64[Array, "second dimension"]]:
        """Args and returns: see `TwoSampleProblem.draw_samples`."""
        # Key for each sample
        first_key, second_key = jax.random.split(key)
        # Corners from P, whose signs multiply to +1, and from Q, whose signs multiply to −1
        return (
            self._draw_corners(first_key, self.first_sample_size, 1.0),
            self._draw_corners(second_key, self.second_sample_size, -1.0),
        )

    def _draw_corners(
        self, key: PRNGKeyArray, point_count: int, sign_product: float
    ) -> Float64[Array, "points dimension"]:
        """Draw corners uniformly among those whose d signs multiply to sign_product.

        The first d − 1 signs are independent and each ±1 with probability ½; the last sign is
        the one that makes the product equal sign_product.

        Args:
            key: Random key for the signs.
            point_count: Number of corners.
            sign_product: +1 for P, −1 for Q.

        Returns:
            The corners: shape (point_count, d).
        """
        # Signs of the first d − 1 coordinates, each ±1 with probability ½,
        # shape (point_count, d − 1)
        leading_signs = jnp.where(
            jax.random.bernoulli(key, 0.5, (point_count, self.dimension - 1)), 1.0, -1.0
        ).astype(jnp.float64)
        # Sign of the last coordinate, making the product of all d signs equal sign_product,
        # shape (point_count, 1)
        last_sign = sign_product * jnp.prod(leading_signs, axis=1, keepdims=True)
        # Corners (R/√d) s, shape (point_count, d)
        return (self.radius / math.sqrt(self.dimension)) * jnp.concatenate(
            [leading_signs, last_sign], axis=1
        )


@jax.tree_util.register_static
@dataclass(frozen=True)
class GaussianMeanShiftPair:
    """P = N(−(s/2) u, I_d) and Q = N((s/2) u, I_d), with u = (1, …, 1)/√d.

    The two means are s apart along the diagonal. The samples are not moved onto a ball.
    """

    dimension: int
    """d ≥ 1."""
    mean_separation: float
    """s, the distance between the two means."""
    first_sample_size: int
    """n, the number of points drawn from P."""
    second_sample_size: int
    """m, the number of points drawn from Q."""

    def draw_samples(
        self, key: PRNGKeyArray
    ) -> tuple[Float64[Array, "first dimension"], Float64[Array, "second dimension"]]:
        """Args and returns: see `TwoSampleProblem.draw_samples`."""
        # Key for each sample
        first_key, second_key = jax.random.split(key)
        # Half shift (s/2) u, shape (d,)
        half_shift = jnp.full(
            (self.dimension,), 0.5 * self.mean_separation / math.sqrt(self.dimension)
        )
        # Points from P: standard Gaussian draws moved by −(s/2) u, shape (n, d)
        first = (
            jax.random.normal(
                first_key, (self.first_sample_size, self.dimension), dtype=jnp.float64
            )
            - half_shift
        )
        # Points from Q: standard Gaussian draws moved by (s/2) u, shape (m, d)
        second = (
            jax.random.normal(
                second_key, (self.second_sample_size, self.dimension), dtype=jnp.float64
            )
            + half_shift
        )
        return first, second


def _build_gaussian(
    mean: tuple[float, ...], covariance: tuple[tuple[float, ...], ...]
) -> GaussianMixture:
    """Build the single Gaussian N(μ, Σ) as a mixture with one component."""
    return GaussianMixture(weights=(1.0,), means=(mean,), covariances=(covariance,))


def _build_isotropic_covariance(
    standard_deviation: float, dimension: int
) -> tuple[tuple[float, ...], ...]:
    """Build σ² I_d."""
    # Variance σ²
    variance = standard_deviation**2
    # Covariance matrix σ² I_d as nested tuples, shape (d, d)
    return tuple(
        tuple(variance if row == column else 0.0 for column in range(dimension))
        for row in range(dimension)
    )


MEAN_SHIFT_1D = GaussianMixturePairOnBall(
    first_mixture=_build_gaussian((-0.40,), _build_isotropic_covariance(0.18, 1)),
    second_mixture=_build_gaussian((0.40,), _build_isotropic_covariance(0.18, 1)),
    first_sample_size=100,
    second_sample_size=100,
    radius=1.40,
)
"""1D Mean Shift: P = N(−0.40, 0.18²), Q = N(0.40, 0.18²), R = 1.40."""

SAME_MEAN_DIFFERENT_STD_1D = GaussianMixturePairOnBall(
    first_mixture=_build_gaussian((0.0,), _build_isotropic_covariance(0.25, 1)),
    second_mixture=_build_gaussian((0.0,), _build_isotropic_covariance(0.55, 1)),
    first_sample_size=100,
    second_sample_size=100,
    radius=2.00,
)
"""1D Same Mean, Different Std: P = N(0, 0.25²), Q = N(0, 0.55²), R = 2.00."""

BIMODAL_VS_UNIMODAL_1D = GaussianMixturePairOnBall(
    first_mixture=GaussianMixture(
        weights=(0.5, 0.5),
        means=((-0.65,), (0.65,)),
        covariances=(_build_isotropic_covariance(0.15, 1),) * 2,
    ),
    second_mixture=_build_gaussian((0.0,), ((0.65**2 + 0.15**2,),)),
    first_sample_size=200,
    second_sample_size=200,
    radius=2.00,
)
"""1D Bimodal vs Unimodal: P = ½N(−0.65, 0.15²) + ½N(0.65, 0.15²), Q = N(0, 0.65² + 0.15²)."""

TWO_GAUSSIAN_LOCATION = GaussianMixturePairOnBall(
    first_mixture=_build_gaussian((0.55, 0.0), _build_isotropic_covariance(0.18, 2)),
    second_mixture=_build_gaussian((-0.55, 0.0), _build_isotropic_covariance(0.18, 2)),
    first_sample_size=70,
    second_sample_size=70,
    radius=1.35,
)
"""Two-Gaussian Location: P = N((0.55, 0), 0.18² I₂), Q = N((−0.55, 0), 0.18² I₂), R = 1.35."""

COVARIANCE_SWAP = GaussianMixturePairOnBall(
    first_mixture=_build_gaussian((0.0, 0.0), ((0.28, 0.18), (0.18, 0.28))),
    second_mixture=_build_gaussian((0.0, 0.0), ((0.28, -0.18), (-0.18, 0.28))),
    first_sample_size=100,
    second_sample_size=100,
    radius=2.40,
)
"""Covariance Swap: P = N(0, [[0.28, 0.18], [0.18, 0.28]]), Q with −0.18 off the diagonal."""

BIMODAL_VS_UNIMODAL = GaussianMixturePairOnBall(
    first_mixture=GaussianMixture(
        weights=(0.5, 0.5),
        means=((-0.75, 0.0), (0.75, 0.0)),
        covariances=(_build_isotropic_covariance(0.18, 2),) * 2,
    ),
    second_mixture=_build_gaussian((0.0, 0.0), ((0.75**2 + 0.18**2, 0.0), (0.0, 0.18**2))),
    first_sample_size=200,
    second_sample_size=200,
    radius=2.40,
)
"""Bimodal vs Unimodal: P = ½N((∓0.75, 0), 0.18² I₂), Q = N(0, diag(0.75² + 0.18², 0.18²))."""

CHECKERBOARD = GaussianMixturePairOnBall(
    first_mixture=GaussianMixture(
        weights=(0.5, 0.5),
        means=((-2.2, -1.7), (2.2, 1.7)),
        covariances=(_build_isotropic_covariance(0.35, 2),) * 2,
    ),
    second_mixture=GaussianMixture(
        weights=(0.5, 0.5),
        means=((-2.0, 1.9), (2.0, -1.9)),
        covariances=(_build_isotropic_covariance(0.35, 2),) * 2,
    ),
    first_sample_size=200,
    second_sample_size=200,
    radius=4.40,
)
"""Checkerboard: P and Q are each ½ + ½ of N(μ, 0.35² I₂) at opposite corners, R = 4.40."""
