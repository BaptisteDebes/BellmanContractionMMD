"""Tests of the task samplers: the ball, the moments, the weights, the corners and the level."""

import jax
import numpy as np

from mmd_contraction.kernels import PowerDistanceKernel
from mmd_contraction.permutation_test import estimate_test_power
from mmd_contraction.tradeoff_tasks import (
    CHECKERBOARD,
    GaussianMeanShiftPair,
    GaussianMixture,
    GaussianMixturePairOnBall,
    ParityCheckerboardPair,
)


def test_points_end_in_the_ball_and_points_inside_it_do_not_move() -> None:
    """With R = 3, every point lies in B(0, 3); those already inside are drawn unchanged.

    The same key with a radius large enough to move nothing gives the unmoved points.
    """
    unmoved_task = GaussianMixturePairOnBall(
        CHECKERBOARD.first_mixture, CHECKERBOARD.second_mixture, 200, 200, radius=1e6
    )
    task = GaussianMixturePairOnBall(
        CHECKERBOARD.first_mixture, CHECKERBOARD.second_mixture, 200, 200, radius=3.0
    )
    key = jax.random.key(0)

    for unmoved, moved in zip(unmoved_task.draw_samples(key), task.draw_samples(key), strict=True):
        unmoved, moved = np.asarray(unmoved), np.asarray(moved)
        unmoved_norms = np.linalg.norm(unmoved, axis=1)
        moved_norms = np.linalg.norm(moved, axis=1)
        inside = unmoved_norms <= 3.0
        assert 0 < inside.sum() < inside.size  # both cases occur
        np.testing.assert_array_equal(moved[inside], unmoved[inside])
        np.testing.assert_allclose(moved_norms[~inside], 3.0, rtol=1e-12)
        # moved along the same direction: unit vectors agree
        np.testing.assert_allclose(
            moved[~inside] / moved_norms[~inside, None],
            unmoved[~inside] / unmoved_norms[~inside, None],
            rtol=0.0,
            atol=1e-12,
        )


def test_one_gaussian_has_its_mean_and_covariance() -> None:
    """20,000 draws of N(μ, Σ) have sample mean and covariance within sampling error.

    Standard errors are about 0.004 for the mean and 0.003 for the covariance entries; the
    tolerances are four times larger.
    """
    mean = (0.3, -0.2)
    covariance = ((0.28, 0.18), (0.18, 0.28))
    gaussian = GaussianMixture(weights=(1.0,), means=(mean,), covariances=(covariance,))

    points = np.asarray(gaussian.draw_points(jax.random.key(1), 20_000))

    np.testing.assert_allclose(points.mean(axis=0), mean, rtol=0.0, atol=0.016)
    np.testing.assert_allclose(np.cov(points.T), covariance, rtol=0.0, atol=0.012)


def test_mixture_components_are_drawn_with_their_weights() -> None:
    """With components far apart and weights ¼ and ¾, about ¾ of 20,000 points are positive.

    The standard error of that fraction is about 0.003; the tolerance is four times larger.
    """
    mixture = GaussianMixture(
        weights=(0.25, 0.75), means=((-10.0,), (10.0,)), covariances=(((1.0,),), ((1.0,),))
    )

    points = np.asarray(mixture.draw_points(jax.random.key(2), 20_000))

    assert abs(float(np.mean(points[:, 0] > 0.0)) - 0.75) <= 0.012


def test_checkerboard_points_are_corners_whose_signs_multiply_to_the_class_sign() -> None:
    """In d = 5 with R = 2, every coordinate is ±R/√d; P's signs multiply to +1, Q's to −1."""
    problem = ParityCheckerboardPair(
        dimension=5, first_sample_size=300, second_sample_size=400, radius=2.0
    )

    first, second = (np.asarray(sample) for sample in problem.draw_samples(jax.random.key(3)))

    assert first.shape == (300, 5) and second.shape == (400, 5)
    for sample, sign_product in ((first, 1.0), (second, -1.0)):
        np.testing.assert_allclose(np.abs(sample), 2.0 / np.sqrt(5.0), rtol=1e-15, atol=0.0)
        np.testing.assert_array_equal(np.prod(np.sign(sample), axis=1), sign_product)


def test_checkerboard_corners_are_equally_likely_within_each_class() -> None:
    """In d = 3, each class has 4 corners; each gets about ¼ of 20,000 points.

    The standard error of each fraction is about 0.003; the tolerance is four times larger.
    """
    problem = ParityCheckerboardPair(
        dimension=3, first_sample_size=20_000, second_sample_size=20_000, radius=1.0
    )

    for sample in problem.draw_samples(jax.random.key(4)):
        _, counts = np.unique(np.sign(np.asarray(sample)), axis=0, return_counts=True)
        assert counts.size == 4
        np.testing.assert_allclose(counts / 20_000, 0.25, rtol=0.0, atol=0.012)


def test_mean_shift_samples_have_their_means_and_identity_covariance() -> None:
    """In d = 8 with s = 0.8, 20,000 draws per sample have mean ∓(s/2) u and covariance I.

    Standard errors are about 0.007 for the means and at most 0.01 for the covariance
    entries; the tolerances are four times larger.
    """
    problem = GaussianMeanShiftPair(
        dimension=8, mean_separation=0.8, first_sample_size=20_000, second_sample_size=20_000
    )

    first, second = (np.asarray(sample) for sample in problem.draw_samples(jax.random.key(5)))

    half_shift = 0.4 / np.sqrt(8.0)  # each coordinate of (s/2) u
    np.testing.assert_allclose(first.mean(axis=0), -half_shift, rtol=0.0, atol=0.028)
    np.testing.assert_allclose(second.mean(axis=0), half_shift, rtol=0.0, atol=0.028)
    for sample in (first, second):
        np.testing.assert_allclose(np.cov(sample.T), np.eye(8), rtol=0.0, atol=0.04)


def test_rejection_rate_is_the_level_when_the_means_coincide() -> None:
    """With s = 0, P = Q and the permutation test rejects with probability α = 5/100.

    Over 400 datasets with B_perm = 99, the rejection rate has standard deviation about
    0.011; the test allows three of them.
    """
    problem = GaussianMeanShiftPair(
        dimension=4, mean_separation=0.0, first_sample_size=15, second_sample_size=20
    )

    rejection_rate = estimate_test_power(
        jax.random.key(6),
        PowerDistanceKernel(exponent=1.0),
        problem,
        dataset_count=400,
        permutation_count=99,
        level=0.05,
        datasets_per_batch=20,
    )

    assert abs(float(rejection_rate) - 0.05) <= 3.0 * np.sqrt(0.05 * 0.95 / 400)
