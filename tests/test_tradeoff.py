"""Test of the tradeoff heatmap on a case worked out by hand."""

import jax.numpy as jnp
import numpy as np

from mmd_contraction.tradeoff import build_tradeoff_heatmap


def test_heatmap_keeps_the_best_power_among_the_parameters_within_budget() -> None:
    """Three parameters, two discounts, three budgets, worked out by hand.

    Thresholds (rows: parameters 0, 1, 2; columns: discounts 0, 1) and powers:
        parameter 0: 0.2, 0.9   power 0.1
        parameter 1: 0.5, 0.4   power 0.6
        parameter 2: 0.8, 0.95  power 0.9
    Budget 0.1: nothing fits, NaN. Budget 0.5: parameters 0 and 1 fit at discount 0 (0.5
    counts, as τ̂ ≤ τ̄), only 1 at discount 1: 0.6 and 0.6. Budget 0.9: all fit at discount 0
    (best 0.9); at discount 1, parameter 2 (0.95) does not, so 0.6.
    """
    thresholds = jnp.asarray([[0.2, 0.9], [0.5, 0.4], [0.8, 0.95]])
    powers = jnp.asarray([0.1, 0.6, 0.9])
    budgets = jnp.asarray([0.1, 0.5, 0.9])

    heatmap = build_tradeoff_heatmap(thresholds, powers, budgets)

    expected = np.asarray([[np.nan, np.nan], [0.6, 0.6], [0.9, 0.6]])
    np.testing.assert_array_equal(np.asarray(heatmap), expected)
