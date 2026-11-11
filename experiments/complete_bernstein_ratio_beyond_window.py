"""Evaluate the ratio R_λ of Entry 78 beyond the search window (Appendix G.1.1).

At every discount, the search finds Entry 78's best point at the left end of its window,
u = −80, where its estimate stays below 1. Evaluating R_λ(e^u) = m(e^u/λ²) / (λ² m(e^u))
further left shows where the ratio goes as s = e^u → 0⁺.

Usage:
    python experiments/complete_bernstein_ratio_beyond_window.py --output <folder>

Running time: a few seconds.
"""

import argparse
from pathlib import Path

import numpy as np
from complete_bernstein_entries import SEARCH_SETTINGS

from mmd_contraction import complete_bernstein

SCHILLING_NUMBER = 78  # φ(t) = arcosh²(1 + t)/2, m(s) = e^(−s) K₀(s)/s, no parameter
DISCOUNTS = (0.5, 0.99)  # λ
LOG_ARGUMENTS = (-80.0, -800.0, -8000.0)  # u = log s: the left end of the window, then beyond
ARCHIVE_NAME = "complete_bernstein_ratio_beyond_window.npz"


def main() -> None:
    """Evaluate R_λ(e^u) at each discount and log argument, print the table and save it."""
    parser = argparse.ArgumentParser(description="Evaluate Entry 78's ratio beyond the window.")
    parser.add_argument("--output", type=Path, required=True, help="folder for the results")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    entry = complete_bernstein.ENTRIES[SCHILLING_NUMBER]
    # R_λ(e^u) for each discount and log argument, shape (discounts, log arguments)
    ratios = np.asarray(
        [
            [
                complete_bernstein.compute_ratio_at_log_argument(
                    entry,
                    entry.default_parameters,
                    discount,
                    log_argument,
                    SEARCH_SETTINGS.decimal_digits,
                )
                for log_argument in LOG_ARGUMENTS
            ]
            for discount in DISCOUNTS
        ]
    )
    for discount, row in zip(DISCOUNTS, ratios, strict=True):
        print(
            f"λ = {discount}: "
            + ", ".join(
                f"R = {ratio:.10f} at u = {u:g}"
                for u, ratio in zip(LOG_ARGUMENTS, row, strict=True)
            )
        )
    np.savez(
        arguments.output / ARCHIVE_NAME,
        allow_pickle=False,
        discounts=np.asarray(DISCOUNTS),
        log_arguments=np.asarray(LOG_ARGUMENTS),
        ratios=ratios,
    )


if __name__ == "__main__":
    main()
