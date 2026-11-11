# Bellman Contraction under MMD

Baptiste Debes, Tinne Tuytelaars (KU Leuven)

This is the official repository for the NeurIPS 2026 paper "Bellman Contraction under MMD: A Unified Framework".

The paper shows that the distributional Bellman operator contracts in Maximum Mean Discrepancy (MMD) exactly when an associated kernel, the Bellman difference kernel, is conditionally positive definite, and turns this condition into criteria on Bernstein functions for radial kernels. This repository contains the package `mmd_contraction`, which implements the paper's numerical tests, and the scripts that compute and draw every figure of the paper.

## Installation

The code runs on the CPU with Python 3.12. With [uv](https://docs.astral.sh/uv/), from the root of the repository:

```bash
uv sync
```

This creates an environment in `.venv` with the exact package versions listed in `uv.lock`, among them JAX 0.8.0, NumPy 2.4.3 and SciPy 1.16.0. Every command below is run with `uv run`, which uses that environment.

## A first run

Figure 4 of the paper checks the numerical test of Appendix D on kernels whose smallest contraction factor is known in closed form. Recomputing and drawing it takes about two CPU-minutes:

```bash
uv run python experiments/bounded_support_known_factors.py --output results
uv run python paper_figures/bounded_support_known_factors_figure.py --archive results/bounded_support_known_factors.npz --output figures
```

The first command writes the estimates to `results/bounded_support_known_factors.npz`. The second draws them in `figures/bounded_support_theoretical_confirmation.pdf`. Recomputed figures can differ from the paper's in the last digits of the estimates, never visibly.

## The package

`mmd_contraction` computes in double precision: call `jax.config.update("jax_enable_x64", True)` before using it.

| Module | What it computes | In the paper |
|---|---|---|
| `kernels` | The radial kernels: RBF, logarithmic, generalized multiquadric, power distance, and the truncated chained RBF kernel | Table 11, Theorem 6.1 |
| `finite_support` | The contraction threshold of a finite set of points, and its estimate on random sets in a ball | Proposition 5.2, Algorithm 1 |
| `complete_bernstein` | The search for the smallest contraction factor of a complete Bernstein function | Theorem E.5, Algorithm 2 |
| `permutation_test` | The permutation test with the MMD statistic, and its power | Algorithm 3 |
| `tradeoff` | The best power a contraction budget allows | Algorithm 4 |
| `tradeoff_tasks` | The pairs of distributions of the tradeoff experiments | Tables 12 and 13 |
| `distances` | Squared distances between all pairs of points | |

## Example

The package computes the contraction threshold of any finite set of points (Proposition 5.2). For the power-distance kernel −‖x − y‖^α, the threshold is γ^α on every set of points, which this example checks:

```python
import jax
import jax.numpy as jnp

from mmd_contraction.finite_support import build_zero_sum_basis, compute_contraction_threshold
from mmd_contraction.kernels import PowerDistanceKernel

jax.config.update("jax_enable_x64", True)

# 100 random points in the square [−1, 1]², and the power-distance kernel −‖x − y‖^1.5
points = jax.random.uniform(jax.random.key(0), (100, 2), minval=-1.0, maxval=1.0)
kernel = PowerDistanceKernel(exponent=1.5)
# The contraction threshold of these points at discount γ = 0.9
threshold = compute_contraction_threshold(
    points, kernel, jnp.asarray(0.9), build_zero_sum_basis(100), relative_eigenvalue_tolerance=1e-9
)
print(threshold, 0.9**1.5)  # 0.8538149682463805 0.8538149682454624
```

## Reproducing the paper's figures

Most figures take two commands: the first computes the results into `results/`, the second draws the figures into `figures/`. A computation uses all CPUs of the machine unless `--processes N` is given, and a stopped computation continues where it stopped when it is started again with the same folder.

| Figures | Compute | Draw | CPU-hours |
|---|---|---|---|
| 2 | `uv run python experiments/bounded_support_heatmaps.py --output results` | `uv run python paper_figures/bounded_support_heatmaps.py --archives results --output figures` | 7 |
| 3, 16, 17 | `uv run python experiments/tradeoff_experiments.py --output results` | `uv run python paper_figures/tradeoff_figures.py --archives results --output figures` | 46 |
| 4 | `uv run python experiments/bounded_support_known_factors.py --output results` | `uv run python paper_figures/bounded_support_known_factors_figure.py --archive results/bounded_support_known_factors.npz --output figures` | 0.03 |
| 6–11 | `uv run python experiments/complete_bernstein_entries.py --output results`, then `uv run python experiments/complete_bernstein_parameter_sweeps.py --output results` and `uv run python experiments/complete_bernstein_parameter_crossings.py --output results` | `uv run python paper_figures/complete_bernstein_figures.py --archives results --output figures` | 36 (estimate) |
| 18 | `uv run python experiments/tradeoff_experiments_higher_dimensions.py --output results` | `uv run python paper_figures/dimension_tradeoff_figure.py --archive results/dimension_tradeoff.npz --output figures` | 11 |

The computations of Figures 6–11 also give the Num. column of Tables 3–6 and the parameter limits of Table 8.

Figures 1, 5 and 12–15 are drawn from formulas, with no computation step, in a few seconds each: `bernstein_function_examples.py`, `single_chain_figure.py`, `chained_rbf_kernel_figures.py` and `omitted_terms_share.py` in `paper_figures/`, each with `--output figures`. The statement on Entry 78 in Appendix G.1.1 comes from `uv run python experiments/complete_bernstein_ratio_beyond_window.py --output results`, which takes a few seconds. Table 9 comes from `uv run python experiments/truncated_chain_contraction.py --output results`, which prints it, in about 0.2 CPU-hours.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{debes2026bellman,
  title     = {Bellman Contraction under {MMD}: A Unified Framework},
  author    = {Debes, Baptiste and Tuytelaars, Tinne},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the MIT License; see [LICENSE](LICENSE).
