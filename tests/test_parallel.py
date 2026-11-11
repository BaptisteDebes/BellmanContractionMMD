"""Tests of experiments/parallel.py: every computation is run once, a stopped run continues,
and on Linux each process runs on one CPU."""

import json
import os
from pathlib import Path

import pytest

from experiments.parallel import Computation, ComputationResult, run_computations_in_parallel


def compute_square(computation: Computation) -> ComputationResult:
    """A computation whose result is known: the square of its number."""
    return {"square": int(computation[0]) ** 2}


def stop_process_at_one(computation: Computation) -> ComputationResult:
    """A computation whose process stops without a result when its number is 1."""
    if int(computation[0]) == 1:
        os._exit(1)
    return {"square": int(computation[0]) ** 2}


def count_cpus_of_process(computation: Computation) -> ComputationResult:
    """A computation whose result is the number of CPUs its process may use."""
    return {"cpus": len(os.sched_getaffinity(0))}


def write_saved_lines(results_file: Path, lines: list[str]) -> None:
    """Write the file of an earlier run, line by line, as given."""
    results_file.write_text("".join(lines))


def test_every_computation_is_run_and_saved(tmp_path: Path) -> None:
    """Five computations on two processes: all results returned in order, one line saved each."""
    computations: list[Computation] = [(number,) for number in range(5)]
    results = run_computations_in_parallel(
        computations, compute_square, tmp_path / "results.jsonl", 2
    )
    assert results == {(number,): {"square": number**2} for number in range(5)}
    assert list(results) == computations
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 5


def test_saved_results_are_not_computed_again(tmp_path: Path) -> None:
    """Computations 0 and 1, saved with a value no run gives (−1), keep it; the others run."""
    results_file = tmp_path / "results.jsonl"
    write_saved_lines(
        results_file,
        [
            json.dumps({"computation": [number], "result": {"square": -1}}) + "\n"
            for number in (0, 1)
        ],
    )
    results = run_computations_in_parallel(
        [(n,) for n in range(4)], compute_square, results_file, 2
    )
    assert [results[(n,)]["square"] for n in range(4)] == [-1, -1, 4, 9]


def test_an_unfinished_last_line_is_computed_again(tmp_path: Path) -> None:
    """A run stopped while writing computation 1's line: 0 is kept, 1 is run again."""
    results_file = tmp_path / "results.jsonl"
    complete = json.dumps({"computation": [0], "result": {"square": -1}}) + "\n"
    unfinished = json.dumps({"computation": [1], "result": {"square": -1}})[:10]
    write_saved_lines(results_file, [complete, unfinished])
    results = run_computations_in_parallel([(0,), (1,)], compute_square, results_file, 2)
    assert results == {(0,): {"square": -1}, (1,): {"square": 1}}
    assert all(json.loads(line) for line in results_file.read_text().splitlines())


def test_results_of_another_experiment_are_refused(tmp_path: Path) -> None:
    """A results file that holds a computation outside the list raises instead of being mixed in."""
    results_file = tmp_path / "results.jsonl"
    write_saved_lines(
        results_file, [json.dumps({"computation": [99], "result": {"square": 0}}) + "\n"]
    )
    with pytest.raises(ValueError, match="another experiment"):
        run_computations_in_parallel([(0,), (1,)], compute_square, results_file, 2)


def test_a_stopped_process_raises_instead_of_waiting(tmp_path: Path) -> None:
    """The process of computation 1 stops without a result: the run raises instead of waiting."""
    with pytest.raises(RuntimeError, match="stopped before finishing"):
        run_computations_in_parallel(
            [(0,), (1,), (2,)], stop_process_at_one, tmp_path / "results.jsonl", 2
        )


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="processes are pinned on Linux")
def test_each_process_runs_on_one_cpu(tmp_path: Path) -> None:
    """Four computations on two processes: each process may use exactly one CPU."""
    results = run_computations_in_parallel(
        [(number,) for number in range(4)], count_cpus_of_process, tmp_path / "results.jsonl", 2
    )
    assert all(result["cpus"] == 1 for result in results.values())
