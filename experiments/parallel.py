"""Run the independent computations of an experiment in parallel on this machine.

A script lists its computations, each named by a tuple such as (entry, discount index), and
gives the function that runs one. run_computations_in_parallel runs them in separate
processes, each computing with one thread, on its own CPU on Linux, and appends every result
to a file as soon as it is computed, so a stopped run continues where it stopped.
"""

import json
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

type Computation = tuple[int | str, ...]
type ComputationResult = dict[str, Any]  # values that JSON can hold

# Each process computes with one thread, so that process_count processes use process_count
# CPUs. The processes read these variables when their libraries start.
ONE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
}


def count_usable_cpus() -> int:
    """The number of CPUs this process may use: the default number of processes."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def move_process_to_one_cpu(cpu_queue: "multiprocessing.Queue[int]") -> None:
    """Keep the calling process on one CPU, the next one in cpu_queue (Linux only)."""
    os.sched_setaffinity(0, {cpu_queue.get()})


def read_saved_results(results_file: Path) -> dict[Computation, ComputationResult]:
    """Read the results saved by an earlier run, dropping an unfinished last line.

    A run stopped while writing leaves a last line without its line break; that computation
    is then run again.

    Args:
        results_file: The file of saved results, one JSON line {"computation": …, "result": …}.

    Returns:
        Computation → its saved result; empty if the file does not exist.
    """
    if not results_file.exists():
        return {}
    text = results_file.read_text()
    # The complete lines: everything up to the last line break
    complete_text = text[: text.rfind("\n") + 1]
    if complete_text != text:
        results_file.write_text(complete_text)
    saved: dict[Computation, ComputationResult] = {}
    for line in complete_text.splitlines():
        record = json.loads(line)
        saved[tuple(record["computation"])] = record["result"]
    return saved


def run_computations_in_parallel(
    computations: Sequence[Computation],
    run_computation: Callable[[Computation], ComputationResult],
    results_file: Path,
    process_count: int,
) -> dict[Computation, ComputationResult]:
    """Run every computation in parallel on this machine and return all results.

    Each result is appended to results_file, one JSON line per computation, as soon as it is
    computed. Computations already in that file are not run again. Each process uses one
    thread, and on Linux its own CPU, so process_count processes use process_count CPUs.

    Args:
        computations: Every computation of the experiment, each one distinct.
        run_computation: Runs one computation. It must be defined at the top level of a
            module, so that the processes can import it.
        results_file: The file of saved results, created if missing.
        process_count: Number of processes running computations at the same time.

    Returns:
        Computation → its result, for every computation, in the order of computations.

    Raises:
        ValueError: If two computations are equal, or if results_file holds a computation
            that is not in computations.
        RuntimeError: If a process stops before finishing its computation, for example when
            it is killed for lack of memory. The results finished before are kept.
    """
    # The computations as a set, to check membership
    computation_set = set(computations)
    if len(computation_set) != len(computations):
        raise ValueError("the computations are not distinct")
    # Computation → result, for the computations an earlier run finished
    saved = read_saved_results(results_file)
    # Saved computations that are not computations of this experiment
    unknown = [computation for computation in saved if computation not in computation_set]
    if unknown:
        raise ValueError(
            f"{results_file} holds computations of another experiment, e.g. {unknown[0]}"
        )
    # The computations still to run, in the order of computations
    remaining = [computation for computation in computations if computation not in saved]
    if remaining:
        # New processes start from a fresh interpreter ("spawn"), with one thread each
        context = multiprocessing.get_context("spawn")
        # On Linux, each process runs on its own CPU. JAX and LLVM size their thread pools by
        # the number of CPUs a process may use, some as soon as they are loaded, so the
        # processes start on one CPU, then process i moves to the i-th CPU of this process
        pin_processes = hasattr(os, "sched_setaffinity")
        # The CPUs this process may use, restored once the processes exist
        usable_cpus = sorted(os.sched_getaffinity(0)) if pin_processes else []
        # This process's values of the thread variables, restored once the processes exist
        previous_environment = {name: os.environ.get(name) for name in ONE_THREAD_ENVIRONMENT}
        os.environ.update(ONE_THREAD_ENVIRONMENT)
        try:
            if pin_processes:
                os.sched_setaffinity(0, usable_cpus[:1])
                # The CPU of each process, taken in turn as the processes start
                cpu_queue: multiprocessing.Queue[int] = context.Queue()
                for index in range(process_count):
                    cpu_queue.put(usable_cpus[index % len(usable_cpus)])
                executor = ProcessPoolExecutor(
                    max_workers=process_count,
                    mp_context=context,
                    initializer=move_process_to_one_cpu,
                    initargs=(cpu_queue,),
                )
            else:
                executor = ProcessPoolExecutor(max_workers=process_count, mp_context=context)
            # Future → its computation; the processes start while the computations are handed
            # over, and inherit the one-thread variables and the one CPU
            futures: dict[Future[ComputationResult], Computation] = {
                executor.submit(run_computation, computation): computation
                for computation in remaining
            }
        finally:
            if pin_processes:
                os.sched_setaffinity(0, usable_cpus)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        try:
            with results_file.open("a") as handle:
                # Results in the order they finish
                for future in as_completed(futures):
                    computation = futures[future]
                    try:
                        result = future.result()
                    except BrokenProcessPool as error:
                        raise RuntimeError(
                            "a process stopped before finishing its computation, for example "
                            f"killed for lack of memory; the results so far are in {results_file}, "
                            "and running again continues from them"
                        ) from error
                    handle.write(
                        json.dumps({"computation": list(computation), "result": result}) + "\n"
                    )
                    handle.flush()
                    saved[computation] = result
        except BaseException:
            # After a failure, no new computation starts; the running ones finish on their own
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        # Every computation is done: wait until the processes have exited and freed their threads
        executor.shutdown(wait=True)
    return {computation: saved[computation] for computation in computations}
