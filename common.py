from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Sequence, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    *,
    workers: int = 12,
    mode: str = "process",
) -> list[R]:
    if not tasks:
        return []
    executor_cls = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
    results: list[R | None] = [None] * len(tasks)
    with executor_cls(max_workers=workers) as executor:
        futures = {executor.submit(worker, task): idx for idx, task in enumerate(tasks)}
        completed = 0
        total = len(tasks)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            print(f"{completed}/{total}", flush=True)
    return [cast(R, r) for r in results]
