from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, TypeVar, cast

import numpy as np
from PIL import Image

from morphology import binary_close

T = TypeVar("T")
R = TypeVar("R")

BLUE_DOM_PIX_THRESH = 0.08
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Component:
    bbox: tuple[int, int, int, int]
    pixels: int
    solidity: float


def load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return img, arr


def initial_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    blue_dom = b - (0.45 * r + 0.55 * g)
    mask = (b > 0.42) & (blue_dom > BLUE_DOM_PIX_THRESH) & (b > g)
    return binary_close(mask, iterations=1)


def component_stats(mask: np.ndarray) -> List[Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Component] = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            pixel_count = 0

            while stack:
                cy, cx = stack.pop()
                pixel_count += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)

                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if pixel_count == 0:
                continue
            bbox = (min_x, min_y, max_x, max_y)
            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            comps.append(Component(bbox=bbox, pixels=pixel_count, solidity=pixel_count / bbox_area))
    return comps


def is_white_cell(comp: Component, image_area: int) -> bool:
    min_pixels = max(600, int(image_area * 0.003))
    return comp.pixels >= min_pixels and comp.solidity >= 0.42


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
