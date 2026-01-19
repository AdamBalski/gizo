#!/usr/bin/env python3
"""Create a bounding-box database for dataset2 microscope images."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from common import run_tasks
from morphology import binary_close

_BLUE_DOM_PIX_THRESH = 0.08
_VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Component:
    bbox: tuple[int, int, int, int]
    pixels: int
    solidity: float


@dataclass(frozen=True)
class ImageTask:
    label: str
    path: str
    image_name: str


def _load_image(path: Path) -> Tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return img, arr


def _initial_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    blue_dom = b - (0.45 * r + 0.55 * g)
    mask = (b > 0.42) & (blue_dom > _BLUE_DOM_PIX_THRESH) & (b > g)
    return binary_close(mask, iterations=1)


def _component_stats(mask: np.ndarray) -> List[Component]:
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

                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy

                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            bbox = (min_x, min_y, max_x, max_y)
            bbox_area = (max_x - min_x + 1) * (max_y - min_y + 1)
            if pixel_count == 0:
                continue
            comps.append(
                Component(
                    bbox=bbox,
                    pixels=pixel_count,
                    solidity=pixel_count / bbox_area,
                )
            )
    return comps


def _is_white_cell(comp: Component, image_area: int) -> bool:
    min_pixels = max(600, int(image_area * 0.003))
    if comp.pixels < min_pixels:
        return False
    if comp.solidity < 0.42:
        return False
    return True


def _get_bounding_boxes(image_path: Path) -> List[tuple[int, int, int, int]]:
    _img, arr = _load_image(image_path)
    mask = _initial_mask(arr)
    comps = _component_stats(mask)
    image_area = arr.shape[0] * arr.shape[1]
    return [comp.bbox for comp in comps if _is_white_cell(comp, image_area)]


def _process_task(task: ImageTask) -> List[tuple[int, int, int, int]]:
    return _get_bounding_boxes(Path(task.path))


def _iter_image_files(images_root: Path, splits: Sequence[str]) -> Iterable[tuple[str, Path]]:
    for split in splits:
        split_dir = images_root / split
        if not split_dir.exists():
            print(f"[WARN] Split {split_dir} not found", file=sys.stderr)
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.upper()
            for path in sorted(label_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in _VALID_SUFFIXES:
                    continue
                yield label, path


def _build_tasks(images_root: Path, splits: Sequence[str], repo_root: Path) -> List[ImageTask]:
    repo_root = repo_root.resolve()
    tasks: List[ImageTask] = []
    for label, path in _iter_image_files(images_root, splits):
        try:
            rel_path = path.relative_to(repo_root)
            image_name = str(rel_path)
        except ValueError:
            image_name = str(path)
        tasks.append(ImageTask(label=label, path=str(path), image_name=image_name))
    return tasks


def _write_payload(output_path: Path, images_root: Path, splits: Sequence[str], items: List[dict], interrupted: bool, missing_boxes: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images_root": str(images_root),
        "splits": list(splits),
        "interrupted": interrupted,
        "missing_boxes": missing_boxes,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    images_root = Path("data/dataset2-master/dataset2-master/images")
    splits = ("TRAIN",)
    repo_root = Path.cwd()
    output = Path("target/bounding-boxes.json")
    workers = 12

    tasks = _build_tasks(images_root, splits, repo_root)
    total = len(tasks)
    if total == 0:
        print("No images found; nothing to do", file=sys.stderr)
        return

    records: List[dict] = []
    processed = 0
    missing_boxes = 0
    interrupted = False

    print(f"Processing {total} images with {workers} worker(s)...", file=sys.stderr)
    results = run_tasks(tasks, _process_task, workers=workers, mode="process")

    missing_boxes = 0
    for task, boxes in zip(tasks, results):
        if not boxes:
            missing_boxes += 1
        records.append(
            {
                "image": task.image_name,
                "label": task.label,
                "boxes": boxes,
            }
        )

    _write_payload(output, images_root, splits, records, False, missing_boxes)


if __name__ == "__main__":
    main()
