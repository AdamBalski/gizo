#!/usr/bin/env python3
"""Create a bounding-box database for dataset2 microscope images."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from common import (
    VALID_SUFFIXES,
    component_stats,
    initial_mask,
    is_white_cell,
    load_image,
    run_tasks,
)


@dataclass(frozen=True)
class ImageTask:
    label: str
    path: str
    image_name: str


def _get_bounding_boxes(image_path: Path) -> List[tuple[int, int, int, int]]:
    _img, arr = load_image(image_path)
    mask = initial_mask(arr)
    comps = component_stats(mask)
    image_area = arr.shape[0] * arr.shape[1]
    return [comp.bbox for comp in comps if is_white_cell(comp, image_area)]


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
                if path.suffix.lower() not in VALID_SUFFIXES:
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

    print(f"Processing {total} images with {workers} worker(s)...", file=sys.stderr)
    results = run_tasks(tasks, _process_task, workers=workers, mode="process")

    records: List[dict] = []
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
