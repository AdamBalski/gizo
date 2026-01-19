#!/usr/bin/env python3
"""Evaluate the decision tree against dataset2 TEST images."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

_VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}

from common import run_tasks
from decision_tree_annotator import (
    DecisionTreeClassifier,
    _component_stats,
    _extract_feature_vector,
    _initial_mask,
    _is_white_cell,
    _load_image,
    _load_training_data,
)


@dataclass(frozen=True)
class EvalTask:
    label: str
    path: str
    rel_path: str


def _iter_eval_tasks(test_root: Path, data_root: Path) -> List[EvalTask]:
    tasks: List[EvalTask] = []
    data_root = data_root.resolve()
    if not test_root.exists():
        raise FileNotFoundError(f"Test root {test_root} not found")
    for label_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        label = label_dir.name.upper()
        for path in sorted(label_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _VALID_SUFFIXES:
                continue
            try:
                rel_path = path.resolve().relative_to(data_root)
                rel = str(rel_path)
            except ValueError:
                rel = str(path)
            tasks.append(EvalTask(label=label, path=str(path.resolve()), rel_path=rel))
    return tasks


def _run_detector(
    task: EvalTask,
    tree: DecisionTreeClassifier,
) -> Dict[str, object]:
    _image, arr = _load_image(Path(task.path))
    mask = _initial_mask(arr)
    comps = _component_stats(mask, arr)
    image_shape = (arr.shape[0], arr.shape[1])
    image_area = image_shape[0] * image_shape[1]
    predictions: List[str] = []
    for comp in comps:
        if not _is_white_cell(comp, image_area):
            continue
        x0, y0, x1, y1 = comp.bbox
        region = arr[y0 : y1 + 1, x0 : x1 + 1]
        features = _extract_feature_vector(region, comp.bbox, image_shape)
        pred = tree.predict(features).upper()
        predictions.append(pred)
    matched = task.label in predictions
    return {
        "image": task.rel_path,
        "expected": task.label,
        "predictions": predictions,
        "matched": matched,
    }


def _write_results(
    output_path: Path,
    test_root: Path,
    total: int,
    processed: int,
    correct: int,
    interrupted: bool,
    records: Sequence[Dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = (correct / processed) if processed else 0.0
    payload = {
        "test_root": str(test_root),
        "total": total,
        "processed": processed,
        "correct": correct,
        "accuracy": accuracy,
        "interrupted": interrupted,
        "items": list(records),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    features_path = Path("target/features.json")
    feature_rows, labels = _load_training_data(features_path)
    if not feature_rows:
        raise ValueError("Feature manifest contains no rows; run feature-extraction.py first")
    tree = DecisionTreeClassifier()
    tree.fit(feature_rows, labels)

    test_root = Path("data/dataset2-master/dataset2-master/images/TEST")
    data_root = Path("data")
    output = Path("target/eval-results.json")
    workers = 12

    tasks = _iter_eval_tasks(test_root, data_root)
    total = len(tasks)
    if total == 0:
        print("No evaluation images found", file=sys.stderr)
        _write_results(output, test_root, 0, 0, 0, False, [])
        return

    print(f"Evaluating {total} TEST images with {workers} worker(s)...", file=sys.stderr)

    results = run_tasks(tasks, lambda task: _run_detector(task, tree), workers=workers, mode="thread")
    records: List[Dict[str, object]] = list(results)
    correct = sum(1 for res in records if res["matched"])
    processed = len(records)

    _write_results(output, test_root, total, processed, correct, False, records)


if __name__ == "__main__":
    main()
