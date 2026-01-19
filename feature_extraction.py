#!/usr/bin/env python3
"""Extract feature vectors per white-cell bounding box and emit features.json."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from common import BLUE_DOM_PIX_THRESH, run_tasks
from morphology import binary_close, binary_open, binary_erode


@dataclass(frozen=True)
class FeatureTask:
    image_value: str
    label: str
    boxes: Tuple[Tuple[int, int, int, int], ...]
    path: str


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                comps += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
    return comps


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    holes = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    inv = ~mask
    for y in range(h):
        for x in range(w):
            if inv[y, x] and not visited[y, x]:
                touches_border = y in (0, h - 1) or x in (0, w - 1)
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if inv[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                                if ny in (0, h - 1) or nx in (0, w - 1):
                                    touches_border = True
                if not touches_border:
                    holes += 1
    return holes


def _component_metrics(mask: np.ndarray) -> Tuple[int, int, int]:
    comps = _count_components(mask)
    holes = _count_holes(mask)
    return comps, holes, comps - holes


def _prep_mask(mask: np.ndarray, close_iter: int = 1, open_iter: int = 1) -> np.ndarray:
    result = mask
    if close_iter:
        result = binary_close(result, close_iter)
    if open_iter:
        result = binary_open(result, open_iter)
    return result


def _extract_masks(region: np.ndarray) -> Dict[str, np.ndarray]:
    r = region[..., 0]
    g = region[..., 1]
    b = region[..., 2]

    blue_mask = (b > 0.42) & (b - (0.45 * r + 0.55 * g) > BLUE_DOM_PIX_THRESH) & (b > g)
    nucleus_mask = (b > 0.5) & ((b - r) > 0.15) & ((b - g) > 0.15)
    eos_pink_mask = (r > 0.64) & (g > 0.48) & (b < 0.78)
    pale_pink_mask = (r > 0.52) & (g > 0.45) & (b < 0.84) & ((r - g) < 0.12)
    monocyte_mask = (r > 0.45) & (g > 0.4) & (b > 0.35) & (np.abs(r - g) < 0.1) & (g >= b - 0.05)
    purple_mask = (r > 0.45) & (b > 0.5)

    masks = {
        "blue": _prep_mask(blue_mask, 1, 1),
        "nucleus": _prep_mask(nucleus_mask, 1, 1),
        "eosin": _prep_mask(eos_pink_mask, 1, 1),
        "pale": _prep_mask(pale_pink_mask, 1, 1),
        "monocyte": _prep_mask(monocyte_mask, 1, 1),
        "purple": _prep_mask(purple_mask, 1, 1),
    }
    masks["cell"] = masks["blue"] | masks["eosin"] | masks["pale"] | masks["monocyte"]
    return masks


def _perimeter(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    eroded = binary_erode(mask, 1)
    border = mask & ~eroded
    return int(border.sum())


def _moment_features(mask: np.ndarray) -> Tuple[float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 2:
        return 0.0, 0.0
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    ratio = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else 0.0
    spread = float(eigvals.sum())
    return ratio, spread


FEATURE_NAMES: Sequence[str] = (
    "bbox_width_ratio",
    "bbox_height_ratio",
    "bbox_aspect_ratio",
    "bbox_area_ratio",
    "blue_fraction",
    "nucleus_fraction",
    "eosin_fraction",
    "pale_fraction",
    "monocyte_fraction",
    "cell_fraction",
    "mean_red",
    "mean_green",
    "mean_blue",
    "std_red",
    "std_green",
    "std_blue",
    "mean_blue_dom",
    "mean_saturation",
    "purple_fraction",
    "nucleus_components",
    "nucleus_holes",
    "pale_components",
    "pale_euler",
    "blue_holes",
    "compactness",
    "moment_ratio",
    "moment_spread",
)


def extract_feature_vector(
    region: np.ndarray,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> List[float]:
    h, w = region.shape[:2]
    bbox_w = bbox[2] - bbox[0] + 1
    bbox_h = bbox[3] - bbox[1] + 1
    image_h, image_w = image_shape
    bbox_area = bbox_w * bbox_h
    image_area = image_h * image_w

    r = region[..., 0]
    g = region[..., 1]
    b = region[..., 2]
    flat = region.reshape(-1, 3)

    masks = _extract_masks(region)

    blue_fraction = float(masks["blue"].mean())
    nucleus_fraction = float(masks["nucleus"].mean())
    eos_fraction = float(masks["eosin"].mean())
    pale_fraction = float(masks["pale"].mean())
    monocyte_fraction = float(masks["monocyte"].mean())
    cell_fraction = float(masks["cell"].mean())
    purple_fraction = float(masks["purple"].mean())

    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    blue_dom = b - 0.5 * (r + g)
    mean_blue_dom = float(blue_dom.mean())
    max_rgb = region.max(axis=2)
    min_rgb = region.min(axis=2)
    mean_saturation = float((max_rgb - min_rgb).mean())

    metrics = {}
    for key in ("nucleus", "pale", "blue"):
        comps, holes, euler = _component_metrics(masks[key])
        metrics[key] = (float(comps), float(holes), float(euler))

    perimeter = _perimeter(masks["cell"])
    compactness = (perimeter ** 2) / max(1.0, masks["cell"].sum())
    moment_ratio, moment_spread = _moment_features(masks["cell"])

    features = [
        bbox_w / image_w,
        bbox_h / image_h,
        bbox_w / max(1.0, float(bbox_h)),
        bbox_area / max(1.0, float(image_area)),
        blue_fraction,
        nucleus_fraction,
        eos_fraction,
        pale_fraction,
        monocyte_fraction,
        cell_fraction,
        float(mean_rgb[0]),
        float(mean_rgb[1]),
        float(mean_rgb[2]),
        float(std_rgb[0]),
        float(std_rgb[1]),
        float(std_rgb[2]),
        mean_blue_dom,
        mean_saturation,
        purple_fraction,
        metrics["nucleus"][0],
        metrics["nucleus"][1],
        metrics["pale"][0],
        metrics["pale"][2],
        metrics["blue"][1],
        compactness,
        moment_ratio,
        moment_spread,
    ]
    return features


def _clamp_bbox(bbox: Sequence[int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    image_h, image_w = image_shape
    x0, y0, x1, y1 = bbox
    x0 = int(min(max(x0, 0), image_w - 1))
    y0 = int(min(max(y0, 0), image_h - 1))
    x1 = int(min(max(x1, x0), image_w - 1))
    y1 = int(min(max(y1, y0), image_h - 1))
    return x0, y0, x1, y1


def _resolve_image_path(repo_root: Path, image_value: str) -> Path:
    candidate = Path(image_value)
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _build_tasks(items: List[dict], repo_root: Path) -> List[FeatureTask]:
    repo_root = repo_root.resolve()
    tasks: List[FeatureTask] = []
    for entry in items:
        image_value = entry.get("image")
        if not image_value:
            continue
        label = entry.get("label", "UNKNOWN")
        raw_boxes = entry.get("boxes", [])
        boxes: List[Tuple[int, int, int, int]] = []
        for bbox in raw_boxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                print(f"[WARN] Skipping malformed bbox {bbox} for {image_value}", file=sys.stderr)
                continue
            boxes.append(tuple(int(v) for v in bbox))
        if not boxes:
            print(f"[WARN] No bounding boxes for {image_value}; skipping", file=sys.stderr)
            continue
        path = _resolve_image_path(repo_root, image_value)
        if not path.exists():
            print(f"[WARN] Image {path} missing; skipping", file=sys.stderr)
            continue
        tasks.append(
            FeatureTask(
                image_value=image_value,
                label=label,
                boxes=tuple(boxes),
                path=str(path),
            )
        )
    return tasks


def _process_task(task: FeatureTask) -> List[dict]:
    path = Path(task.path)
    arr = _load_image(path)
    image_shape = (arr.shape[0], arr.shape[1])
    entries: List[dict] = []
    for bbox in task.boxes:
        x0, y0, x1, y1 = _clamp_bbox(bbox, image_shape)
        region = arr[y0 : y1 + 1, x0 : x1 + 1]
        features = extract_feature_vector(region, (x0, y0, x1, y1), image_shape)
        entries.append(
            {
                "image": task.image_value,
                "label": task.label,
                "bbox": [x0, y0, x1, y1],
                "features": features,
            }
        )
    return entries


def _write_output(output_path: Path, records: List[dict], interrupted: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": list(FEATURE_NAMES),
        "interrupted": interrupted,
        "items": records,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    bounding_boxes = Path("target/bounding-boxes.json")
    output = Path("target/features.json")
    repo_root = Path.cwd()
    workers = 12

    if not bounding_boxes.exists():
        raise FileNotFoundError(
            f"Bounding box manifest {bounding_boxes} not found. Run bounding_boxes_creation.py first."
        )

    payload = json.loads(bounding_boxes.read_text())
    items = payload.get("items", [])
    if not items:
        print("No bounding boxes to process", file=sys.stderr)
        _write_output(output, [], False)
        return

    tasks = _build_tasks(items, repo_root)
    if not tasks:
        print("No valid bounding boxes to process after filtering", file=sys.stderr)
        _write_output(output, [], False)
        return

    total = len(tasks)
    records: List[dict] = []

    print(f"Processing {total} images with {workers} worker(s)...", file=sys.stderr)

    results = run_tasks(tasks, _process_task, workers=workers, mode="process")
    for entries in results:
        records.extend(entries)

    _write_output(output, records, False)


if __name__ == "__main__":
    main()
