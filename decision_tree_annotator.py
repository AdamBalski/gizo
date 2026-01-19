#!/usr/bin/env python3
"""Train a decision tree on extracted features and annotate new images."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from morphology import binary_close, binary_open, binary_erode

_BLUE_DOM_PIX_THRESH = 0.08
@dataclass
class Component:
    bbox: tuple[int, int, int, int]
    pixels: int
    solidity: float


@dataclass
class TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode" | None = None
    right: "TreeNode" | None = None
    label: str | None = None

    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 8,
        min_leaf: int = 5,
        min_gain: float = 1e-3,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.min_gain = min_gain
        self.purity_threshold = 0.9
        self._features: List[Sequence[float]] = []
        self._labels: List[str] = []
        self.root: TreeNode | None = None

    def fit(self, features: Sequence[Sequence[float]], labels: Sequence[str]) -> None:
        if not features:
            raise ValueError("Cannot train tree without features")
        self._features = [tuple(row) for row in features]
        self._labels = list(labels)
        indices = list(range(len(self._features)))
        self.root = self._build_node(indices, depth=0)

    def predict(self, feature_vector: Sequence[float]) -> str:
        if self.root is None:
            raise ValueError("Decision tree not trained")
        node = self.root
        while not node.is_leaf():
            assert node.feature_index is not None
            assert node.threshold is not None
            if feature_vector[node.feature_index] <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right
        assert node.label is not None
        return node.label

    def _build_node(self, indices: List[int], depth: int) -> TreeNode:
        label_counts = Counter(self._labels[i] for i in indices)
        majority_label, majority_count = label_counts.most_common(1)[0]
        purity = majority_count / len(indices)
        if len(label_counts) == 1 or depth >= self.max_depth or len(indices) <= self.min_leaf:
            return TreeNode(label=majority_label)
        if purity >= self.purity_threshold:
            return TreeNode(label=majority_label)

        best_feature, best_threshold, gain = self._best_split(indices, label_counts)
        if best_feature is None or gain < self.min_gain:
            return TreeNode(label=majority_label)

        left_indices = [i for i in indices if self._features[i][best_feature] <= best_threshold]
        right_indices = [i for i in indices if self._features[i][best_feature] > best_threshold]
        if not left_indices or not right_indices:
            return TreeNode(label=majority_label)

        left_child = self._build_node(left_indices, depth + 1)
        right_child = self._build_node(right_indices, depth + 1)
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
        )

    def _best_split(
        self, indices: Sequence[int], label_counts: Counter
    ) -> Tuple[int | None, float | None, float]:
        base_entropy = _entropy_counts(label_counts)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None
        num_features = len(self._features[0])
        total = len(indices)

        for feature_idx in range(num_features):
            values = sorted((self._features[i][feature_idx], i) for i in indices)
            left_counts: Dict[str, int] = {label: 0 for label in label_counts}
            left_size = 0
            for pos in range(len(values) - 1):
                value, sample_idx = values[pos]
                label = self._labels[sample_idx]
                left_counts[label] = left_counts.get(label, 0) + 1
                left_size += 1
                next_value = values[pos + 1][0]
                if next_value == value:
                    continue
                right_size = total - left_size
                if left_size < self.min_leaf or right_size < self.min_leaf:
                    continue
                right_counts = {
                    lbl: label_counts[lbl] - left_counts.get(lbl, 0)
                    for lbl in label_counts
                }
                left_entropy = _entropy_counts(left_counts)
                right_entropy = _entropy_counts(right_counts)
                weighted = (left_size / total) * left_entropy + (right_size / total) * right_entropy
                gain = base_entropy - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = (value + next_value) / 2.0
            if best_gain >= self.min_gain and best_feature == feature_idx:
                # early continue only if we already found a good split for this feature
                continue

        return best_feature, best_threshold, best_gain


def _entropy_counts(counts: Dict[str, int]) -> float:
    total = sum(max(c, 0) for c in counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _load_font() -> ImageFont.ImageFont:
    for font_name in ("Arial.ttf", "Helvetica.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(font_name, 24)
        except OSError:
            continue
    return ImageFont.load_default()


_LABEL_FONT = _load_font()


def _load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
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


def _component_stats(mask: np.ndarray, arr: np.ndarray) -> List[Component]:
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


def _extract_masks(region: np.ndarray) -> Dict[str, np.ndarray]:
    r = region[..., 0]
    g = region[..., 1]
    b = region[..., 2]
    blue_mask = (b > 0.42) & (b - (0.45 * r + 0.55 * g) > _BLUE_DOM_PIX_THRESH) & (b > g)
    nucleus_mask = (b > 0.5) & ((b - r) > 0.15) & ((b - g) > 0.15)
    eos_pink_mask = (r > 0.64) & (g > 0.48) & (b < 0.78)
    pale_pink_mask = (r > 0.52) & (g > 0.45) & (b < 0.84) & ((r - g) < 0.12)
    monocyte_mask = (r > 0.45) & (g > 0.4) & (b > 0.35) & (np.abs(r - g) < 0.1) & (g >= b - 0.05)
    purple_mask = (r > 0.45) & (b > 0.5)
    masks = {
        "blue": binary_open(binary_close(blue_mask, 1), 1),
        "nucleus": binary_open(binary_close(nucleus_mask, 1), 1),
        "eosin": binary_open(binary_close(eos_pink_mask, 1), 1),
        "pale": binary_open(binary_close(pale_pink_mask, 1), 1),
        "monocyte": binary_open(binary_close(monocyte_mask, 1), 1),
        "purple": binary_open(binary_close(purple_mask, 1), 1),
    }
    masks["cell"] = masks["blue"] | masks["eosin"] | masks["pale"] | masks["monocyte"]
    return masks


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


def _component_metrics(mask: np.ndarray) -> Tuple[float, float, float]:
    comps = _count_components(mask)
    holes = _count_holes(mask)
    return float(comps), float(holes), float(comps - holes)


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


def _extract_feature_vector(
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

    metrics = {key: _component_metrics(mask) for key, mask in masks.items() if key in {"nucleus", "pale", "blue"}}

    perimeter = _perimeter(masks["cell"])
    perimeter_ratio = perimeter / max(1.0, 2.0 * (h + w))
    compactness = (perimeter ** 2) / max(1.0, masks["cell"].sum())
    moment_ratio, moment_spread = _moment_features(masks["cell"])

    return [
        bbox_w / max(1.0, image_w),
        bbox_h / max(1.0, image_h),
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


def _clamp_bbox(bbox: Sequence[int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    image_h, image_w = image_shape
    x0, y0, x1, y1 = bbox
    x0 = int(min(max(x0, 0), image_w - 1))
    y0 = int(min(max(y0, 0), image_h - 1))
    x1 = int(min(max(x1, x0), image_w - 1))
    y1 = int(min(max(y1, y0), image_h - 1))
    return x0, y0, x1, y1


def _load_training_data(features_path: Path) -> Tuple[List[Sequence[float]], List[str]]:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature manifest {features_path} missing. Run feature-extraction.py first."
        )
    payload = json.loads(features_path.read_text())
    feature_names = payload.get("features")
    if feature_names != list(FEATURE_NAMES):
        raise ValueError(
            "Feature manifest columns do not match expected FEATURE_NAMES order."
        )
    rows = []
    labels = []
    for item in payload.get("items", []):
        rows.append(item.get("features", []))
        labels.append(item.get("label", "UNKNOWN"))
    return rows, labels


def _draw_label(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    text: str,
    color: str,
    canvas_size: tuple[int, int],
) -> None:
    label = text.upper()
    padding = 5
    try:
        text_box = _LABEL_FONT.getbbox(label)
        text_w, text_h = text_box[2] - text_box[0], text_box[3] - text_box[1]
    except AttributeError:
        text_w, text_h = _LABEL_FONT.getsize(label)
    x0, y0, x1, y1 = bbox
    canvas_w, canvas_h = canvas_size
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    left = min(max(x0 + 2, 0), max(0, x1 - box_w))
    top = min(max(y0 + 2, 0), max(0, y1 - box_h))
    if x1 - x0 < box_w:
        left = min(max(0, x0), max(0, canvas_w - box_w))
    if y1 - y0 < box_h:
        below = y1 + 2
        if below + box_h <= canvas_h:
            top = below
        else:
            top = max(0, y0 - box_h)

    draw.rectangle([left, top, left + box_w, top + box_h], fill=color)
    draw.text((left + padding, top + padding), label, fill="black", font=_LABEL_FONT)


def _draw_legend(draw: ImageDraw.ImageDraw, counts: Dict[str, int], canvas_size: tuple[int, int]) -> None:
    if not counts:
        return
    padding = 5
    spacing = 8
    x = padding
    y = padding
    for label in sorted(counts.keys()):
        count = counts[label]
        text = f"{label}:{count}"
        try:
            bbox = _LABEL_FONT.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = _LABEL_FONT.getsize(text)
        box = [x, y, x + text_w + padding * 2, y + text_h + padding * 2]
        color = LABEL_COLORS.get(label, "white")
        draw.rectangle(box, fill=color)
        draw.text((x + padding, y + padding), text, fill="black", font=_LABEL_FONT)
        x = box[2] + spacing
        if x + text_w + padding * 2 > canvas_size[0]:
            x = padding
            y = box[3] + spacing


LABEL_COLORS = {
    "LYMPHOCYTE": "lime",
    "EOSINOPHIL": "orange",
    "NEUTROPHIL": "cyan",
    "MONOCYTE": "magenta",
}


def _annotate_image(
    tree: DecisionTreeClassifier,
    image_path: Path,
    output_path: Path,
) -> Tuple[Path, Dict[str, int]]:
    image, arr = _load_image(image_path)
    mask = _initial_mask(arr)
    comps = _component_stats(mask, arr)
    image_area = arr.shape[0] * arr.shape[1]

    detections: List[Tuple[tuple[int, int, int, int], str]] = []
    for comp in comps:
        if not _is_white_cell(comp, image_area):
            continue
        bbox = comp.bbox
        x0, y0, x1, y1 = bbox
        region = arr[y0 : y1 + 1, x0 : x1 + 1]
        features = _extract_feature_vector(region, bbox, (arr.shape[0], arr.shape[1]))
        label = tree.predict(features)
        detections.append((bbox, label))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw = ImageDraw.Draw(image)
    counts: Dict[str, int] = {}
    for bbox, label in detections:
        color = LABEL_COLORS.get(label, "white")
        draw.rectangle(bbox, outline=color, width=4)
        _draw_label(draw, bbox, label, color, image.size)
        counts[label] = counts.get(label, 0) + 1
    _draw_legend(draw, counts, image.size)
    image.save(output_path)
    return output_path, counts


def main() -> None:
    feature_rows, labels = _load_training_data(Path("target/features.json"))
    if not feature_rows:
        raise ValueError("Feature manifest contains no rows")

    tree = DecisionTreeClassifier()
    tree.fit(feature_rows, labels)

    test_root = Path("data/dataset2-master/dataset2-master/images/TEST")
    output_dir = Path("target")
    data_root = Path("data")

    for path in sorted(test_root.rglob("*.jpeg")):
        rel = path.relative_to(data_root)
        out_path, counts = _annotate_image(tree, path, output_dir / rel)
        if counts:
            summary = ", ".join(f"{label}:{counts[label]}" for label in sorted(counts.keys()))
        else:
            summary = "No white cells detected"
        print(f"Annotated {out_path} :: {summary}")


if __name__ == "__main__":
    main()
