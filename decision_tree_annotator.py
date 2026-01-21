#!/usr/bin/env python3
"""Train a decision tree on extracted features and annotate new images."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from common import component_stats, initial_mask, is_white_cell, load_image
from feature_extraction import FEATURE_NAMES, extract_feature_vector


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
    image, arr = load_image(image_path)
    mask = initial_mask(arr)
    comps = component_stats(mask)
    image_area = arr.shape[0] * arr.shape[1]

    detections: List[Tuple[tuple[int, int, int, int], str]] = []
    for comp in comps:
        if not is_white_cell(comp, image_area):
            continue
        bbox = comp.bbox
        x0, y0, x1, y1 = bbox
        region = arr[y0 : y1 + 1, x0 : x1 + 1]
        features = extract_feature_vector(region, bbox, (arr.shape[0], arr.shape[1]))
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
    # Show usage help
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        print("Usage:")
        print("  python decision_tree_annotator.py                    # Process all TEST images")
        print("  python decision_tree_annotator.py <input> <output>  # Process single image")
        print("  python decision_tree_annotator.py --help             # Show this help")
        return
    
    # Check if running in single file mode (input and output arguments)
    if len(sys.argv) == 3:
        input_path = Path(sys.argv[1])
        output_path = Path(sys.argv[2])
        
        # Load training data and train model
        feature_rows, labels = _load_training_data(Path("target/features.json"))
        if not feature_rows:
            raise ValueError("Feature manifest contains no rows")

        tree = DecisionTreeClassifier()
        tree.fit(feature_rows, labels)

        # Process single image
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        out_path, counts = _annotate_image(tree, input_path, output_path)
        if counts:
            summary = ", ".join(f"{label}:{counts[label]}" for label in sorted(counts.keys()))
        else:
            summary = "No white cells detected"
        print(f"Annotated {out_path} :: {summary}")
        return

    # Default mode: process all TEST images
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
