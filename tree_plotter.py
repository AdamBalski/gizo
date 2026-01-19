#!/usr/bin/env python3
"""Visualize the trained decision tree as a PNG diagram."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

from decision_tree_annotator import (
    FEATURE_NAMES,
    DecisionTreeClassifier,
    TreeNode,
    _load_training_data,
)


def _layout_tree(node: TreeNode, depth: int, positions: Dict[int, Tuple[float, float]], x_counter: list[int]) -> None:
    if node is None:
        return
    _layout_tree(node.left, depth + 1, positions, x_counter)
    x = x_counter[0]
    positions[id(node)] = (x, -depth)
    x_counter[0] += 1
    _layout_tree(node.right, depth + 1, positions, x_counter)


def _label_for(node: TreeNode) -> str:
    if node.is_leaf():
        return node.label or "UNKNOWN"
    assert node.feature_index is not None
    assert node.threshold is not None
    feature_name = FEATURE_NAMES[node.feature_index]
    return f"{feature_name}\n<= {node.threshold:.3f}"


def _collect_depth(node: TreeNode | None) -> int:
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + max(_collect_depth(node.left), _collect_depth(node.right))


def _plot_tree(root: TreeNode, output: Path, dpi: int) -> None:
    positions: Dict[int, Tuple[float, float]] = {}
    _layout_tree(root, 0, positions, [0])
    if not positions:
        raise ValueError("Decision tree is empty")

    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.6), 4 + _collect_depth(root) * 0.8))
    ax.axis("off")

    def _draw_edges(node: TreeNode) -> None:
        if node is None or node.is_leaf():
            return
        parent_pos = positions[id(node)]
        for child in (node.left, node.right):
            if child is None:
                continue
            child_pos = positions[id(child)]
            ax.plot(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                color="#666666",
                linewidth=1.5,
            )
            _draw_edges(child)

    _draw_edges(root)

    leaf_color = "#8bc34a"
    internal_color = "#03a9f4"

    def _draw_nodes(node: TreeNode) -> None:
        if node is None:
            return
        x, y = positions[id(node)]
        color = leaf_color if node.is_leaf() else internal_color
        ax.scatter([x], [y], color=color, s=500, zorder=3)
        label = _label_for(node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        _draw_nodes(node.left)
        _draw_nodes(node.right)

    _draw_nodes(root)
    ax.set_title("Decision tree structure")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    features_path = Path("target/features.json")
    output_path = Path("target/decision-tree.png")
    dpi = 200

    features, labels = _load_training_data(features_path)
    if not features:
        raise ValueError("Feature manifest is empty; run feature-extraction.py first")
    tree = DecisionTreeClassifier()
    tree.fit(features, labels)
    assert tree.root is not None
    _plot_tree(tree.root, output_path, dpi)
    print(f"Wrote decision tree diagram to {output_path}")


if __name__ == "__main__":
    main()
