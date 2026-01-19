#!/usr/bin/env python3
"""Plot evaluation summaries from target/eval-results.json."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation results {path} not found. Run evaluator.py first."
        )
    return json.loads(path.read_text())


def _compute_stats(items: Sequence[dict]) -> dict:
    expected_totals: Counter[str] = Counter()
    expected_correct: Counter[str] = Counter()
    predicted_totals: Counter[str] = Counter()
    predicted_correct: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    overall_correct = 0

    for item in items:
        expected = str(item.get("expected", "UNKNOWN")).upper()
        predictions = [str(p).upper() for p in item.get("predictions", [])]
        matched = bool(item.get("matched"))
        expected_totals[expected] += 1
        if matched:
            expected_correct[expected] += 1
            overall_correct += 1
        if not predictions:
            confusion[expected]["(NONE)"] += 1
        for pred in predictions:
            predicted_totals[pred] += 1
            if pred == expected:
                predicted_correct[pred] += 1
            confusion[expected][pred] += 1

    labels = sorted(expected_totals.keys())
    predicted_labels = sorted(predicted_totals.keys())
    return {
        "expected_totals": expected_totals,
        "expected_correct": expected_correct,
        "predicted_totals": predicted_totals,
        "predicted_correct": predicted_correct,
        "confusion": confusion,
        "labels": labels,
        "predicted_labels": predicted_labels,
        "overall_correct": overall_correct,
    }


def _bar_stats(ax, labels: Sequence[str], numerators: Counter[str], denominators: Counter[str], title: str, ylabel: str) -> None:
    if not labels:
        ax.set_visible(False)
        return
    values = []
    for label in labels:
        total = denominators.get(label, 0)
        correct = numerators.get(label, 0)
        pct = (correct / total * 100.0) if total else 0.0
        values.append(pct)
    bars = ax.bar(labels, values, color="#4c72b0")
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for rect, pct in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _plot_confusion(ax, stats: dict) -> None:
    expected_labels = stats["labels"]
    predicted_labels = stats["predicted_labels"]
    if not expected_labels or not predicted_labels:
        ax.set_visible(False)
        return
    matrix = np.zeros((len(expected_labels), len(predicted_labels)), dtype=float)
    for i, exp in enumerate(expected_labels):
        for j, pred in enumerate(predicted_labels):
            matrix[i, j] = stats["confusion"][exp][pred]
    if matrix.max() > 0:
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(predicted_labels)))
    ax.set_xticklabels(predicted_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(expected_labels)))
    ax.set_yticklabels(expected_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Expected label")
    ax.set_title("Confusion heatmap (row-normalized)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _add_summary_text(ax, results: dict, stats: dict) -> None:
    ax.axis("off")
    processed = int(results.get("processed", 0))
    correct = int(stats["overall_correct"])
    accuracy = (correct / processed * 100.0) if processed else 0.0
    lines = [
        f"Total evaluated: {processed}",
        f"Correct matches: {correct}",
        f"Overall accuracy: {accuracy:.2f}%",
        f"Interrupted: {results.get('interrupted', False)}",
    ]
    if missing := [label for label in stats["labels"] if stats["expected_totals"][label] == 0]:
        lines.append(f"Labels with zero coverage: {', '.join(missing)}")
    ax.text(0.0, 0.9, "Summary", fontsize=14, fontweight="bold")
    ax.text(0.0, 0.75, "\n".join(lines), fontsize=12, va="top")


def _plot(results: dict, stats: dict, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_expected, ax_predicted, ax_confusion, ax_summary = axes.flatten()

    _bar_stats(
        ax_expected,
        stats["labels"],
        stats["expected_correct"],
        stats["expected_totals"],
        "Accuracy per expected class",
        "Correct (%)",
    )
    _bar_stats(
        ax_predicted,
        stats["predicted_labels"],
        stats["predicted_correct"],
        stats["predicted_totals"],
        "Precision per predicted class",
        "Precision (%)",
    )
    _plot_confusion(ax_confusion, stats)
    _add_summary_text(ax_summary, results, stats)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    results_path = Path("target/eval-results.json")
    output_path = Path("target/eval-summary.png")
    dpi = 150

    results = _load_results(results_path)
    items = results.get("items", [])
    if not items:
        raise ValueError("Evaluation file has no items; run evaluator.py first.")
    stats = _compute_stats(items)
    _plot(results, stats, output_path, dpi)
    print(
        f"Wrote summary plot to {output_path} (overall accuracy {results.get('accuracy', 0):.2%})"
    )


if __name__ == "__main__":
    main()
