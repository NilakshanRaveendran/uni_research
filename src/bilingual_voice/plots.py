"""Generate thesis-ready diagnostic plots from auditable result rows."""

from __future__ import annotations

import math
from pathlib import Path


def _finite(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _paired(
    rows: list[dict[str, str]], first_field: str, second_field: str
) -> list[tuple[float, float]]:
    pairs = []
    for row in rows:
        try:
            first = float(row.get(first_field, ""))
            second = float(row.get(second_field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(first) and math.isfinite(second):
            pairs.append((first, second))
    return pairs


def create_plots(rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if row.get("status") == "ok"]
    directions = sorted({row.get("direction", "unknown") for row in successful})
    written: list[Path] = []

    similarity_groups: list[list[float]] = []
    similarity_labels: list[str] = []
    for direction in directions:
        group = [row for row in successful if row["direction"] == direction]
        for field, label in (
            ("speaker_similarity", "generated"),
            ("human_target_speaker_similarity", "human target"),
        ):
            values = _finite(group, field)
            if values:
                similarity_groups.append(values)
                similarity_labels.append(f"{direction}\n{label}")
    if similarity_groups:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.boxplot(similarity_groups, tick_labels=similarity_labels, showmeans=True)
        axis.set_ylabel("ECAPA cosine similarity (higher is better)")
        axis.set_title("Speaker identity: generated speech vs human cross-language anchor")
        axis.grid(axis="y", alpha=0.25)
        path = output_dir / "speaker_similarity_by_direction.png"
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written.append(path)

    for feature, label in (
        ("f0_mean", "Mean F0 (Hz)"),
        ("f0_std", "F0 variation (Hz)"),
        ("energy", "Mean RMS energy"),
        ("duration", "Duration (seconds)"),
        ("speaking_rate", "Speaking rate (words/second)"),
    ):
        figure, axes = plt.subplots(
            1,
            max(1, len(directions)),
            figsize=(6 * max(1, len(directions)), 5),
            squeeze=False,
        )
        plotted = False
        for axis, direction in zip(axes[0], directions, strict=False):
            group = [row for row in successful if row["direction"] == direction]
            pairs = _paired(
                group,
                f"{feature}_target_reference",
                f"{feature}_generated",
            )
            if not pairs:
                continue
            x_values, y_values = zip(*pairs, strict=True)
            axis.scatter(x_values, y_values, alpha=0.55, s=18)
            low = min(min(x_values), min(y_values))
            high = max(max(x_values), max(y_values))
            axis.plot([low, high], [low, high], "--", color="gray", linewidth=1)
            axis.set_title(direction)
            axis.set_xlabel(f"Human target reference: {label}")
            axis.set_ylabel(f"Generated: {label}")
            axis.grid(alpha=0.2)
            plotted = True
        if plotted:
            path = output_dir / f"prosody_{feature}_generated_vs_reference.png"
            figure.tight_layout()
            figure.savefig(path, dpi=200)
            written.append(path)
        plt.close(figure)
    return written
