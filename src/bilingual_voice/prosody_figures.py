"""Figures for the final presentation, all regenerated from committed artefacts.

    python -m bilingual_voice.prosody_figures all --outdir plots/prosody

Nothing here hardcodes a number. Every value is read from the CSVs that `prosody_eval` and
`prosody_ladder` write, because a hardcoded figure title has already gone wrong once in this
project -- a chart claimed "63-107%" when the data said 69-128%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARM_ORDER = (
    "S0",
    "S0_identity",
    "S1_copy",
    "S2_mlp",
    "S2_cnn",
    "S2_cnn_pitch",
    "S3_dct",
    "S3_full",
    "S3_full_pitch",
)
ARM_LABELS = {
    "S0": "S0\nraw XTTS",
    "S0_identity": "S0′\nvocoder only",
    "S1_copy": "S1\ncopy source",
    "S2_mlp": "S2a\nProsoMLP",
    "S2_cnn": "S2b\nProsoCNN",
    "S3_dct": "S3-dct\noracle, 8 coef",
    "S3_full": "S3\noracle, full",
    "S2_cnn_pitch": "S2b-p\nCNN, pitch only",
    "S3_full_pitch": "S3-p\noracle, pitch only",
}
# Short codes for axis ticks. Two-line labels collide once there are more than a few arms, so the
# ticks carry codes and the figure carries one legend line explaining them.
ARM_SHORT = {
    "S0": "S0",
    "S0_identity": "S0′",
    "S1_copy": "S1",
    "S2_mlp": "S2a",
    "S2_cnn": "S2b",
    "S2_cnn_pitch": "S2b-p",
    "S3_dct": "S3d",
    "S3_full": "S3",
    "S3_full_pitch": "S3-p",
}
ARM_KEY = (
    "S0 raw XTTS  ·  S0′ vocoder round-trip only  ·  S1 copy source prosody  ·  "
    "S2a ProsoMLP  ·  S2b ProsoCNN  ·  S2b-p ProsoCNN pitch only (no retiming)  ·  "
    "S3d oracle via 8 DCT coefficients  ·  S3 oracle full contour  ·  S3-p oracle pitch only"
)
ARM_COLOURS = {
    "S0": "#9aa0a6",
    "S0_identity": "#bdc1c6",
    "S1_copy": "#5f6368",
    "S2_mlp": "#1a73e8",
    "S2_cnn": "#12b5cb",
    "S3_dct": "#f9ab00",
    "S3_full": "#e8710a",
    "S2_cnn_pitch": "#7cb342",
    "S3_full_pitch": "#c0ca33",
}
SYSTEM_ORDER = (
    "B1_train_mean",
    "B0_copy_source",
    "B2_copy_plus_offset",
    "ridge_t0",
    "ridge_t1",
    "ridge_full",
    "ProsoMLP",
    "ProsoCNN",
)
TARGET_LABELS = {
    "f0_mean_st": "Pitch level\n(semitones)",
    "f0_std_st": "Pitch span\n(semitones)",
    "log_dur_ratio": "Duration\n(log ratio)",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def figure_baselines(results_csv: Path, outdir: Path, split: str) -> Path:
    """Grouped bars: every system on every target, both directions. The headline table as a chart."""
    import numpy as np
    import pandas as pd

    plt = _style()
    table = pd.read_csv(results_csv)
    directions = sorted(table["direction"].unique())
    targets = [t for t in TARGET_LABELS if t in set(table["target"])]

    figure, axes = plt.subplots(
        len(directions),
        len(targets),
        figsize=(3.9 * len(targets), 3.1 * len(directions)),
        gridspec_kw={"wspace": 0.12},
    )
    axes = np.atleast_2d(axes)

    for row, direction in enumerate(directions):
        for column, target in enumerate(targets):
            axis = axes[row][column]
            part = table[(table["direction"] == direction) & (table["target"] == target)]
            systems = [s for s in SYSTEM_ORDER if s in set(part["system"])]
            values = [float(part[part["system"] == s]["mae"].iloc[0]) for s in systems]
            b2 = float(part[part["system"] == "B2_copy_plus_offset"]["mae"].iloc[0])
            colours = [
                "#1a73e8"
                if s == "ProsoMLP"
                else "#12b5cb"
                if s == "ProsoCNN"
                else "#f9ab00"
                if s.startswith("ridge")
                else "#9aa0a6"
                for s in systems
            ]
            bars = axis.barh(range(len(systems)), values, color=colours)
            axis.axvline(b2, color="#5f6368", ls="--", lw=1, zorder=3)
            axis.set_yticks(range(len(systems)))
            # Only the leftmost panel carries system names; repeating them collides with the
            # neighbouring panel's bars.
            axis.set_yticklabels(
                [s.replace("_", " ") for s in systems] if column == 0 else [], fontsize=7
            )
            axis.invert_yaxis()
            # B1 can be ~6x the others and would flatten every real difference, so the axis is
            # clipped -- which means a label for a clipped bar must be drawn INSIDE the panel.
            limit = min(max(values), b2 * 1.6) * 1.25
            axis.set_xlim(0, limit)
            for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
                delta = 100.0 * (value - b2) / b2 if b2 else 0.0
                if systems[index] == "B2_copy_plus_offset":
                    continue
                clipped = bar.get_width() > limit * 0.93
                axis.text(
                    limit * 0.97 if clipped else bar.get_width() + limit * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{delta:+.1f}%" + (" \u2192" if clipped else ""),
                    va="center",
                    ha="right" if clipped else "left",
                    fontsize=6.5,
                    color="white" if clipped else "black",
                    fontweight="bold" if clipped else "normal",
                )
            if row == 0:
                axis.set_title(TARGET_LABELS[target], fontsize=9)
            if column == 0:
                axis.set_ylabel(direction, fontweight="bold")
            axis.set_xlabel("mean absolute error (lower is better)", fontsize=7)

    figure.suptitle(
        f"From-scratch models vs every baseline — {split} split "
        f"(dashed line = B2, the published baseline)",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = outdir / f"P1_baselines_{split}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_ladder(summary_csv: Path, outdir: Path) -> Path:
    """The ladder: what each arm achieves, and what it costs."""
    import numpy as np
    import pandas as pd

    plt = _style()
    table = pd.read_csv(summary_csv)
    directions = sorted(table["direction"].unique())
    panels = [
        ("f0_mean_mae_st", "Pitch error vs human target\n(semitones, lower better)", False),
        ("shape_r_mean", "Contour correlation\nwith human target (higher better)", True),
        ("duration_ratio", "Duration / human target\n(1.0 = matched)", True),
    ]
    for metric, label in (
        ("ecapa_mean", "COST: speaker similarity\n(ECAPA cosine, higher better)"),
        ("wer_mean", "COST: intelligibility\n(WER, lower better)"),
    ):
        if metric in table.columns and table[metric].notna().any():
            panels.append((metric, label, metric == "ecapa_mean"))

    figure, axes = plt.subplots(
        len(directions),
        len(panels),
        figsize=(2.9 * len(panels), 3.1 * len(directions)),
        gridspec_kw={"wspace": 0.28},
    )
    axes = np.atleast_2d(axes)

    for row, direction in enumerate(directions):
        part = table[table["direction"] == direction]
        arms = [a for a in ARM_ORDER if a in set(part["arm"])]
        for column, (metric, label, higher_better) in enumerate(panels):
            axis = axes[row][column]
            values = [
                float(part[part["arm"] == a][metric].iloc[0])
                if part[part["arm"] == a][metric].notna().all()
                else np.nan
                for a in arms
            ]
            axis.bar(range(len(arms)), values, color=[ARM_COLOURS[a] for a in arms])
            axis.set_xticks(range(len(arms)))
            axis.set_xticklabels(
                [ARM_SHORT[a] for a in arms], fontsize=6.5, rotation=45, ha="right"
            )
            if metric == "duration_ratio":
                axis.axhline(1.0, color="#202124", ls="--", lw=1)
            for index, value in enumerate(values):
                if not np.isnan(value):
                    axis.text(
                        index,
                        value,
                        f"{value:.3f}" if metric in ("ecapa_mean", "wer_mean") else f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                    )
            if row == 0:
                axis.set_title(label, fontsize=8)
            if column == 0:
                axis.set_ylabel(direction, fontweight="bold")
            _ = higher_better

    figure.suptitle(
        "The evaluation ladder — prosody improves left-to-right; the two right panels are the cost",
        fontsize=10,
    )
    figure.text(0.5, 0.015, ARM_KEY, ha="center", fontsize=6, color="#5f6368")
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    path = outdir / "P2_ladder.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_curves(netdir: Path, outdir: Path) -> Path:
    """Training curves. Proof the models were trained, and that dev selected the checkpoint."""
    import json

    plt = _style()
    files = sorted(netdir.glob("history-*.json"))
    if not files:
        raise FileNotFoundError(f"no history-*.json in {netdir}")

    figure, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 3.0), squeeze=False)
    for axis, path in zip(axes[0], files, strict=True):
        blob = json.loads(path.read_text())
        summary = blob["summary"]
        for index, history in enumerate(blob["runs"]):
            epochs = [h["epoch"] for h in history]
            axis.plot(
                epochs,
                [h["train_loss"] for h in history],
                color="#9aa0a6",
                lw=0.8,
                label="train" if index == 0 else None,
            )
            axis.plot(
                epochs,
                [h["dev_loss"] for h in history],
                color="#1a73e8",
                lw=0.9,
                label="dev" if index == 0 else None,
            )
        best = summary["best_epoch"]
        axis.axvline(
            sum(best) / len(best),
            color="crimson",
            ls="--",
            lw=1,
            label=f"selected (mean epoch {sum(best) / len(best):.0f})",
        )
        axis.set_title(
            f"{summary['arch']} / {summary['direction']}\n"
            f"{summary['n_parameters']:,} parameters, {len(blob['runs'])} seeds",
            fontsize=8,
        )
        axis.set_xlabel("epoch")
        axis.set_ylabel("weighted SmoothL1 loss")
        axis.legend(fontsize=6)

    figure.suptitle("Trained from random initialisation; checkpoint chosen on dev", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = outdir / "P3_training_curves.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_contours(ladder_dir: Path, outdir: Path) -> Path:
    """Real contours, overlaid. The most persuasive figure -- you can see the prosody move.

    Examples are chosen to SPAN the models' behaviour: the best, the median and the worst quartile
    by ProsoMLP's own contour correlation. Picking the three best would misrepresent the method,
    and picking by the oracle's score (an earlier version of this figure) selected cases where the
    oracle did well and the models happened to do badly, which was equally misleading.
    """
    import numpy as np
    import pandas as pd

    from .contours import N_POINTS, extract_contour, resample_contour

    plt = _style()
    prosody = pd.read_csv(ladder_dir / "prosody.csv")
    prosody = prosody[prosody["error"].isna() | (prosody["error"].astype(str) == "")]
    prosody["shape_r"] = pd.to_numeric(prosody["shape_r"], errors="coerce")

    ranked = (
        prosody[prosody["arm"] == "S2_mlp"].dropna(subset=["shape_r"]).sort_values("shape_r")
    )
    if ranked.empty:
        raise ValueError("no S2_mlp rows with a contour correlation")
    picks = [
        ("best case", ranked.iloc[-1]),
        ("median case", ranked.iloc[len(ranked) // 2]),
        ("lower quartile", ranked.iloc[len(ranked) // 4]),
    ]

    show = ["S0", "S1_copy", "S2_mlp", "S2_cnn", "S3_full"]
    figure, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.4), squeeze=False)
    grid = np.linspace(0, 1, N_POINTS)

    for axis, (caption, row) in zip(axes[0], picks, strict=True):
        finite: list[float] = []
        for arm in show:
            match = prosody[
                (prosody["pair_id"] == row["pair_id"])
                & (prosody["direction"] == row["direction"])
                & (prosody["arm"] == arm)
            ]
            if match.empty or not str(match["path"].iloc[0]):
                continue
            track = extract_contour(str(match["path"].iloc[0]))
            got = resample_contour(track["f0_hz"], track["voiced"], N_POINTS)
            if got is None:
                continue
            axis.plot(
                grid,
                got[0],
                color=ARM_COLOURS[arm],
                lw=1.8 if arm.startswith(("S2", "S3")) else 1.0,
                alpha=0.55 if arm == "S0" else 1.0,
                label=ARM_LABELS[arm].replace("\n", " "),
            )
            if arm != "S0":  # S0 carries pyin octave errors and would set an absurd scale
                finite.extend(got[0].tolist())
        # Clip to a plausible speech range so real movement is visible. Raw XTTS measurements reach
        # +/-19 semitones, which is a pitch-tracking failure rather than a voice.
        if finite:
            span = max(4.0, float(np.percentile(np.abs(finite), 98)) * 1.25)
            axis.set_ylim(-span, span)
        axis.set_title(
            f"{caption} — {row['pair_id']} {row['direction']}\n"
            f"ProsoMLP contour r = {row['shape_r']:.2f}",
            fontsize=7.5,
        )
        axis.set_xlabel("normalised time")
        axis.set_ylabel("semitones from median")
        axis.legend(fontsize=6, loc="best")

    figure.suptitle(
        "Measured pitch contours after injection — best, median and lower-quartile cases",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.015,
        "S0 is drawn faded and is clipped: raw XTTS measurements reach ±19 semitones, "
        "which is a pitch-tracking failure, not speech.",
        ha="center",
        fontsize=6,
        color="#5f6368",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.92))
    path = outdir / "P4_contours.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_power(results_csv: Path, outdir: Path) -> Path:
    """Effect size vs the sign test, and the speaker-clustered CI that actually settles it.

    The right panel is the one to trust. The per-speaker signed-rank test only uses the SIGN of
    each speaker's change, so with 13-14 speakers it demands near-unanimity before it will report
    significance. The bootstrap CI resamples speakers and uses the magnitudes, so it is better
    powered on the same data -- and it is what separates the results that hold from the one that
    does not.
    """
    import numpy as np
    import pandas as pd

    plt = _style()
    table = pd.read_csv(results_csv)
    primary = table[table["system"].isin(("ProsoMLP", "ProsoCNN"))].dropna(
        subset=["mae_vs_b2_pct", "wilcoxon_p_vs_b2_by_speaker"]
    )
    speakers = int(table["n_speakers"].dropna().max())

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})

    # ---- left: effect size against the raw sign test
    axis = axes[0]
    for system, colour in (("ProsoMLP", "#1a73e8"), ("ProsoCNN", "#12b5cb")):
        part = primary[primary["system"] == system]
        axis.scatter(
            -part["mae_vs_b2_pct"],
            part["wilcoxon_p_vs_b2_by_speaker"],
            color=colour,
            label=system,
            s=40,
            zorder=3,
        )
    axis.axhline(0.05, color="crimson", ls="--", lw=1, label="p = 0.05")
    floor = 2.0 / 2.0**speakers
    axis.axhline(floor, color="#5f6368", ls=":", lw=1.2)
    axis.text(
        0.5,
        floor * 1.4,
        f"floor with {speakers} speakers ({floor:.1e})\nreached only if EVERY speaker improves",
        fontsize=6.5,
        color="#5f6368",
    )
    axis.set_yscale("log")
    axis.set_xlabel("improvement over B2 (%)")
    axis.set_ylabel("per-speaker Wilcoxon p (raw)")
    axis.set_title("The sign test needs near-unanimity", fontsize=9)
    axis.legend(fontsize=7, loc="upper right")

    # ---- right: forest plot of speaker-clustered bootstrap CIs
    axis = axes[1]
    rows = primary.dropna(subset=["mae_diff_cluster_ci95_low", "mae_diff_cluster_ci95_high"])
    rows = rows.sort_values(["target", "direction", "system"])
    labels, centres, lows, highs, colours = [], [], [], [], []
    for _, row in rows.iterrows():
        low = float(row["mae_diff_cluster_ci95_low"])
        high = float(row["mae_diff_cluster_ci95_high"])
        # Normalise to a percentage of B2 so the three targets share one axis.
        b2 = table[
            (table["direction"] == row["direction"])
            & (table["target"] == row["target"])
            & (table["system"] == "B2_copy_plus_offset")
        ]["mae"]
        scale = float(b2.iloc[0]) if len(b2) else 1.0
        labels.append(f"{row['target']}  {row['direction']}  {row['system'].replace('Proso', '')}")
        centres.append(float(row["mae_vs_b2_pct"]))
        lows.append(100.0 * low / scale)
        highs.append(100.0 * high / scale)
        colours.append("#1a73e8" if row["system"] == "ProsoMLP" else "#12b5cb")

    positions = np.arange(len(labels))
    for index, (low, high, centre, colour) in enumerate(
        zip(lows, highs, centres, colours, strict=True)
    ):
        excludes = high < 0
        axis.plot(
            [low, high],
            [index, index],
            color=colour if excludes else "#9aa0a6",
            lw=3 if excludes else 2,
            solid_capstyle="butt",
        )
        # The marker is the POINT ESTIMATE, not the CI midpoint -- the bootstrap distribution is
        # not symmetric, so the midpoint is not the estimate.
        axis.plot([centre], [index], "o", color=colour if excludes else "#9aa0a6", ms=5)
    axis.axvline(0, color="#202124", lw=1.2)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=6.5)
    axis.invert_yaxis()
    axis.set_xlabel("change in MAE vs B2 (% of B2) — negative is better")
    axis.set_title(
        f"Speaker-clustered bootstrap 95% CI (resampling {speakers} speakers)\n"
        "coloured = CI excludes zero",
        fontsize=9,
    )

    figure.suptitle(
        "Pitch span and duration hold up under speaker-clustered inference; pitch level does not",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = outdir / "P5_power.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def build_all(netdir: Path, ladder_dir: Path, outdir: Path, split: str = "test") -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    attempts = [
        ("baselines", lambda: figure_baselines(netdir / f"results-{split}.csv", outdir, split)),
        ("power", lambda: figure_power(netdir / f"results-{split}.csv", outdir)),
        ("curves", lambda: figure_curves(netdir, outdir)),
        ("ladder", lambda: figure_ladder(ladder_dir / "ladder_summary.csv", outdir)),
        ("contours", lambda: figure_contours(ladder_dir, outdir)),
    ]
    for name, build in attempts:
        try:
            path = build()
            made.append(path)
            print(f"  {name:10s} -> {path}")
        except Exception as exc:  # noqa: BLE001 - a missing input must not lose the other figures
            print(f"  {name:10s} SKIPPED ({type(exc).__name__}: {exc})")
    return made


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_figures")
    sub = parser.add_subparsers(dest="command", required=True)
    every = sub.add_parser("all")
    every.add_argument("--netdir", type=Path, default=Path("artifacts/prosody_net"))
    every.add_argument("--ladder", dest="ladder_dir", type=Path, default=Path("artifacts/ladder"))
    every.add_argument("--outdir", type=Path, default=Path("plots/prosody"))
    every.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args(argv)
    if args.command == "all":
        made = build_all(args.netdir, args.ladder_dir, args.outdir, args.split)
        print(f"\n{len(made)} figures in {args.outdir}")


if __name__ == "__main__":
    main()
