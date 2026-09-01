"""Generate the dissertation figures that are not already in artifacts/.

    python thesis/scripts/thesis_figures.py

Every value plotted is read from thesis/numbers.json (produced by thesis_numbers.py) or from the
committed result CSVs -- nothing is typed in by hand. Output is vector PDF so the figures stay
sharp at print size, plus PNG copies for slides.

Palette is Okabe-Ito (colour-vision-deficiency safe). Every bar carries a printed value so the
figures survive greyscale printing.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NUM = json.loads((ROOT / "thesis" / "numbers.json").read_text())

BLUE, ORANGE, GREEN, VERMILION, SKY, YELLOW, PURPLE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#CC79A7",
)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"  {stem}.pdf")


def label_bars(ax, bars, fmt="{:.3f}", dy=0.008):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


# ---------------------------------------------------------------------------------------
# 1. Prosody transfer gap -- within-speaker correlation, human ceiling vs system
# ---------------------------------------------------------------------------------------
def fig_prosody_gap() -> None:
    feats = [
        ("f0_mean", "F0 level\n(semitones)"),
        ("f0_std", "F0 range\n(semitones)"),
        ("duration", "Duration\n(s)"),
        ("speaking_rate", "Speaking rate\n(words/s)"),
    ]
    human = [NUM["human"]["features"][k]["within_r"] for k, _ in feats]
    enes = [NUM["pipeline"]["directions"]["en-es"][f"gen_vs_human_{k}"]["within_r"] for k, _ in feats]
    esen = [NUM["pipeline"]["directions"]["es-en"][f"gen_vs_human_{k}"]["within_r"] for k, _ in feats]

    x = np.arange(len(feats))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    b1 = ax.bar(x - w, human, w, label="Human EN$\\leftrightarrow$ES (ceiling)", color=BLUE)
    b2 = ax.bar(x, enes, w, label="System EN$\\rightarrow$ES", color=ORANGE)
    b3 = ax.bar(x + w, esen, w, label="System ES$\\rightarrow$EN", color=GREEN)
    for b in (b1, b2, b3):
        label_bars(ax, b)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in feats])
    ax.set_ylabel("Within-speaker Pearson $r$ with the human target")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    save(fig, "F_prosody_gap")


# ---------------------------------------------------------------------------------------
# 2. Pooled vs within-speaker -- how much of the pooled r is just speaker identity
# ---------------------------------------------------------------------------------------
def fig_pooled_vs_within() -> None:
    rows = [
        ("Human\nF0 level", NUM["human"]["features"]["f0_mean"]),
        ("Human\nDuration", NUM["human"]["features"]["duration"]),
        ("EN$\\rightarrow$ES\nF0 level", NUM["pipeline"]["directions"]["en-es"]["gen_vs_human_f0_mean"]),
        ("ES$\\rightarrow$EN\nF0 level", NUM["pipeline"]["directions"]["es-en"]["gen_vs_human_f0_mean"]),
        ("EN$\\rightarrow$ES\nDuration", NUM["pipeline"]["directions"]["en-es"]["gen_vs_human_duration"]),
        ("ES$\\rightarrow$EN\nDuration", NUM["pipeline"]["directions"]["es-en"]["gen_vs_human_duration"]),
    ]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    b1 = ax.bar(x - w / 2, [r[1]["pooled_r"] for r in rows], w, label="Pooled", color=SKY)
    b2 = ax.bar(x + w / 2, [r[1]["within_r"] for r in rows], w, label="Within-speaker", color=VERMILION)
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "F_pooled_vs_within")


# ---------------------------------------------------------------------------------------
# 3. Speaker identity against the human cross-language ceiling
# ---------------------------------------------------------------------------------------
def fig_identity_ceiling() -> None:
    dirs = [("en-es", "EN$\\rightarrow$ES"), ("es-en", "ES$\\rightarrow$EN")]
    ceiling = [NUM["pipeline"]["directions"][d]["human_target_speaker_similarity_speaker_mean"] for d, _ in dirs]
    system = [NUM["pipeline"]["directions"][d]["speaker_similarity_speaker_mean"] for d, _ in dirs]
    pct = [NUM["pipeline"]["directions"][d]["identity_pct_of_ceiling"] for d, _ in dirs]

    x = np.arange(len(dirs))
    w = 0.34
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    b1 = ax.bar(x - w / 2, ceiling, w, label="Human cross-language anchor (ceiling)", color=BLUE)
    b2 = ax.bar(x + w / 2, system, w, label="Synthesised speech", color=ORANGE)
    label_bars(ax, b1, dy=0.004)
    label_bars(ax, b2, dy=0.004)
    for i, p in enumerate(pct):
        ax.text(i + w / 2, system[i] / 2, f"{p:.1f}%\nof ceiling", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in dirs])
    ax.set_ylabel("ECAPA cosine similarity to the source recording")
    ax.set_ylim(0, 0.56)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "F_identity_ceiling")


# ---------------------------------------------------------------------------------------
# 4. Duration inflation of raw synthesised speech
# ---------------------------------------------------------------------------------------
def fig_duration_ratio() -> None:
    rows = [r for r in csv.DictReader((ROOT / "results/synthesis_final.csv").open(encoding="utf-8-sig"))
            if r["status"] == "ok"]
    ratios = np.array([float(r["duration_generated"]) / float(r["duration_target_reference"]) for r in rows])
    med = float(np.median(ratios))
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.hist(np.clip(ratios, 0, 6), bins=60, color=BLUE, alpha=0.85)
    ax.axvline(1.0, color="0.35", ls=":", lw=1.2, label="1.0 = matches the human target")
    ax.axvline(med, color=VERMILION, ls="--", lw=1.5, label=f"median = {med:.3f}")
    ax.set_xlabel("Synthesised duration / human target duration")
    ax.set_ylabel("Number of utterances")
    ax.legend(frameon=False)
    ax.text(0.98, 0.55, f"$n$ = {len(ratios)}\n{100*float(np.mean(ratios>2)):.1f}% exceed 2.0$\\times$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    save(fig, "F_duration_ratio")


# ---------------------------------------------------------------------------------------
# 5. Fine-tuning: pretrained vs fine-tuned
# ---------------------------------------------------------------------------------------
def fig_finetune() -> None:
    dirs = [("en-es", "EN$\\rightarrow$ES"), ("es-en", "ES$\\rightarrow$EN")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, metric, title in zip(axes, ("bleu", "chrf"), ("BLEU", "chrF")):
        pre = [NUM["finetune"][d]["pretrained"][metric] for d, _ in dirs]
        fin = [NUM["finetune"][d]["finetuned"][metric] for d, _ in dirs]
        x = np.arange(len(dirs))
        w = 0.35
        b1 = ax.bar(x - w / 2, pre, w, label="Pretrained", color=SKY)
        b2 = ax.bar(x + w / 2, fin, w, label="Fine-tuned on DRAL", color=VERMILION)
        label_bars(ax, b1, fmt="{:.2f}", dy=0.3)
        label_bars(ax, b2, fmt="{:.2f}", dy=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in dirs])
        ax.set_title(title)
        ax.set_ylim(0, max(fin + pre) * 1.25)
        if metric == "bleu":
            ax.set_ylabel("Corpus score on the held-out test split")
            ax.legend(frameon=False, loc="upper left")
    save(fig, "F_finetune_bleu_chrf")


# ---------------------------------------------------------------------------------------
# 6. Cascade error propagation
# ---------------------------------------------------------------------------------------
def fig_cascade() -> None:
    dirs = [("en-es", "EN$\\rightarrow$ES"), ("es-en", "ES$\\rightarrow$EN")]
    clean = [NUM["finetune"][d]["pretrained"]["bleu"] for d, _ in dirs]
    cascade = [NUM["corpus_scores"]["directions"][d]["corpus_bleu"] for d, _ in dirs]
    wer = [NUM["corpus_scores"]["directions"][d]["corpus_asr_wer"] for d, _ in dirs]
    x = np.arange(len(dirs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    b1 = ax.bar(x - w / 2, clean, w, label="Translating the human transcript", color=GREEN)
    b2 = ax.bar(x + w / 2, cascade, w, label="Translating the ASR output (full cascade)", color=PURPLE)
    label_bars(ax, b1, fmt="{:.2f}", dy=0.3)
    label_bars(ax, b2, fmt="{:.2f}", dy=0.3)
    for i in range(len(dirs)):
        ax.annotate(f"$-${clean[i]-cascade[i]:.2f} BLEU\n(ASR WER {wer[i]:.3f})",
                    xy=(i, min(clean[i], cascade[i]) / 2), ha="center", va="center",
                    fontsize=8, color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in dirs])
    ax.set_ylabel("Corpus BLEU on the test split")
    ax.set_ylim(0, max(clean) * 1.3)
    ax.legend(frameon=False, loc="upper center")
    save(fig, "F_cascade_error")


# ---------------------------------------------------------------------------------------
# 7. Copy the figures already produced by the analysis modules
# ---------------------------------------------------------------------------------------
def copy_existing() -> None:
    for src, dest in [
        ("artifacts/out_wideband/F2_duration_ratio_hist.png", "F_corpus_duration_ratio.png"),
        ("artifacts/out_wideband/F3_f0_en_vs_es.png", "F_corpus_f0_scatter.png"),
        ("artifacts/out_wideband/F4_paired_deltas_box.png", "F_corpus_paired_deltas.png"),
        ("artifacts/out_wideband/F5_test_mae_en-es.png", "F_ridge_mae_en-es.png"),
        ("artifacts/out_wideband/F5_test_mae_es-en.png", "F_ridge_mae_es-en.png"),
        ("artifacts/finetune/F6_finetune_curves_en-es.png", "F_finetune_curves_en-es.png"),
        ("artifacts/finetune/F6_finetune_curves_es-en.png", "F_finetune_curves_es-en.png"),
    ]:
        s = ROOT / src
        if s.exists():
            shutil.copy2(s, FIG / dest)
            print(f"  {dest} (copied)")


if __name__ == "__main__":
    fig_prosody_gap()
    fig_pooled_vs_within()
    fig_identity_ceiling()
    fig_duration_ratio()
    fig_finetune()
    fig_cascade()
    copy_existing()
    print(f"figures in {FIG}")
