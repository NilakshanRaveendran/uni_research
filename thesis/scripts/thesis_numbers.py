"""Compute every number quoted in the dissertation from the committed result files.

    python thesis/scripts/thesis_numbers.py

Reads:
    results/synthesis_final.csv        870 pipeline rows, prosody re-measured at HEAD
    artifacts/features_wideband.csv    paired human EN/ES prosody, 2,893 pairs
    artifacts/out_wideband/*.json|csv  ridge study outputs (t1 feature set)
    artifacts/finetune/comparison-*.json
    data/splits/dral_manifest_with_text.csv

Writes thesis/numbers.json and prints a readable digest.

F0 is correlated in SEMITONES with a plausibility gate (60-400 Hz), matching
bilingual_voice.metrics.summarize_rows. Correlating in Hz without a gate is the bug that
commit ea7542c fixed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

F0_MIN, F0_MAX = 60.0, 400.0


def st(hz: float) -> float:
    return 12.0 * math.log2(hz / 100.0)


def num(row: dict, key: str):
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def within_between(x, y, groups):
    """Split a pooled Pearson r into a within-group and a between-group component.

    Between: correlate the per-group means (one point per speaker).
    Within:  centre both variables inside each group, then correlate the residuals.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    pooled = float(pearsonr(x, y).statistic)

    gx, gy, xw, yw = [], [], [], []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        gx.append(x[m].mean())
        gy.append(y[m].mean())
        xw.extend(x[m] - x[m].mean())
        yw.extend(y[m] - y[m].mean())
    between = (
        float(pearsonr(gx, gy).statistic)
        if len(gx) >= 3 and np.std(gx) > 1e-12 and np.std(gy) > 1e-12
        else None
    )
    within = (
        float(pearsonr(xw, yw).statistic)
        if len(xw) >= 3 and np.std(xw) > 1e-12 and np.std(yw) > 1e-12
        else None
    )
    # One-way ANOVA decomposition of the predictor: how much of its total variance is
    # differences BETWEEN speakers rather than variation within a speaker's own utterances.
    grand = float(x.mean())
    ss_between = 0.0
    ss_within = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        ss_between += float(m.sum()) * (float(x[m].mean()) - grand) ** 2
        ss_within += float(((x[m] - x[m].mean()) ** 2).sum())
    total_ss = ss_between + ss_within
    return {
        "pooled_r": pooled,
        "within_r": within,
        "between_r": between,
        "n": int(len(x)),
        "n_groups": int(len(gx)),
        "between_speaker_variance_share": (ss_between / total_ss) if total_ss > 0 else None,
    }


def pipeline_numbers(path: Path) -> dict:
    rows = [r for r in csv.DictReader(path.open(newline="", encoding="utf-8-sig"))]
    ok = [r for r in rows if r.get("status") == "ok"]
    out = {
        "attempted": len(rows),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "failure_rate": (len(rows) - len(ok)) / len(rows),
        "directions": {},
    }
    for direction in ("en-es", "es-en"):
        g = [r for r in ok if r["direction"] == direction]
        d: dict = {"n": len(g)}

        # --- identity ---
        for field in ("speaker_similarity", "human_target_speaker_similarity"):
            per_speaker = defaultdict(list)
            vals = []
            for r in g:
                v = num(r, field)
                if v is None:
                    continue
                vals.append(v)
                per_speaker[r["speaker_id"]].append(v)
            d[f"{field}_row_mean"] = float(np.mean(vals))
            d[f"{field}_speaker_mean"] = float(np.mean([np.mean(v) for v in per_speaker.values()]))
            d[f"{field}_n_speakers"] = len(per_speaker)
        d["identity_pct_of_ceiling"] = 100.0 * (
            d["speaker_similarity_speaker_mean"] / d["human_target_speaker_similarity_speaker_mean"]
        )

        # --- intelligibility ---
        for field in ("asr_wer", "tts_intelligibility_wer"):
            per_speaker = defaultdict(list)
            vals = []
            for r in g:
                v = num(r, field)
                if v is None:
                    continue
                vals.append(v)
                per_speaker[r["speaker_id"]].append(v)
            d[f"{field}_row_mean"] = float(np.mean(vals))
            d[f"{field}_speaker_mean"] = float(np.mean([np.mean(v) for v in per_speaker.values()]))

        # --- prosody: generated vs human target, same-speaker pairs only ---
        same = [r for r in g if r.get("same_speaker_pair") == "true"]
        d["prosody_n_same_speaker"] = len(same)
        for feature in ("f0_mean", "f0_std", "energy", "duration", "speaking_rate"):
            gen, ref, spk = [], [], []
            for r in same:
                a = num(r, f"{feature}_generated")
                b = num(r, f"{feature}_target_reference")
                if a is None or b is None:
                    continue
                if feature.startswith("f0"):
                    if a <= 0 or b <= 0:
                        continue
                    if feature == "f0_mean" and not (
                        F0_MIN <= a <= F0_MAX and F0_MIN <= b <= F0_MAX
                    ):
                        continue
                    a, b = st(a), st(b)
                gen.append(a)
                ref.append(b)
                spk.append(r["speaker_id"])
            if len(gen) >= 3:
                d[f"gen_vs_human_{feature}"] = within_between(gen, ref, spk)

        # --- duration inflation ---
        ratios = []
        for r in g:
            a = num(r, "duration_generated")
            b = num(r, "duration_target_reference")
            if a and b:
                ratios.append(a / b)
        d["duration_ratio_generated_over_human"] = {
            "median": float(np.median(ratios)),
            "mean": float(np.mean(ratios)),
            "p90": float(np.percentile(ratios, 90)),
            "n": len(ratios),
            "mean_generated_s": float(
                np.mean([num(r, "duration_generated") for r in g if num(r, "duration_generated")])
            ),
            "mean_human_s": float(
                np.mean(
                    [
                        num(r, "duration_target_reference")
                        for r in g
                        if num(r, "duration_target_reference")
                    ]
                )
            ),
        }
        out["directions"][direction] = d
    return out


def human_numbers(path: Path) -> dict:
    """Human EN-vs-ES prosody correspondence, same quality gate as analysis.load_features."""
    import pandas as pd

    from bilingual_voice.analysis import load_features

    df, stats = load_features(path, same_speaker_only=True)
    out = {"gate_stats": stats, "features": {}}
    pairs = {
        "f0_mean": ("en_f0_mean_st", "es_f0_mean_st"),
        "f0_std": ("en_f0_std_st", "es_f0_std_st"),
        "duration": ("en_duration", "es_duration"),
        "speaking_rate": ("en_speaking_rate", "es_speaking_rate"),
    }
    for name, (a, b) in pairs.items():
        sub = df[[a, b, "en_speaker"]].dropna()
        out["features"][name] = within_between(
            sub[a].to_numpy(float), sub[b].to_numpy(float), sub["en_speaker"].to_numpy()
        )
    ratio = (df["es_duration"] / df["en_duration"]).replace([np.inf, -np.inf], np.nan).dropna()
    out["es_over_en_duration_ratio"] = {
        "median": float(ratio.median()),
        "mean": float(ratio.mean()),
        "sd": float(ratio.std(ddof=1)),
        "n": int(len(ratio)),
    }
    out["n_speakers"] = int(pd.unique(df["en_speaker"]).size)
    return out


def _csv_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))


def corpus_counts(manifest: Path) -> dict:
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8-sig")))
    speakers = set()
    for r in rows:
        speakers.add(r["english_speaker_id"])
        speakers.add(r["spanish_speaker_id"])
    by_split = defaultdict(int)
    for r in rows:
        by_split[r["split"]] += 1
    return {
        "pairs": len(rows),
        "unique_speaker_ids": len(speakers),
        "same_speaker_pairs": sum(1 for r in rows if r["same_speaker_pair"] == "true"),
        "different_speaker_pairs": sum(1 for r in rows if r["same_speaker_pair"] != "true"),
        "by_split": dict(by_split),
        "with_english_text": sum(1 for r in rows if r.get("english_text", "").strip()),
        "with_spanish_text": sum(1 for r in rows if r.get("spanish_text", "").strip()),
    }


def main() -> None:
    out = {
        "corpus": corpus_counts(ROOT / "data/splits/dral_manifest_with_text.csv"),
        "pipeline": pipeline_numbers(ROOT / "results/synthesis_final.csv"),
        "human": human_numbers(ROOT / "artifacts/features_wideband.csv"),
        "finetune": {
            d: json.loads((ROOT / f"artifacts/finetune/comparison-{d}.json").read_text())
            for d in ("en-es", "es-en")
        },
        "finetune_history": {
            d: json.loads((ROOT / f"artifacts/finetune/history-{d}.json").read_text())
            for d in ("en-es", "es-en")
        },
        "ridge_gate": json.loads((ROOT / "artifacts/out_wideband/T1_gate_stats.json").read_text()),
        "provenance": json.loads((ROOT / "artifacts/out_wideband/PROVENANCE.json").read_text()),
        "ridge_results": _csv_rows(ROOT / "artifacts/out_wideband/T4_results.csv"),
        "ridge_coefficients": _csv_rows(ROOT / "artifacts/out_wideband/T5_coefficients.csv"),
        "corpus_table": _csv_rows(ROOT / "artifacts/out_wideband/T3_corpus.csv"),
    }
    # Corpus-level scores come straight from the shipped summariser so the thesis and the CLI
    # can never disagree.
    from bilingual_voice.metrics import summarize_rows

    rows = list(csv.DictReader((ROOT / "results/synthesis_final.csv").open(encoding="utf-8-sig")))
    out["corpus_scores"] = summarize_rows(rows)

    dest = ROOT / "thesis/numbers.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    main()
