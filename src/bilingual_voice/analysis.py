"""Cross-lingual prosody correspondence: paired feature extraction and supervised mapping.

Two entry points, runnable without touching the existing CLI:

    python -m bilingual_voice.analysis features <manifest.csv> <features.csv> [--jobs N] [--limit N]
    python -m bilingual_voice.analysis report   <features.csv> <outdir>      [--features t0|t1]

`features` is the only expensive step; it is resumable and per-file fault tolerant.
`report` runs in seconds and may be re-run freely.

Note on extraction: this module runs its own single-pass pyin extraction rather than
calling prosody.extract_prosody, because the quality gate needs a voiced-frame ratio and
the F0 targets need a true semitone-domain standard deviation -- neither of which that
function returns. The pyin settings here mirror prosody.py exactly (C2-C7, sr=16 kHz), so
a single extraction method can be described in the write-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dral import read_manifest

# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------

SIDE_FIELDS = (
    "duration",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_mean_st",
    "f0_std_st",
    "voiced_ratio",
    "n_voiced",
    "rms",
    "n_words",
    "speaking_rate",
)


# pyin search range. prosody.py inherited a MUSICAL range (C2-C7, 65-2093 Hz), which is far
# wider than plausible speaking F0 and invites octave-doubling: measured on this corpus, the
# 99th percentile of mean F0 reached 1284 Hz and the max 2048 Hz, which is not speech. Adult
# conversational F0 means sit roughly 80-250 Hz, so the search is narrowed to a speech range.
SPEECH_F0_MIN = 60.0
SPEECH_F0_MAX = 400.0


def _semitones(hz):
    """Convert Hz to semitones relative to a 100 Hz reference."""
    import numpy as np

    return 12.0 * np.log2(np.asarray(hz, dtype=float) / 100.0)


def _extract_side(path: str, transcript: str) -> dict[str, float]:
    """One pyin pass over one wav. Mirrors prosody.py settings."""
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    f0, _, _ = librosa.pyin(
        audio,
        fmin=SPEECH_F0_MIN,
        fmax=SPEECH_F0_MAX,
        sr=sample_rate,
    )
    voiced = f0[~np.isnan(f0)]
    voiced = voiced[voiced > 0]
    rms = librosa.feature.rms(y=audio)[0]
    n_words = len(transcript.split()) if transcript else 0
    nan = float("nan")
    st = _semitones(voiced) if voiced.size else np.array([])
    return {
        "duration": float(duration),
        "f0_mean_hz": float(np.mean(voiced)) if voiced.size else nan,
        "f0_std_hz": float(np.std(voiced)) if voiced.size else nan,
        "f0_mean_st": float(np.mean(st)) if st.size else nan,
        "f0_std_st": float(np.std(st)) if st.size else nan,
        "voiced_ratio": float(voiced.size / len(f0)) if len(f0) else 0.0,
        "n_voiced": float(voiced.size),
        "rms": float(np.mean(rms)) if rms.size else nan,
        "n_words": float(n_words),
        "speaking_rate": float(n_words / duration) if duration and n_words else nan,
    }


def _feature_fields() -> list[str]:
    return (
        ["pair_id", "split", "same_speaker_pair", "en_speaker", "es_speaker"]
        + [f"en_{name}" for name in SIDE_FIELDS]
        + [f"es_{name}" for name in SIDE_FIELDS]
        + ["error"]
    )


def _one_pair(row: dict[str, str]) -> dict[str, str]:
    """Extract both sides of one pair. Never raises -- failures land in `error`."""
    out: dict[str, str] = {
        "pair_id": row["pair_id"],
        "split": row.get("split", ""),
        "same_speaker_pair": row.get("same_speaker_pair", ""),
        "en_speaker": row.get("english_speaker_id", ""),
        "es_speaker": row.get("spanish_speaker_id", ""),
        "error": "",
    }
    for prefix, audio_key, text_key in (
        ("en", "english_audio", "english_text"),
        ("es", "spanish_audio", "spanish_text"),
    ):
        try:
            values = _extract_side(row[audio_key], row.get(text_key, ""))
            for name in SIDE_FIELDS:
                out[f"{prefix}_{name}"] = repr(values[name])
        except Exception as exc:  # noqa: BLE001 - one bad wav must not kill a long run
            out["error"] = f"{prefix}:{type(exc).__name__}: {exc}"[:300]
            for name in SIDE_FIELDS:
                out.setdefault(f"{prefix}_{name}", "")
    return out


def build_feature_table(
    manifest: Path, out_csv: Path, jobs: int = 1, limit: int | None = None
) -> int:
    """Extract paired prosody for every manifest pair. Resumable; appends as it goes."""
    rows = read_manifest(manifest)
    if limit is not None:
        rows = rows[:limit]

    done: set[str] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8-sig") as handle:
            done = {r["pair_id"] for r in csv.DictReader(handle) if r.get("pair_id")}
        print(f"resuming: {len(done)} pairs already done")

    todo = [r for r in rows if r["pair_id"] not in done]
    print(f"{len(todo)} pairs to extract (jobs={jobs})")
    if not todo:
        return 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = _feature_fields()
    new_file = not out_csv.exists()
    written = 0
    with out_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                results = pool.map(_one_pair, todo, chunksize=4)
                for result in results:
                    writer.writerow(result)
                    written += 1
                    if written % 100 == 0:
                        handle.flush()
                        print(f"  {written}/{len(todo)}")
        else:
            for result_row in todo:
                writer.writerow(_one_pair(result_row))
                written += 1
                if written % 100 == 0:
                    handle.flush()
                    print(f"  {written}/{len(todo)}")
    print(f"wrote {written} rows to {out_csv}")
    return written


# --------------------------------------------------------------------------------------
# Quality gate and derived columns
# --------------------------------------------------------------------------------------

MIN_VOICED_RATIO = 0.30
MIN_DURATION = 0.4
MAX_DURATION = 20.0

DIRECTIONS = {"en-es": ("en", "es"), "es-en": ("es", "en")}
# target name -> (unit label, is_ratio_target)
TARGETS = {
    "f0_mean_st": ("semitones", False),
    "f0_std_st": ("semitones", False),
    "log_dur_ratio": ("log-ratio", True),
}
FEATURES_T0 = ["s_f0_mean_st", "s_f0_std_st", "s_log_dur", "s_voiced_ratio"]
FEATURES_T1 = FEATURES_T0 + ["s_n_words", "s_speaking_rate"]


def load_features(path: Path, same_speaker_only: bool = True):
    """Load the feature table, apply the quality gate, and report what was dropped.

    same_speaker_only restricts to pairs where one bilingual speaker performed both sides.
    The methodology is stated in terms of same-speaker re-enactments, so cross-speaker pairs
    are excluded from the primary analysis rather than silently included.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    total = len(df)
    stats = {"rows_extracted": total}

    df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    stats["dropped_extraction_error"] = total - len(df)

    for side in ("en", "es"):
        for name in SIDE_FIELDS:
            df[f"{side}_{name}"] = pd.to_numeric(df[f"{side}_{name}"], errors="coerce")

    before = len(df)
    keep = np.ones(len(df), dtype=bool)
    reasons = {}
    for side in ("en", "es"):
        checks = {
            "nan_f0": df[f"{side}_f0_mean_st"].notna().to_numpy(),
            "voiced_ratio": (df[f"{side}_voiced_ratio"] >= MIN_VOICED_RATIO).to_numpy(),
            "duration": (
                (df[f"{side}_duration"] >= MIN_DURATION) & (df[f"{side}_duration"] <= MAX_DURATION)
            ).to_numpy(),
            # Backstop: a mean F0 outside the plausible speaking band is a tracking failure,
            # not a voice. One such pair contributes ~30x a normal semitone error and would
            # dominate MAE across every system.
            "f0_implausible": (
                (df[f"{side}_f0_mean_hz"] >= SPEECH_F0_MIN)
                & (df[f"{side}_f0_mean_hz"] <= SPEECH_F0_MAX)
            ).to_numpy(),
        }
        for name, ok in checks.items():
            reasons[name] = reasons.get(name, 0) + int((~ok).sum())
            keep &= ok
    df = df[keep]
    stats["dropped_by_criterion_either_side"] = reasons
    edge = 0.05
    at_bound = (
        (df["en_f0_mean_hz"] <= SPEECH_F0_MIN * (1 + edge))
        | (df["en_f0_mean_hz"] >= SPEECH_F0_MAX * (1 - edge))
        | (df["es_f0_mean_hz"] <= SPEECH_F0_MIN * (1 + edge))
        | (df["es_f0_mean_hz"] >= SPEECH_F0_MAX * (1 - edge))
    )
    stats["retained_but_f0_near_search_bound"] = int(at_bound.sum())
    is_same = df["same_speaker_pair"].astype(str).str.lower() == "true"
    stats["same_speaker_pairs"] = int(is_same.sum())
    stats["different_speaker_pairs"] = int((~is_same).sum())
    if same_speaker_only:
        df = df[is_same]
        stats["restricted_to_same_speaker"] = True
        stats["rows_after_same_speaker_filter"] = len(df)
        stats["kept_by_split"] = df["split"].value_counts().to_dict()
    else:
        stats["restricted_to_same_speaker"] = False
    stats["dropped_quality_gate"] = before - len(df)
    stats["rows_kept"] = len(df)
    stats["drop_rate"] = round(1 - len(df) / total, 4) if total else 0.0
    stats["kept_by_split"] = df["split"].value_counts().to_dict()
    stats["same_speaker_kept"] = int((df["same_speaker_pair"] == True).sum()) or int(
        (df["same_speaker_pair"].astype(str).str.lower() == "true").sum()
    )
    stats["gate"] = {
        "min_voiced_ratio": MIN_VOICED_RATIO,
        "min_duration_s": MIN_DURATION,
        "max_duration_s": MAX_DURATION,
    }
    return df.reset_index(drop=True), stats


def directional_frame(df, direction: str):
    """Reshape the symmetric pair table into source/target columns for one direction."""
    import numpy as np
    import pandas as pd

    src, tgt = DIRECTIONS[direction]
    out = pd.DataFrame(
        {
            "pair_id": df["pair_id"].to_numpy(),
            "split": df["split"].to_numpy(),
            "speaker": df[f"{src}_speaker"].astype(str).to_numpy(),
            "s_f0_mean_st": df[f"{src}_f0_mean_st"].to_numpy(),
            "s_f0_std_st": df[f"{src}_f0_std_st"].to_numpy(),
            "s_duration": df[f"{src}_duration"].to_numpy(),
            "s_voiced_ratio": df[f"{src}_voiced_ratio"].to_numpy(),
            "s_n_words": df[f"{src}_n_words"].to_numpy(),
            "s_speaking_rate": df[f"{src}_speaking_rate"].to_numpy(),
            "t_f0_mean_st": df[f"{tgt}_f0_mean_st"].to_numpy(),
            "t_f0_std_st": df[f"{tgt}_f0_std_st"].to_numpy(),
            "t_duration": df[f"{tgt}_duration"].to_numpy(),
        }
    )
    out["s_log_dur"] = np.log(out["s_duration"])
    out["log_dur_ratio"] = np.log(out["t_duration"] / out["s_duration"])
    # "copy the source" prediction for each target
    out["copy_f0_mean_st"] = out["s_f0_mean_st"]
    out["copy_f0_std_st"] = out["s_f0_std_st"]
    out["copy_log_dur_ratio"] = 0.0
    out["target_f0_mean_st"] = out["t_f0_mean_st"]
    out["target_f0_std_st"] = out["t_f0_std_st"]
    out["target_log_dur_ratio"] = out["log_dur_ratio"]
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------------------
# Closed-form ridge (no scikit-learn)
# --------------------------------------------------------------------------------------

# The top of this grid must be large enough that the weights effectively vanish, so that
# B2 is genuinely reachable as ridge's limiting case on the dev set. Without a large enough
# lambda, ridge can end up worse than B2 on test purely through selection variance.
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]


def ridge_fit(X, y, lam: float):
    """Standardised closed-form ridge with an unpenalised intercept."""
    import numpy as np

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0  # drop zero-variance columns from the scaling
    Xs = (X - mu) / sigma
    y_mean = y.mean()
    yc = y - y_mean
    gram = Xs.T @ Xs + max(lam, 1e-6) * np.eye(Xs.shape[1])
    try:
        w = np.linalg.solve(gram, Xs.T @ yc)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(gram, Xs.T @ yc, rcond=None)[0]
    return {"w": w, "mu": mu, "sigma": sigma, "intercept": y_mean, "lam": lam}


def ridge_predict(model, X):
    import numpy as np

    Xs = (np.asarray(X, dtype=float) - model["mu"]) / model["sigma"]
    return Xs @ model["w"] + model["intercept"]


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _errors(pred, truth):
    import numpy as np

    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.abs(pred - truth)


def _scores(pred, truth):
    import numpy as np
    from scipy.stats import pearsonr

    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = pred - truth
    r = None
    if len(truth) >= 3 and np.std(pred) > 1e-12 and np.std(truth) > 1e-12:
        r = float(pearsonr(pred, truth).statistic)
    return {
        "n": len(truth),
        "mae": float(np.mean(np.abs(err))),
        # Median absolute error. A few pairs retain F0-tracking failures pinned at the pyin
        # search bounds; for same-speaker pairs a 20+ semitone cross-language gap is
        # physiologically impossible. Those rows are NOT removed -- filtering by error size
        # would be target-dependent -- so the median is reported as the robust statistic.
        "medae": float(np.median(np.abs(err))),
        "rmse": float(math.sqrt(float(np.mean(err**2)))),
        "pearson_r": r,
    }


def _cluster_stats(err, b2_err, speakers, seed: int = 498, draws: int = 2000):
    """Speaker-clustered inference.

    The test split holds ~357 pairs but only ~14 speakers, so treating each recording as an
    independent observation overstates significance: errors are correlated within a speaker.
    Two corrections are reported instead of the naive per-recording test:
      * a Wilcoxon test on per-speaker mean absolute errors (n = number of speakers), and
      * a bootstrap CI for the MAE difference that resamples SPEAKERS, not recordings.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    frame = pd.DataFrame({"speaker": speakers, "err": err, "b2": b2_err})
    per = frame.groupby("speaker")[["err", "b2"]].mean()
    out = {"n_speakers": len(per)}

    if len(per) >= 6 and np.any(np.abs(per["err"] - per["b2"]) > 1e-12):
        out["wilcoxon_p_vs_b2_by_speaker"] = float(wilcoxon(per["err"], per["b2"]).pvalue)
    else:
        out["wilcoxon_p_vs_b2_by_speaker"] = None

    groups = [g[["err", "b2"]].to_numpy() for _, g in frame.groupby("speaker")]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        sample = np.vstack([groups[i] for i in pick])
        diffs.append(float(sample[:, 0].mean() - sample[:, 1].mean()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out["mae_diff_cluster_ci95_low"] = float(lo)
    out["mae_diff_cluster_ci95_high"] = float(hi)
    # A CI straddling zero means the improvement is not distinguishable from speaker-level noise.
    out["cluster_ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


def _select_lambda(X_train, resid_train, X_dev, y_dev, copy_dev) -> float:
    """One-standard-error rule on dev: the largest lambda that is statistically as good.

    Plain argmin over dev MAE is unstable on a few-hundred-row dev set, and because ridge
    nests B2 (lambda -> inf), an unlucky small lambda makes the model look worse than the
    trivial baseline for reasons that are pure selection noise rather than a real finding.
    Preferring the most-shrunk lambda within one standard error biases toward B2, which is
    the conservative and more defensible choice.
    """
    import numpy as np

    errors = {}
    for lam in LAMBDAS:
        model = ridge_fit(X_train, resid_train, lam)
        errors[lam] = np.abs(copy_dev + ridge_predict(model, X_dev) - y_dev)
    best_lam = min(errors, key=lambda key: float(np.mean(errors[key])))
    best = errors[best_lam]
    if len(best) < 2:
        return best_lam
    threshold = float(np.mean(best)) + float(np.std(best, ddof=1)) / math.sqrt(len(best))
    within = [lam for lam in LAMBDAS if float(np.mean(errors[lam])) <= threshold]
    return max(within) if within else best_lam


def evaluate(df, feature_set: str = "t0"):
    """Fit on train, select lambda on dev, report on test. Returns (results, coefficients)."""
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    features = FEATURES_T0 if feature_set == "t0" else FEATURES_T1
    rows = []
    coefficients = []

    for direction in DIRECTIONS:
        frame = directional_frame(df, direction)
        for target in TARGETS:
            needed = features + [f"target_{target}", f"copy_{target}"]
            work = frame.dropna(subset=needed)
            parts = {s: work[work["split"] == s] for s in ("train", "dev", "test")}
            if min(len(p) for p in parts.values()) < 10:
                print(
                    f"  skip {direction}/{target}: too few rows {[len(p) for p in parts.values()]}"
                )
                continue

            def xy(part, target=target):
                return (
                    part[features].to_numpy(float),
                    part[f"target_{target}"].to_numpy(float),
                    part[f"copy_{target}"].to_numpy(float),
                )

            Xtr, ytr, ctr = xy(parts["train"])
            Xdv, ydv, cdv = xy(parts["dev"])
            Xte, yte, cte = xy(parts["test"])
            work_idx = parts["test"].index

            # --- baselines, all fitted on train only ---
            b1 = float(np.mean(ytr))  # predict train mean
            offset = float(np.mean(ytr - ctr))  # B2's single global scalar
            preds = {
                "B1_train_mean": np.full_like(yte, b1),
                "B0_copy_source": cte,
                "B2_copy_plus_offset": cte + offset,
            }

            # --- ridge on the RESIDUAL from copy-source, lambda chosen on dev ---
            # Fitting the residual makes ridge strictly nest B2: as lambda -> inf the weights
            # vanish and the prediction becomes copy + mean(residual), which IS B2. Fitting the
            # absolute target instead would let shrinkage pull toward the global mean, which is
            # far worse than copying (see B1). So ridge can only improve on B2 up to noise.
            lam = _select_lambda(Xtr, ytr - ctr, Xdv, ydv, cdv)
            model = ridge_fit(Xtr, ytr - ctr, lam)
            preds["ridge"] = cte + ridge_predict(model, Xte)
            best = (None, lam)
            coefficients.append(
                {
                    "direction": direction,
                    "target": target,
                    "lambda": best[1],
                    "intercept": model["intercept"],
                    **{f"w_{name}": float(v) for name, v in zip(features, model["w"])},
                }
            )

            b2_err = _errors(preds["B2_copy_plus_offset"], yte)
            for system, pred in preds.items():
                score = _scores(pred, yte)
                score.update(direction=direction, target=target, system=system)
                score["unit"] = TARGETS[target][0]
                if system != "B2_copy_plus_offset":
                    err = _errors(pred, yte)
                    b2_mae = float(np.mean(b2_err))
                    score["mae_vs_b2_pct"] = (
                        round(100.0 * (score["mae"] - b2_mae) / b2_mae, 2) if b2_mae > 0 else None
                    )
                    # Ridge nests B2, so it can land numerically ON B2. A signed-rank test over
                    # near-identical error vectors returns a tiny p-value for a zero-size effect,
                    # which reads as "significantly better" when nothing happened. Treat
                    # practically-identical predictions as no difference.
                    identical = np.allclose(err, b2_err, rtol=1e-6, atol=1e-9)
                    b2_med = float(np.median(b2_err))
                    score["medae_vs_b2_pct"] = (
                        round(100.0 * (score["medae"] - b2_med) / b2_med, 2) if b2_med > 0 else None
                    )
                    score["wilcoxon_p_vs_b2"] = (
                        1.0 if identical else float(wilcoxon(err, b2_err).pvalue)
                    )
                    score["equivalent_to_b2"] = bool(identical)
                    score.update(
                        _cluster_stats(err, b2_err, parts["test"].loc[work_idx, "speaker"])
                    )
                rows.append(score)

    order = [
        "direction",
        "target",
        "system",
        "unit",
        "n",
        "mae",
        "medae",
        "rmse",
        "pearson_r",
        "mae_vs_b2_pct",
        "medae_vs_b2_pct",
        "wilcoxon_p_vs_b2",
        "n_speakers",
        "wilcoxon_p_vs_b2_by_speaker",
        "mae_diff_cluster_ci95_low",
        "mae_diff_cluster_ci95_high",
        "cluster_ci_excludes_zero",
        "equivalent_to_b2",
    ]
    results = pd.DataFrame(rows)
    return results[[c for c in order if c in results.columns]], pd.DataFrame(coefficients)


def corpus_table(df):
    """Descriptive EN-vs-ES prosody with paired deltas and 95% CIs (Chapter 4, T3)."""
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr

    rows = []
    for label, en_col, es_col in (
        ("F0 mean (st)", "en_f0_mean_st", "es_f0_mean_st"),
        ("F0 range (st)", "en_f0_std_st", "es_f0_std_st"),
        ("Duration (s)", "en_duration", "es_duration"),
        ("Speaking rate (w/s)", "en_speaking_rate", "es_speaking_rate"),
    ):
        sub = df[[en_col, es_col]].dropna()
        en = sub[en_col].to_numpy(float)
        es = sub[es_col].to_numpy(float)
        delta = es - en
        sem = (
            float(np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else float("nan")
        )
        rows.append(
            {
                "measure": label,
                "n": len(sub),
                "en_mean": float(np.mean(en)),
                "en_sd": float(np.std(en, ddof=1)),
                "es_mean": float(np.mean(es)),
                "es_sd": float(np.std(es, ddof=1)),
                "delta_es_minus_en": float(np.mean(delta)),
                "ci95_low": float(np.mean(delta) - 1.96 * sem),
                "ci95_high": float(np.mean(delta) + 1.96 * sem),
                "pearson_r": float(pearsonr(en, es).statistic) if len(sub) >= 3 else None,
            }
        )
    ratio = (df["es_duration"] / df["en_duration"]).replace([np.inf, -np.inf], np.nan).dropna()
    rows.append(
        {
            "measure": "ES/EN duration ratio",
            "n": len(ratio),
            "es_mean": float(np.mean(ratio)),
            "es_sd": float(np.std(ratio, ddof=1)),
            "delta_es_minus_en": float(np.median(ratio)),
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def make_figures(df, results, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig, name):
        path = outdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # F2 -- histogram of log ES/EN duration ratio
    ratio = (
        np.log(df["es_duration"] / df["en_duration"]).replace([np.inf, -np.inf], np.nan).dropna()
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(ratio, bins=60, color="#4477aa", alpha=0.85)
    ax.axvline(
        float(np.median(ratio)),
        color="crimson",
        ls="--",
        lw=1.5,
        label=f"median = {float(np.median(ratio)):.3f}",
    )
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("log(Spanish duration / English duration)")
    ax.set_ylabel("Number of pairs")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "F2_duration_ratio_hist.png")

    # F3 -- EN vs ES F0 mean in semitones
    sub = df[["en_f0_mean_st", "es_f0_mean_st"]].dropna()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(sub["en_f0_mean_st"], sub["es_f0_mean_st"], s=10, alpha=0.4, color="#4477aa")
    low = float(min(sub.min()))
    high = float(max(sub.max()))
    ax.plot([low, high], [low, high], "--", color="gray", lw=1, label="y = x")
    ax.set_xlabel("English F0 mean (semitones re 100 Hz)")
    ax.set_ylabel("Spanish F0 mean (semitones re 100 Hz)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "F3_f0_en_vs_es.png")

    # F4 -- boxplots of paired deltas
    deltas, labels = [], []
    for label, en_col, es_col in (
        ("$\\Delta$F0 mean\n(st)", "en_f0_mean_st", "es_f0_mean_st"),
        ("$\\Delta$F0 range\n(st)", "en_f0_std_st", "es_f0_std_st"),
        ("$\\Delta$rate\n(w/s)", "en_speaking_rate", "es_speaking_rate"),
    ):
        s = df[[en_col, es_col]].dropna()
        deltas.append((s[es_col] - s[en_col]).to_numpy())
        labels.append(label)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.boxplot(deltas, tick_labels=labels, showmeans=True)
    ax.axhline(0, color="crimson", ls="--", lw=1)
    ax.set_ylabel("Spanish minus English")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "F4_paired_deltas_box.png")

    # F5 -- test MAE by system, one panel per target.
    # Separate y-axes are essential: B1's error on F0 mean is ~7x the others, so a single
    # shared axis flattens the B0/B2/ridge comparison -- the only one that matters -- to
    # indistinguishable bars.
    systems = ["B1_train_mean", "B0_copy_source", "B2_copy_plus_offset", "ridge"]
    colours = ["#bbbbbb", "#ee8866", "#44bb99", "#4477aa"]
    if len(results):
        for direction in sorted(results["direction"].unique()):
            part = results[results["direction"] == direction]
            targets = sorted(part["target"].unique())
            fig, axes = plt.subplots(
                1, len(targets), figsize=(4.0 * len(targets), 4.2), squeeze=False
            )
            for ax, target in zip(axes[0], targets, strict=True):
                vals, unit = [], ""
                for system in systems:
                    hit = part[(part["target"] == target) & (part["system"] == system)]
                    vals.append(float(hit["mae"].iloc[0]) if len(hit) else np.nan)
                    if len(hit):
                        unit = str(hit["unit"].iloc[0])
                bars = ax.bar(range(len(systems)), vals, color=colours, edgecolor="black", lw=0.4)
                for rect, value in zip(bars, vals, strict=True):
                    if np.isfinite(value):
                        ax.text(
                            rect.get_x() + rect.get_width() / 2,
                            value,
                            f"{value:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                ax.set_xticks(range(len(systems)))
                ax.set_xticklabels(["B1\nmean", "B0\ncopy", "B2\ncopy+off", "ridge"], fontsize=8)
                ax.set_title(target, fontsize=10)
                ax.set_ylabel(f"Test MAE ({unit})" if unit else "Test MAE")
                ax.margins(y=0.15)
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(f"Direction: {direction} — lower is better", fontsize=11)
            save(fig, f"F5_test_mae_{direction}.png")
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _fmt(table) -> str:
    """Markdown if tabulate is available, plain text otherwise. Neither is a project dep."""
    try:
        return table.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return table.to_string(index=False)


def _report(features_csv: Path, outdir: Path, feature_set: str) -> None:
    from .splits import assert_speaker_disjoint

    outdir.mkdir(parents=True, exist_ok=True)
    df, stats = load_features(features_csv)
    print(json.dumps(stats, indent=2, default=str))

    # Guard: the one error class that would invalidate every result in the thesis.
    try:
        assert_speaker_disjoint(
            [
                {
                    "english_speaker_id": str(r.en_speaker),
                    "spanish_speaker_id": str(r.es_speaker),
                    "split": str(r.split),
                }
                for r in df.itertuples()
            ]
        )
    except ValueError as exc:
        raise SystemExit(
            f"\nFATAL: speaker leakage between splits -- {exc}\n"
            "The same speaker appears in more than one split, so test scores would be\n"
            "contaminated by training data and no result here would be defensible.\n"
            "Fix: rebuild the splits with `bvt split-manifest` (which groups connected\n"
            "speaker IDs before partitioning) and re-run `features` on the new manifest.\n"
        ) from exc
    print("speaker-disjointness: OK")

    corpus = corpus_table(df)
    results, coefficients = evaluate(df, feature_set)

    (outdir / "T1_gate_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    for name, table in (
        ("T3_corpus", corpus),
        ("T4_results", results),
        ("T5_coefficients", coefficients),
    ):
        table.to_csv(outdir / f"{name}.csv", index=False)
        (outdir / f"{name}.md").write_text(_fmt(table) + "\n")
        try:
            # pandas routes to_latex through Styler, which needs jinja2 (not a project dep).
            (outdir / f"{name}.tex").write_text(table.to_latex(index=False, float_format="%.4f"))
        except ImportError:
            print(f"  ({name}: LaTeX export skipped -- pip install jinja2 if you want .tex)")

    # Provenance: every table/figure must be traceable to the exact code and data that made it.
    import platform
    import subprocess

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or "uncommitted"
        )
    except Exception:  # noqa: BLE001 - provenance must never break the report
        commit = "unknown"
    provenance = {
        "features_csv": str(features_csv),
        "feature_set": feature_set,
        "features_used": FEATURES_T0 if feature_set == "t0" else FEATURES_T1,
        "targets": list(TARGETS),
        "directions": list(DIRECTIONS),
        "seed": 498,
        "lambda_grid": LAMBDAS,
        "f0_search_hz": [SPEECH_F0_MIN, SPEECH_F0_MAX],
        "quality_gate": {
            "min_voiced_ratio": MIN_VOICED_RATIO,
            "min_duration_s": MIN_DURATION,
            "max_duration_s": MAX_DURATION,
            "f0_band_hz": [SPEECH_F0_MIN, SPEECH_F0_MAX],
        },
        "same_speaker_only": stats.get("restricted_to_same_speaker"),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "scipy", "librosa", "matplotlib"):
        try:
            provenance[f"{name}_version"] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            provenance[f"{name}_version"] = "unavailable"
    (outdir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    figures = make_figures(df, results, outdir)
    print("\n=== T3 corpus ===")
    print(_fmt(corpus))
    print("\n=== T4 results ===")
    print(_fmt(results))
    print("\nfigures: " + ", ".join(p.name for p in figures))
    print(f"\nall artifacts in {outdir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    feat = sub.add_parser("features")
    feat.add_argument("manifest", type=Path)
    feat.add_argument("output", type=Path)
    feat.add_argument("--jobs", type=int, default=1)
    feat.add_argument("--limit", type=int)

    rep = sub.add_parser("report")
    rep.add_argument("features", type=Path)
    rep.add_argument("outdir", type=Path)
    rep.add_argument("--features", dest="feature_set", choices=("t0", "t1"), default="t0")

    args = parser.parse_args(argv)
    if args.command == "features":
        build_feature_table(args.manifest, args.output, args.jobs, args.limit)
    else:
        _report(args.features, args.outdir, args.feature_set)


if __name__ == "__main__":
    main()
