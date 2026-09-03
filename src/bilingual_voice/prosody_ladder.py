"""The evaluation ladder: impose each system's predicted prosody on the audio XTTS already made.

    python -m bilingual_voice.prosody_ladder render  results/synthesis_test.csv --jobs 6
    python -m bilingual_voice.prosody_ladder measure artifacts/ladder --wer-sample 200
    python -m bilingual_voice.prosody_ladder report  artifacts/ladder

Nothing here re-synthesises speech. The 868 generated wavs already on disk are re-pitched and
re-timed, which is what makes the whole comparison affordable -- and it isolates the prosody
component from every other part of the pipeline, because the words, the voice and the vocoder are
identical across arms.

THE ARMS, and why each one has to be there

    S0            raw XTTS, untouched
    S0_identity   WORLD analysed and resynthesised with NO edit
    S1_copy       B2's prediction imposed: copy the source's prosody plus a global offset
    S2_mlp        ProsoMLP's prediction imposed
    S2_cnn        ProsoCNN's prediction imposed
    S3_dct        the human target's OWN measured prosody, through the 8-coefficient DCT bottleneck
    S3_full       the human target's own measured prosody at full contour resolution

S0_identity is not optional. An identity WORLD round-trip alone moves measured F0 by a
non-trivial amount on real speech, and that footprint is the same order as the effect being
chased. Without it there is no way to tell the model's contribution from the vocoder's damage.

S1_copy, not S0, is the baseline the models must beat: every injected arm shares the same vocoder
path, so its artefacts are common-mode and cancel in the S2 - S1 comparison.

S3_full is the ceiling. It answers "is the predictor weak, or is F0 injection through a frozen TTS
simply unable to deliver more than this?" -- and S3_dct splits the representation's cost out of
that answer.

Every arm is measured with `analysis._extract_side`, the same extractor that produced the thesis
numbers, so pitch is in semitones and directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from . import analysis
from .contours import N_POINTS, dct_decode
from .mt_words import synthesis_pair_id
from .prosody_apply import LEVEL_CAP_ST
from .prosody_net import (
    DCT_OFFSET,
    LEVEL_MED,
    SCALAR_TARGETS,
    load_dataset,
    load_ensemble,
    predict,
    standardise,
)

# Arms ending in "_pitch" impose pitch ONLY and leave the timing alone. They exist because the
# measured cost of injection decomposes as: vocoder round-trip -0.028 ECAPA, TIME-SCALING -0.053,
# and the models' own pitch prediction +0.002 (i.e. nothing). Since retiming is what actually
# damages speaker identity, a pitch-only variant tells a user whether the duration match is worth
# paying for. Added after seeing that decomposition -- it explains an observed cost rather than
# making a new attempt at the headline, and changes no reported model comparison.
ARMS = (
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
PITCH_ONLY_SUFFIX = "_pitch"
LEVEL, SPAN, LOGDUR = 0, 1, 2  # column layout, see prosody_net.SCALAR_TARGETS
N_SCALARS_OUT = len(SCALAR_TARGETS)
MEASURE_FIELDS = (
    "pair_id",
    "direction",
    "arm",
    "path",
    "f0_mean_st",
    "f0_std_st",
    "duration",
    "voiced_ratio",
    "f0_mean_hz",
    "shape_r",
    "target_f0_mean_st",
    "target_f0_std_st",
    "target_duration",
    "requested_level_st",
    "requested_span_st",
    "requested_duration",
    "level_shift_requested_st",
    "level_capped",
    "error",
)


# --------------------------------------------------------------------------------------
# Stage A: render every arm and measure its prosody (parallel, no ML models loaded)
# --------------------------------------------------------------------------------------


def _render_one(job: dict) -> dict:
    """Render and measure one (row, arm). Never raises."""
    import numpy as np
    import soundfile

    out: dict[str, object] = {
        "pair_id": job["pair_id"],
        "direction": job["direction"],
        "arm": job["arm"],
        "path": "",
        "error": "",
        "target_f0_mean_st": job.get("truth_level", ""),
        "target_f0_std_st": job.get("truth_span", ""),
        "target_duration": job.get("truth_duration", ""),
        "requested_level_st": job.get("level", ""),
        "requested_span_st": job.get("span", ""),
        "requested_duration": job.get("duration", ""),
        "level_shift_requested_st": "",
        "level_capped": "",
    }
    target_path = Path(job["out_path"])
    try:
        if job["arm"] == "S0":
            # Nothing to render: measure the original file in place.
            target_path = Path(job["source_wav"])
        elif not target_path.exists():
            from .prosody_apply import inject

            audio, sr = soundfile.read(job["source_wav"], dtype="float64", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # The source wav's own robust level and duration were measured ONCE in the pre-pass
            # (the same file backs all seven arms), so the shift uses one estimator on both sides.
            current_level = float(job.get("current_level", float("nan")))
            current_duration = float(job.get("current_duration", 0.0) or 0.0)

            factor = None
            if job.get("duration") and current_duration > 0:
                factor = float(job["duration"]) / current_duration

            shape = np.asarray(job["shape"], dtype=float) if job.get("shape") is not None else None
            requested = job.get("level")
            edited = inject(
                audio,
                int(sr),
                target_level_st=requested,
                target_shape=shape,
                duration_factor=factor,
                current_level_st=None if math.isnan(current_level) else current_level,
                target_span_st=job.get("span"),
            )
            if requested is not None and not math.isnan(current_level):
                out["level_shift_requested_st"] = float(requested) - current_level
                out["level_capped"] = bool(
                    abs(float(requested) - current_level) > LEVEL_CAP_ST + 1e-9
                )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = target_path.with_suffix(".tmp.wav")
            soundfile.write(tmp, np.asarray(edited, dtype=np.float32), int(sr))
            os.replace(tmp, target_path)

        # Headline scalars come from the SAME extractor that produced the thesis numbers (hop 512),
        # so this table is directly comparable with the published B2 and ridge results.
        measured = analysis._extract_side(str(target_path), "")
        out.update(
            path=str(target_path),
            f0_mean_st=measured["f0_mean_st"],
            f0_std_st=measured["f0_std_st"],
            duration=measured["duration"],
            voiced_ratio=measured["voiced_ratio"],
            f0_mean_hz=measured["f0_mean_hz"],
        )

        # A second, finer pass (hop 160) for the contour SHAPE, which the scalars cannot capture.
        # This is the metric that actually tests whether a contour was transferred.
        truth_shape = job.get("truth_shape")
        if truth_shape is not None:
            from .contours import extract_contour, resample_contour

            track = extract_contour(str(target_path))
            got = resample_contour(track["f0_hz"], track["voiced"], N_POINTS)
            if got is not None:
                shape_out, _level, _ratio = got
                truth = np.asarray(truth_shape, dtype=float)
                if np.std(shape_out) > 1e-9 and np.std(truth) > 1e-9:
                    out["shape_r"] = float(np.corrcoef(shape_out, truth)[0, 1])
    except Exception as exc:  # noqa: BLE001 - one bad row must not end the pass
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def _measure_source(path: str) -> tuple[str, float, float]:
    """Robust level and duration for one generated wav. Measured once, reused by every arm."""
    from .contours import extract_contour, resample_contour

    try:
        track = extract_contour(path)
        duration = float(track["duration"])
        got = resample_contour(track["f0_hz"], track["voiced"], N_POINTS)
        return path, (float("nan") if got is None else float(got[1])), duration
    except Exception:  # noqa: BLE001 - a bad file must not end the pre-pass
        return path, float("nan"), 0.0


def _jobs_for_direction(
    direction: str,
    rows: list[dict],
    netdir: Path,
    outdir: Path,
    arms: tuple[str, ...],
) -> tuple[list[dict], dict]:
    """Turn each test row into one job per arm. Every arm is an 11-vector of predictions."""
    import numpy as np

    data = load_dataset(direction, netdir)
    std = standardise(data)
    test = data["split"] == "test"
    index = {pid: i for i, pid in enumerate(data["pair_id"][test])}

    vectors: dict[str, np.ndarray] = {"S1_copy": std["b2"][test]}
    for arch, label in (("mlp", "S2_mlp"), ("cnn", "S2_cnn")):
        if not any(a.startswith(label) for a in arms):
            continue
        try:
            models, _blob = load_ensemble(arch, direction, netdir)
        except FileNotFoundError:
            print(f"  [{direction}] no {arch} checkpoint; skipping {label}")
            continue
        vectors[label] = np.mean([predict(model, std, test) for model in models], axis=0)
    truth = data["raw_targets"][test]
    vectors["S3_dct"] = truth
    vectors["S3_full"] = truth
    # A pitch-only arm reuses its parent's prediction vector; only the timing handling differs.
    for arm in arms:
        if arm.endswith(PITCH_ONLY_SUFFIX):
            parent = arm[: -len(PITCH_ONLY_SUFFIX)]
            if parent in vectors:
                vectors[arm] = vectors[parent]

    source_duration = np.exp(std_source_log_dur(data)[test])
    target_shape_full = data["target_shape"][test]

    jobs: list[dict] = []
    skipped = {"not_in_dataset": 0, "no_generated_audio": 0}
    for row in rows:
        if row.get("direction") != direction:
            continue
        pair_id = synthesis_pair_id(row)
        position = index.get(pair_id)
        if position is None:
            skipped["not_in_dataset"] += 1
            continue
        generated = row.get("generated_audio", "")
        if not generated or not Path(generated).exists():
            skipped["no_generated_audio"] += 1
            continue

        common = {
            "pair_id": pair_id,
            "direction": direction,
            "source_wav": generated,
            "truth_level": float(truth[position, LEVEL]),
            "truth_span": float(truth[position, SPAN]),
            "truth_duration": float(source_duration[position] * math.exp(truth[position, LOGDUR])),
            # The human target's real contour, for the shape correlation. Measured, never predicted,
            # and identical across arms so the comparison is paired.
            "truth_shape": target_shape_full[position].tolist(),
        }
        for arm in arms:
            job = dict(common, arm=arm, out_path=str(outdir / arm / f"{pair_id}_{direction}.wav"))
            if arm in ("S0", "S0_identity"):
                job["level"] = None
                job["shape"] = None
                job["duration"] = None
                job["span"] = None
            else:
                vector = vectors.get(arm)
                if vector is None:
                    continue
                # LEVEL_MED, not LEVEL: injection is driven by the robust frame-gated median,
                # because f0_mean_st is a mean over the whole 65-1000 Hz search band and a few
                # octave-doubled frames put it tens of semitones off.
                job["level"] = float(vector[position, LEVEL_MED])
                # SPAN is the model's best-supported prediction; without this the injector would
                # impose the contour's form and throw its amplitude away.
                job["span"] = float(vector[position, SPAN])
                pitch_only = arm.endswith(PITCH_ONLY_SUFFIX)
                job["duration"] = (
                    None
                    if pitch_only
                    else float(source_duration[position] * math.exp(vector[position, LOGDUR]))
                )
                uses_full_contour = arm.startswith("S3_full")
                job["shape"] = (
                    target_shape_full[position].tolist()
                    if uses_full_contour
                    else dct_decode(vector[position, DCT_OFFSET:], N_POINTS).tolist()
                )
            jobs.append(job)
    return jobs, skipped


def std_source_log_dur(data):
    """`s_log_dur` column of the cached feature matrix, by name rather than by a magic index."""
    from .prosody_net import INPUT_SCALARS

    return data["scalars"][:, INPUT_SCALARS.index("s_log_dur")]


def render(
    synthesis_csv: Path,
    outdir: Path,
    netdir: Path,
    jobs: int = 1,
    arms: tuple[str, ...] = ARMS,
    limit: int | None = None,
) -> dict:
    """Render and measure every arm. Resumable: an existing wav is measured, not regenerated."""
    from concurrent.futures import ProcessPoolExecutor

    rows = list(csv.DictReader(synthesis_csv.open(newline="", encoding="utf-8-sig")))
    outdir.mkdir(parents=True, exist_ok=True)

    all_jobs: list[dict] = []
    skipped: dict[str, dict] = {}
    for direction in analysis.DIRECTIONS:
        direction_jobs, direction_skipped = _jobs_for_direction(
            direction, rows, netdir, outdir, arms
        )
        if limit:
            direction_jobs = direction_jobs[: limit * len(arms)]
        all_jobs.extend(direction_jobs)
        skipped[direction] = direction_skipped
    print(f"{len(all_jobs)} (row, arm) jobs across {len(arms)} arms (jobs={jobs})", flush=True)
    print(f"  skipped: {json.dumps(skipped)}", flush=True)

    # Pre-pass: measure each unique generated wav once rather than once per arm.
    unique = sorted({job["source_wav"] for job in all_jobs})
    print(f"  pre-measuring {len(unique)} unique source wavs", flush=True)
    cache: dict[str, tuple[float, float]] = {}
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for path, level, duration in pool.map(_measure_source, unique, chunksize=4):
                cache[path] = (level, duration)
    else:
        for path in unique:
            _p, level, duration = _measure_source(path)
            cache[path] = (level, duration)
    unusable = sum(1 for level, _d in cache.values() if math.isnan(level))
    print(f"  {unusable}/{len(unique)} source wavs had no usable pitch level", flush=True)
    for job in all_jobs:
        level, duration = cache.get(job["source_wav"], (float("nan"), 0.0))
        job["current_level"] = level
        job["current_duration"] = duration

    results: list[dict] = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, result in enumerate(pool.map(_render_one, all_jobs, chunksize=4), 1):
                results.append(result)
                if done % 200 == 0:
                    print(f"  {done}/{len(all_jobs)}", flush=True)
    else:
        for done, job in enumerate(all_jobs, 1):
            results.append(_render_one(job))
            if done % 200 == 0:
                print(f"  {done}/{len(all_jobs)}", flush=True)

    out_csv = outdir / "prosody.csv"
    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASURE_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in MEASURE_FIELDS} for r in results)
    os.replace(tmp, out_csv)

    failures = sum(1 for r in results if r.get("error"))
    print(f"\nwrote {out_csv}  ({len(results)} rows, {failures} failed)")
    return {"rows": len(results), "failed": failures, "skipped": skipped, "output": str(out_csv)}


# --------------------------------------------------------------------------------------
# Stage B: what the injection COSTS -- speaker identity and intelligibility
# --------------------------------------------------------------------------------------

QUALITY_FIELDS = ("pair_id", "direction", "arm", "ecapa", "wer", "asr_text", "error")


def measure(
    outdir: Path,
    synthesis_csv: Path,
    wer_sample: int = 200,
    model_root: Path = Path("models"),
) -> dict:
    """Speaker similarity on every rendered file; intelligibility on a paired subsample.

    ECAPA is cheap, so every arm of every row gets it. Whisper is not: transcribing all seven arms
    of 868 rows would take hours, so WER is measured on a random sample of PAIR IDS, with all arms
    of a sampled pair measured. Sampling pairs rather than rows keeps the comparison paired -- every
    arm is scored on exactly the same utterances -- and the sample size is recorded in the output.
    """
    import numpy as np

    from .metrics import normalized_wer
    from .pipeline import LocalModels

    rendered = list(csv.DictReader((outdir / "prosody.csv").open(newline="", encoding="utf-8-sig")))
    if not rendered:
        raise ValueError(f"no rows in {outdir / 'prosody.csv'}")

    # Resume: Whisper is the expensive part, so rows already scored are carried forward untouched.
    # Safe only because the render step skips wavs that already exist -- an arm present in a prior
    # quality.csv was measured against the same audio file that is still on disk. Delete
    # quality.csv (or the arm's directory) to force a re-measure.
    previous: dict[tuple[str, str, str], dict] = {}
    quality_csv = outdir / "quality.csv"
    if quality_csv.exists():
        with quality_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("error"):
                    previous[(row["pair_id"], row["direction"], row["arm"])] = row
        print(f"resuming: {len(previous)} rows already scored", flush=True)

    reference = {
        (synthesis_pair_id(row), row["direction"]): row
        for row in csv.DictReader(synthesis_csv.open(newline="", encoding="utf-8-sig"))
    }

    # Paired WER sample: choose pair ids per direction with a fixed seed, then score every arm.
    rng = np.random.default_rng(498)
    wer_keys: set[tuple[str, str]] = set()
    for direction in analysis.DIRECTIONS:
        pairs = sorted({r["pair_id"] for r in rendered if r["direction"] == direction})
        if not pairs:
            continue
        take = min(wer_sample, len(pairs))
        chosen = rng.choice(np.array(pairs), size=take, replace=False)
        wer_keys.update((str(p), direction) for p in chosen)
    print(f"WER subsample: {len(wer_keys)} (pair, direction) keys x {len(ARMS)} arms", flush=True)

    models: LocalModels | None = None
    results: list[dict] = []
    for done, row in enumerate(rendered, 1):
        cached = previous.get((row["pair_id"], row["direction"], row["arm"]))
        if cached is not None:
            results.append({k: cached.get(k, "") for k in QUALITY_FIELDS})
            continue
        if models is None:  # loaded lazily so a fully-resumed run costs nothing
            models = LocalModels(whisper_size="small", mt_device="cpu", model_root=model_root)
        record = {
            "pair_id": row["pair_id"],
            "direction": row["direction"],
            "arm": row["arm"],
            "ecapa": "",
            "wer": "",
            "asr_text": "",
            "error": "",
        }
        source = reference.get((row["pair_id"], row["direction"]))
        path = row.get("path", "")
        if not source or not path or not Path(path).exists() or row.get("error"):
            record["error"] = "missing input"
            results.append(record)
            continue
        try:
            # Similarity is measured against the SOURCE recording, exactly as the thesis does it.
            record["ecapa"] = models.speaker_similarity(
                Path(source["source_audio"]), Path(path)
            )
            if (row["pair_id"], row["direction"]) in wer_keys:
                language = row["direction"].split("-")[1]
                hypothesis = models.transcribe(Path(path), language)
                record["asr_text"] = hypothesis
                # Reference is what XTTS was asked to say, so this measures whether the injection
                # damaged intelligibility -- not whether the translation was good.
                record["wer"] = normalized_wer(source.get("translated_text", ""), hypothesis)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        results.append(record)
        if done % 100 == 0:
            print(f"  {done}/{len(rendered)}", flush=True)
    fresh = len(rendered) - len(
        [r for r in results if (r["pair_id"], r["direction"], r["arm"]) in previous]
    )

    out_csv = outdir / "quality.csv"
    tmp = out_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in QUALITY_FIELDS} for r in results)
    os.replace(tmp, out_csv)
    print(f"\nwrote {out_csv} ({len(results)} rows, {fresh} newly measured)")
    return {
        "rows": len(results),
        "newly_measured": fresh,
        "reused": len(results) - fresh,
        "wer_keys": len(wer_keys),
        "output": str(out_csv),
    }


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def report(outdir: Path, synthesis_csv: Path | None = None, baseline: str = "S1_copy") -> dict:
    """The seven-arm table. This is the presentation."""
    import numpy as np
    import pandas as pd

    prosody = pd.read_csv(outdir / "prosody.csv")
    quality_path = outdir / "quality.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else None

    prosody = prosody[prosody["error"].isna() | (prosody["error"].astype(str) == "")]
    for column in ("f0_mean_st", "f0_std_st", "duration", "shape_r", "target_f0_mean_st",
                   "target_f0_std_st", "target_duration"):
        prosody[column] = pd.to_numeric(prosody[column], errors="coerce")

    speaker_map: dict[tuple[str, str], str] = {}
    if synthesis_csv and synthesis_csv.exists():
        for row in csv.DictReader(synthesis_csv.open(newline="", encoding="utf-8-sig")):
            speaker_map[(synthesis_pair_id(row), row.get("direction", ""))] = row.get(
                "speaker_id", ""
            )
    prosody["speaker"] = [
        speaker_map.get((p, d), "") for p, d in zip(prosody["pair_id"], prosody["direction"], strict=True)
    ]

    rows = []
    for direction in sorted(prosody["direction"].dropna().unique()):
        part = prosody[prosody["direction"] == direction]
        for arm in ARMS:
            arm_rows = part[part["arm"] == arm].dropna(
                subset=["f0_mean_st", "target_f0_mean_st"]
            )
            if arm_rows.empty:
                continue
            got, want = arm_rows["f0_mean_st"].to_numpy(), arm_rows["target_f0_mean_st"].to_numpy()
            span_got = arm_rows["f0_std_st"].to_numpy()
            span_want = arm_rows["target_f0_std_st"].to_numpy()

            # Within-speaker correlation: remove each speaker's own mean first. The pooled figure
            # is inflated by between-speaker range (men vs women), which no model predicted.
            frame = pd.DataFrame({"s": arm_rows["speaker"].to_numpy(), "g": got, "t": want})
            centred = frame.groupby("s")[["g", "t"]].transform(lambda v: v - v.mean())
            within = (
                float(np.corrcoef(centred["g"], centred["t"])[0, 1])
                if len(centred) > 3
                and centred["g"].std() > 1e-9
                and centred["t"].std() > 1e-9
                else None
            )

            record = {
                "direction": direction,
                "arm": arm,
                "n": len(arm_rows),
                "f0_mean_mae_st": float(np.mean(np.abs(got - want))),
                "f0_std_mae_st": float(np.nanmean(np.abs(span_got - span_want))),
                "f0_r_pooled": float(np.corrcoef(got, want)[0, 1]) if len(got) > 3 else None,
                "f0_r_within_speaker": within,
                "shape_r_mean": float(arm_rows["shape_r"].mean(skipna=True)),
                "duration_ratio": float(
                    np.nanmean(arm_rows["duration"] / arm_rows["target_duration"])
                ),
                "n_speakers": int(arm_rows["speaker"].nunique()),
            }
            if quality is not None:
                q = quality[(quality["direction"] == direction) & (quality["arm"] == arm)]
                ecapa = pd.to_numeric(q["ecapa"], errors="coerce").dropna()
                wer = pd.to_numeric(q["wer"], errors="coerce").dropna()
                record["ecapa_mean"] = float(ecapa.mean()) if len(ecapa) else None
                record["wer_mean"] = float(wer.mean()) if len(wer) else None
                record["wer_n"] = len(wer)
            rows.append(record)

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "ladder_summary.csv", index=False)

    print(f"\n=== the ladder: every arm, {len(prosody)} rendered files ===")
    print("baseline for the model comparison is", baseline, "(same vocoder path, so it cancels)\n")
    for direction in table["direction"].unique():
        part = table[table["direction"] == direction]
        base = part[part["arm"] == baseline]
        base_mae = float(base["f0_mean_mae_st"].iloc[0]) if not base.empty else None
        # If copying the source's prosody is WORSE than just round-tripping the audio, then S1 is
        # a weak baseline and "% better than S1" flatters the models. Report against whichever of
        # the two non-model arms is stronger, and say which one that is.
        identity = part[part["arm"] == "S0_identity"]
        oracle = part[part["arm"] == "S3_full"]
        identity_mae = float(identity["f0_mean_mae_st"].iloc[0]) if not identity.empty else None
        oracle_mae = float(oracle["f0_mean_mae_st"].iloc[0]) if not oracle.empty else None
        fair_mae, fair_name = base_mae, baseline
        if base_mae is not None and identity_mae is not None and identity_mae < base_mae:
            fair_mae, fair_name = identity_mae, "S0_identity"
            print(
                f"{direction}  NOTE: copying source prosody is WORSE than an identity round-trip "
                f"here ({base_mae:.4f} vs {identity_mae:.4f}).\n"
                f"  Naive prosody transfer hurts in this direction, so the fair baseline is "
                f"S0_identity and '% vs S1' overstates the gain."
            )
        else:
            print(f"{direction}")
        header = (
            f"  {'arm':13s} {'n':>4s} {'f0 MAE':>8s} {'vs base':>8s} {'span MAE':>9s} "
            f"{'r pool':>7s} {'r within':>9s} {'shape r':>8s} {'dur':>6s}"
        )
        if "ecapa_mean" in part.columns:
            header += f" {'ecapa':>6s} {'WER':>6s}"
        print(header)
        for _, row in part.iterrows():
            delta = (
                f"{100.0 * (row['f0_mean_mae_st'] - base_mae) / base_mae:+7.1f}%"
                if base_mae
                else "      —"
            )
            line = (
                f"  {row['arm']:13s} {row['n']:>4d} {row['f0_mean_mae_st']:>8.4f} {delta:>8s} "
                f"{row['f0_std_mae_st']:>9.4f} "
                f"{_num(row['f0_r_pooled']):>7s} {_num(row['f0_r_within_speaker']):>9s} "
                f"{_num(row['shape_r_mean']):>8s} {row['duration_ratio']:>6.3f}"
            )
            if "ecapa_mean" in part.columns:
                line += f" {_num(row.get('ecapa_mean')):>6s} {_num(row.get('wer_mean')):>6s}"
            print(line)

        # What fraction of the achievable gain (fair baseline -> oracle) each model captured.
        if fair_mae is not None and oracle_mae is not None and fair_mae != oracle_mae:
            print(f"  share of the achievable gain (vs {fair_name}, ceiling = S3_full):")
            for arm in ("S2_mlp", "S2_cnn"):
                row = part[part["arm"] == arm]
                if row.empty:
                    continue
                value = float(row["f0_mean_mae_st"].iloc[0])
                print(
                    f"    {arm:8s} {100 * (value - fair_mae) / fair_mae:+6.1f}% vs {fair_name}"
                    f"   = {100 * (value - fair_mae) / (oracle_mae - fair_mae):3.0f}% of the way "
                    f"to the oracle"
                )
        print()

    print(f"wrote {outdir / 'ladder_summary.csv'}")
    return {"table": table}


def _num(value) -> str:
    import math as _math

    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if _math.isnan(number) else f"{number:.3f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_ladder")
    sub = parser.add_subparsers(dest="command", required=True)

    ren = sub.add_parser("render")
    ren.add_argument("synthesis_csv", type=Path)
    ren.add_argument("--outdir", type=Path, default=Path("artifacts/ladder"))
    ren.add_argument("--netdir", type=Path, default=Path("artifacts/prosody_net"))
    ren.add_argument("--jobs", type=int, default=1)
    ren.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ren.add_argument("--limit", type=int, help="rows per direction, for smoke tests")

    mea = sub.add_parser("measure")
    mea.add_argument("outdir", type=Path)
    mea.add_argument("synthesis_csv", type=Path)
    mea.add_argument("--wer-sample", dest="wer_sample", type=int, default=200)

    rep = sub.add_parser("report")
    rep.add_argument("outdir", type=Path)
    rep.add_argument("--synthesis-csv", dest="synthesis_csv", type=Path)
    rep.add_argument("--baseline", default="S1_copy", choices=list(ARMS))

    args = parser.parse_args(argv)
    if args.command == "render":
        render(
            args.synthesis_csv,
            args.outdir,
            args.netdir,
            args.jobs,
            tuple(args.arms),
            args.limit,
        )
    elif args.command == "measure":
        measure(args.outdir, args.synthesis_csv, args.wer_sample)
    else:
        report(args.outdir, args.synthesis_csv, args.baseline)


if __name__ == "__main__":
    main()
