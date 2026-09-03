"""Frame-level F0 contour extraction and DCT parameterisation.

    python -m bilingual_voice.contours extract <manifest.csv> <outdir> [--jobs N] [--limit N]
    python -m bilingual_voice.contours check   <outdir> [--n-coef 8]

This is the shared input layer for the from-scratch prosody models. `analysis.py` measures four
scalars per utterance, which is all the ridge baseline can use; a neural predictor needs the
*shape* of the pitch track, so this module stores the whole frame-level contour.

Deliberately a separate module rather than a new subcommand inside `analysis.py`: that file
produces the submitted thesis numbers, and this work must not be able to perturb them. The pyin
settings are imported from `prosody.py` so all three measurement paths stay comparable.

Design notes that are load-bearing:

* SEARCH band 65-1000 Hz, PLAUSIBILITY gate 60-400 Hz. Narrowing the search band suppresses pyin's
  voicing decision on quiet files -- measured, and it was the worst of four options tried.
* Everything is in semitones, never Hertz. Correlating F0 in Hertz once made 0.820 look like 0.048.
* `level` is the MEDIAN of voiced frames, not the mean: pyin still emits occasional octave
  doublings inside the wide search band, and a median ignores them.
* The contour is interpolated across unvoiced gaps so it can be resampled to a fixed length. Those
  interpolated values are a *feature-side* convenience only -- the injector never writes pitch onto
  a frame that was not already voiced, so invented pitch inside a gap can never reach the audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dral import read_manifest
from .prosody import F0_PLAUSIBLE_MAX, F0_PLAUSIBLE_MIN, F0_SEARCH_MAX, F0_SEARCH_MIN

SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_POINTS = 64  # contour resampled to this many normalised-time points
N_COEF = 8  # DCT coefficients kept (k=1..8; k=0 is the level, modelled separately)
SIDES = ("en", "es")


def semitones(hz):
    """Hz -> semitones relative to 100 Hz. Same reference as prosody.py and metrics.py."""
    import numpy as np

    return 12.0 * np.log2(np.asarray(hz, dtype=float) / 100.0)


def extract_contour(path: str) -> dict:
    """One pyin pass over one wav, keeping every frame rather than summary statistics."""
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    f0, voiced_flag, _prob = librosa.pyin(
        audio,
        fmin=F0_SEARCH_MIN,
        fmax=F0_SEARCH_MAX,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    voiced = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0) & (np.nan_to_num(f0) > 0)
    return {
        "f0_hz": np.asarray(f0, dtype=np.float32),
        "voiced": voiced,
        "rms": np.asarray(rms[: len(f0)], dtype=np.float32),
        "duration": np.float32(len(audio) / sample_rate if sample_rate else 0.0),
        "sr": np.int32(sample_rate),
        "hop": np.int32(HOP_LENGTH),
    }


def resample_contour(f0_hz, voiced, n_points: int = N_POINTS):
    """Fixed-length normalised-time semitone contour.

    Returns (shape, level_st, voiced_ratio) where `shape` is `n_points` semitone values with the
    median level removed, or None when the track has too little voicing to describe a contour.
    """
    import numpy as np

    f0_hz = np.asarray(f0_hz, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    n_frames = len(f0_hz)
    if n_frames == 0:
        return None

    usable = voiced & np.isfinite(f0_hz) & (f0_hz >= F0_PLAUSIBLE_MIN) & (f0_hz <= F0_PLAUSIBLE_MAX)
    if usable.sum() < 3:
        return None

    frame_index = np.arange(n_frames, dtype=float)
    voiced_st = semitones(f0_hz[usable])
    # np.interp clamps beyond the voiced span rather than extrapolating, which is what we want at
    # the edges: hold the first/last voiced value instead of inventing a trend.
    dense = np.interp(frame_index, frame_index[usable], voiced_st)
    grid = np.linspace(0.0, n_frames - 1.0, n_points)
    contour = np.interp(grid, frame_index, dense)

    level = float(np.median(voiced_st))
    return contour - level, level, float(usable.sum() / n_frames)


def dct_encode(shape, n_coef: int = N_COEF):
    """Orthonormal DCT-II, dropping k=0. k=0 is the mean offset, carried by `level` instead."""
    import numpy as np
    from scipy.fft import dct

    coefs = dct(np.asarray(shape, dtype=float), type=2, norm="ortho")
    return coefs[1 : 1 + n_coef]


def dct_decode(coefs, n_points: int = N_POINTS):
    """Inverse of dct_encode. The result has exactly zero mean by construction."""
    import numpy as np
    from scipy.fft import idct

    coefs = np.asarray(coefs, dtype=float)
    full = np.zeros(n_points, dtype=float)
    full[1 : 1 + len(coefs)] = coefs
    return idct(full, type=2, norm="ortho")


# --------------------------------------------------------------------------------------
# Extraction over the corpus
# --------------------------------------------------------------------------------------


def _one_pair(job: tuple[dict[str, str], str]) -> str:
    """Extract both sides of one pair into one .npz. Never raises."""
    import numpy as np

    row, outdir = job
    pair_id = row["pair_id"]
    target = Path(outdir) / f"{pair_id}.npz"
    if target.exists():
        return "skip"
    payload: dict[str, object] = {}
    try:
        for side, audio_key in (("en", "english_audio"), ("es", "spanish_audio")):
            for name, value in extract_contour(row[audio_key]).items():
                payload[f"{side}_{name}"] = value
    except Exception as exc:  # noqa: BLE001 - one bad wav must not end a long run
        return f"error {pair_id}: {type(exc).__name__}: {exc}"[:200]
    # Write to a temp name and rename, so a killed run never leaves a half-written .npz that a
    # later resume would trust and skip.
    # Pass an open handle, not a path: np.savez_compressed appends ".npz" to any path that does
    # not already end in it, which would silently write ".npz.tmp.npz" and break the rename.
    tmp = target.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(target)
    return "ok"


def build_contours(manifest: Path, outdir: Path, jobs: int = 1, limit: int | None = None) -> dict:
    """Extract contours for every manifest pair. Resumable: existing .npz files are skipped."""
    from concurrent.futures import ProcessPoolExecutor

    rows = read_manifest(manifest)
    if limit is not None:
        rows = rows[:limit]
    outdir.mkdir(parents=True, exist_ok=True)

    todo = [(row, str(outdir)) for row in rows if not (outdir / f"{row['pair_id']}.npz").exists()]
    print(f"{len(rows)} pairs, {len(todo)} to extract (jobs={jobs})", flush=True)

    counts = {"ok": 0, "skip": 0, "error": 0}
    errors: list[str] = []

    def record(status: str, done: int) -> None:
        if status.startswith("error"):
            counts["error"] += 1
            if len(errors) < 20:
                errors.append(status)
        else:
            counts[status] += 1
        if done % 200 == 0:
            print(f"  {done}/{len(todo)}", flush=True)

    if todo and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for done, status in enumerate(pool.map(_one_pair, todo, chunksize=4), 1):
                record(status, done)
    else:
        for done, job in enumerate(todo, 1):
            record(_one_pair(job), done)

    summary = {
        "pairs": len(rows),
        "extracted": counts["ok"],
        "already_present": counts["skip"],
        "failed": counts["error"],
        "errors": errors,
        "outdir": str(outdir),
        "n_files": len(list(outdir.glob("*.npz"))),
    }
    print(
        f"\n{summary['n_files']} .npz files in {outdir} "
        f"({counts['ok']} new, {counts['error']} failed)"
    )
    for err in errors:
        print(f"  {err}")
    return summary


def check_reconstruction(
    outdir: Path, n_coef: int = N_COEF, n_points: int = N_POINTS, limit: int = 400
) -> dict:
    """How much of the contour survives the DCT truncation. Week-1 gate: median r > 0.9."""
    import numpy as np

    files = sorted(outdir.glob("*.npz"))[:limit]
    if not files:
        raise FileNotFoundError(f"no .npz files in {outdir}")

    correlations: list[float] = []
    rmses: list[float] = []
    usable = 0
    for path in files:
        with np.load(path) as data:
            for side in SIDES:
                got = resample_contour(data[f"{side}_f0_hz"], data[f"{side}_voiced"], n_points)
                if got is None:
                    continue
                shape, _level, _ratio = got
                if not np.isfinite(shape).all() or np.std(shape) < 1e-9:
                    continue
                back = dct_decode(dct_encode(shape, n_coef), n_points)
                usable += 1
                correlations.append(float(np.corrcoef(shape, back)[0, 1]))
                rmses.append(float(np.sqrt(np.mean((shape - back) ** 2))))

    out = {
        "files_checked": len(files),
        "contours_usable": usable,
        "n_coef": n_coef,
        "n_points": n_points,
        "r_median": float(np.median(correlations)) if correlations else float("nan"),
        "r_mean": float(np.mean(correlations)) if correlations else float("nan"),
        "r_p10": float(np.percentile(correlations, 10)) if correlations else float("nan"),
        "rmse_median_st": float(np.median(rmses)) if rmses else float("nan"),
    }
    print(
        f"DCT k=1..{n_coef} over {n_points} points, {usable} contours from {len(files)} files:\n"
        f"  correlation with the full contour  median={out['r_median']:.4f} "
        f"mean={out['r_mean']:.4f} p10={out['r_p10']:.4f}\n"
        f"  reconstruction error               median={out['rmse_median_st']:.4f} semitones"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.contours")
    sub = parser.add_subparsers(dest="command", required=True)

    ext = sub.add_parser("extract")
    ext.add_argument("manifest", type=Path)
    ext.add_argument("outdir", type=Path)
    ext.add_argument("--jobs", type=int, default=1)
    ext.add_argument("--limit", type=int)

    chk = sub.add_parser("check")
    chk.add_argument("outdir", type=Path)
    chk.add_argument("--n-coef", type=int, default=N_COEF)
    chk.add_argument("--n-points", type=int, default=N_POINTS)
    chk.add_argument("--limit", type=int, default=400)

    args = parser.parse_args(argv)
    if args.command == "extract":
        build_contours(args.manifest, args.outdir, args.jobs, args.limit)
    else:
        check_reconstruction(args.outdir, args.n_coef, args.n_points, args.limit)


if __name__ == "__main__":
    main()
