"""Re-measure system-versus-human F0 range the way Section 3 defines it.

    python thesis/scripts/remeasure_f0_range.py

Why this exists. Table 4.4's system "F0 range" row (0.035 / 0.040 within-speaker) was computed by
`thesis_numbers.py` from the `f0_std` column of results/synthesis_final.csv, which is a standard
deviation in HERTZ, passed through 12*log2(sd/100), and never gated. The human column uses the
standard deviation of the voiced frames in SEMITONES (analysis._extract_side), gated on mean F0.
The two are different quantities. This script measures the system side the same way as the human
side: pyin with the shared search band, semitone SD over voiced frames, and the 60-400 Hz gate on
the utterance mean F0 of both recordings.

Writes results/f0_range_remeasure.csv (one row per same-speaker utterance) and
results/f0_range_remeasure.json (pooled / within / between per direction).
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from thesis_numbers import within_between

from bilingual_voice.analysis import _extract_side
from bilingual_voice.prosody import F0_PLAUSIBLE_MAX, F0_PLAUSIBLE_MIN

SYNTHESIS = ROOT / "results" / "synthesis_final.csv"
OUT_CSV = ROOT / "results" / "f0_range_remeasure.csv"
OUT_JSON = ROOT / "results" / "f0_range_remeasure.json"


def _measure(row: dict) -> dict:
    gen = _extract_side(row["generated_audio"], "")
    ref = _extract_side(row["target_reference_audio"], "")
    return {
        "sample_id": row["sample_id"],
        "speaker_id": row["speaker_id"],
        "direction": row["direction"],
        "f0_mean_hz_generated": gen["f0_mean_hz"],
        "f0_mean_hz_target_reference": ref["f0_mean_hz"],
        "f0_std_st_generated": gen["f0_std_st"],
        "f0_std_st_target_reference": ref["f0_std_st"],
    }


def _plausible(hz: float) -> bool:
    return np.isfinite(hz) and F0_PLAUSIBLE_MIN <= hz <= F0_PLAUSIBLE_MAX


def main() -> None:
    with SYNTHESIS.open(newline="", encoding="utf-8") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r["status"] == "ok" and r.get("same_speaker_pair") == "true"
        ]
    with ProcessPoolExecutor() as pool:
        measured = list(pool.map(_measure, rows, chunksize=8))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)

    summary = {}
    for direction in ("en-es", "es-en"):
        gen, ref, spk = [], [], []
        for m in measured:
            if m["direction"] != direction:
                continue
            if not (_plausible(m["f0_mean_hz_generated"]) and _plausible(m["f0_mean_hz_target_reference"])):
                continue
            a, b = m["f0_std_st_generated"], m["f0_std_st_target_reference"]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            gen.append(a)
            ref.append(b)
            spk.append(m["speaker_id"])
        summary[direction] = within_between(gen, ref, spk)

    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for direction, s in summary.items():
        print(
            f"{direction}: pooled r = {s['pooled_r']:.3f}, within r = {s['within_r']:.3f}, "
            f"between r = {s['between_r']:.3f}, n = {s['n']}, speakers = {s['n_groups']}"
        )


if __name__ == "__main__":
    main()
