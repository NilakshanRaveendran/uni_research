"""Re-measure the prosody columns of an existing results CSV from the audio on disk.

    python -m bilingual_voice.remeasure results/synthesis_test.csv results/synthesis_fixed.csv --jobs 6

Why this exists: `prosody.extract_prosody` originally used the musical C2-C7 pyin band
(65-2093 Hz). Adult conversational F0 means sit around 80-250 Hz, so that ceiling let pyin lock onto
octave-doubled harmonics -- measured generated F0 spanned -7.3 to 49.4 semitones, which is not
speech. The band is now 60-400 Hz, but every prosody value already written to
`results/synthesis_test.csv` was computed with the old band and is therefore contaminated.

Re-synthesis is NOT needed: all 870 generated wavs already exist. This module re-runs only the
measurement step over the three audio roles per row (source, generated, target reference) and
rewrites the prosody columns, leaving every other column untouched.

Unique audio files are measured once and cached, because each pair's two recordings appear as both
source and target across the two directions -- roughly halving the work.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .pipeline import PROSODY_COLUMNS, RESULT_FIELDS

ROLES = (
    ("source", "source_audio"),
    ("generated", "generated_audio"),
    ("target_reference", "target_reference_audio"),
)


def _measure(path: str) -> tuple[str, dict[str, float] | None]:
    """Measure one file. Returns (path, features) or (path, None) if unreadable."""
    from .prosody import extract_prosody

    if not path or not os.path.exists(path):
        return path, None
    try:
        # transcript="" so speaking_rate comes back NaN; the CSV's word counts are unchanged and
        # speaking rate is recomputed below from the text already stored in the row.
        return path, extract_prosody(Path(path), "")
    except Exception:  # noqa: BLE001 - one bad file must not lose the whole pass
        return path, None


def _speaking_rate(text: str, duration: float | None) -> float:
    words = len((text or "").split())
    if not words or not duration:
        return float("nan")
    return words / duration


def remeasure(in_csv: Path, out_csv: Path, jobs: int = 1) -> dict:
    rows = list(csv.DictReader(in_csv.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"no rows in {in_csv}")

    wanted: set[str] = set()
    for row in rows:
        for _role, column in ROLES:
            if row.get(column):
                wanted.add(row[column])
    paths = sorted(wanted)
    print(f"{len(rows)} rows -> {len(paths)} unique audio files to measure (jobs={jobs})", flush=True)

    cache: dict[str, dict[str, float] | None] = {}
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, (path, feats) in enumerate(pool.map(_measure, paths, chunksize=4), 1):
                cache[path] = feats
                if i % 200 == 0:
                    print(f"  {i}/{len(paths)}", flush=True)
    else:
        for i, path in enumerate(paths, 1):
            cache[path] = _measure(path)[1]
            if i % 200 == 0:
                print(f"  {i}/{len(paths)}", flush=True)

    missing = sum(1 for v in cache.values() if v is None)
    updated = 0
    # Which stored text corresponds to each role, for recomputing speaking rate.
    role_text = {
        "source": "source_reference_text",
        "generated": "translated_text",
        "target_reference": "translation_reference",
    }
    for row in rows:
        touched = False
        for role, column in ROLES:
            feats = cache.get(row.get(column, ""))
            if feats is None:
                continue
            for stem, key in PROSODY_COLUMNS:
                name = f"{stem}_{role}"
                if name not in RESULT_FIELDS:
                    continue
                if stem == "speaking_rate":
                    value = _speaking_rate(row.get(role_text[role], ""), feats.get("duration"))
                else:
                    value = feats[key]
                row[name] = str(value)
            touched = True
        updated += int(touched)

    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in RESULT_FIELDS} for row in rows)
    os.replace(tmp, out_csv)

    summary = {
        "rows": len(rows),
        "unique_files": len(paths),
        "unreadable_files": missing,
        "rows_updated": updated,
        "output": str(out_csv),
    }
    print(f"\nwrote {out_csv}  ({updated}/{len(rows)} rows updated, {missing} files unreadable)")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.remeasure")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    remeasure(args.input, args.output, args.jobs)


if __name__ == "__main__":
    main()
