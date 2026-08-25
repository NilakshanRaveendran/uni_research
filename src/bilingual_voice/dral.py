"""DRAL release inspection, metadata-linked pairing, and transcript merging."""

from __future__ import annotations

import csv
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FIELDS = [
    "pair_id",
    "english_speaker_id",
    "spanish_speaker_id",
    "same_speaker_pair",
    "english_id",
    "spanish_id",
    "english_audio",
    "spanish_audio",
    "english_text",
    "spanish_text",
    "split",
]


@dataclass(frozen=True)
class DralInspection:
    root: Path
    metadata_files: tuple[Path, ...]
    wav_count: int
    english_wav_count: int
    spanish_wav_count: int


def inspect_dral(root: Path) -> DralInspection:
    root = root.expanduser().resolve()
    metadata = tuple(sorted(root.rglob("fragments-short.csv")))
    wavs = list(root.rglob("*.wav"))
    stems = [path.stem.upper() for path in wavs]
    return DralInspection(
        root=root,
        metadata_files=metadata,
        wav_count=len(wavs),
        english_wav_count=sum(stem.startswith("EN_") for stem in stems),
        spanish_wav_count=sum(stem.startswith("ES_") for stem in stems),
    )


def _first(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _row_id(row: dict[str, str]) -> str:
    return _first(row, ("id", "fragment_id", "audio_id", "", "Unnamed: 0"))


def _load_fragment_rows(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                fragment_id = _row_id(row)
                if not fragment_id:
                    continue
                previous = rows.get(fragment_id)
                if previous and previous != row:
                    raise ValueError(f"Conflicting metadata for fragment {fragment_id}")
                rows[fragment_id] = row
    return rows


def _audio_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*.wav"):
        fragment_id = path.stem
        previous = index.get(fragment_id)
        if previous and previous.resolve() != path.resolve():
            raise ValueError(f"Duplicate audio ID {fragment_id}: {previous} and {path}")
        index[fragment_id] = path.resolve()
    return index


def build_manifest(root: Path) -> list[dict[str, str]]:
    """Build one row per EN/ES pair using official trans_id metadata links."""
    inspection = inspect_dral(root)
    if not inspection.metadata_files:
        raise FileNotFoundError(f"No fragments-short.csv found under {inspection.root}")

    fragments = _load_fragment_rows(inspection.metadata_files)
    audio = _audio_index(inspection.root)
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for fragment_id, row in fragments.items():
        peer_id = row.get("trans_id", "").strip()
        peer = fragments.get(peer_id)
        if not peer:
            continue
        language = row.get("lang_code", fragment_id[:2]).lower()
        peer_language = peer.get("lang_code", peer_id[:2]).lower()
        if {language, peer_language} != {"en", "es"}:
            continue

        english_id, spanish_id = (
            (fragment_id, peer_id) if language == "en" else (peer_id, fragment_id)
        )
        key = (english_id, spanish_id)
        if key in seen:
            continue
        seen.add(key)

        english = fragments[english_id]
        spanish = fragments[spanish_id]
        en_speaker = _first(english, ("participant_id_unique", "participant_id"))
        es_speaker = _first(spanish, ("participant_id_unique", "participant_id"))
        if english_id not in audio or spanish_id not in audio:
            continue

        output.append(
            {
                "pair_id": f"{english_id}__{spanish_id}",
                "english_speaker_id": en_speaker,
                "spanish_speaker_id": es_speaker,
                "same_speaker_pair": str(en_speaker == es_speaker).lower(),
                "english_id": english_id,
                "spanish_id": spanish_id,
                "english_audio": str(audio[english_id]),
                "spanish_audio": str(audio[spanish_id]),
                "english_text": _first(
                    english, ("text", "transcript", "transcription", "orthography")
                ),
                "spanish_text": _first(
                    spanish, ("text", "transcript", "transcription", "orthography")
                ),
                "split": "",
            }
        )

    if not output:
        raise ValueError("Metadata was found, but no complete EN/ES audio pairs were resolved")
    return sorted(output, key=lambda row: row["pair_id"])


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_transcripts(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not {"audio_id", "text"}.issubset(rows[0]):
            raise ValueError(f"Transcript CSV must contain audio_id,text columns: {path}")
        return {Path(row["audio_id"].strip()).stem: row["text"].strip() for row in rows}

    transcripts = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split("|", 2)
        if len(parts) == 2 and len(parts[1]) >= 2:
            # The official ES file currently has one missing delimiter at line 393:
            # ES_010_19.wav|VB¡Andale! Okay
            filename, tail = parts
            parts = [filename, tail[:2], tail[2:]]
            warnings.warn(
                f"Recovered missing annotator delimiter at {path}:{line_number}",
                stacklevel=2,
            )
        if len(parts) != 3:
            raise ValueError(f"Expected filename|annotator|text at {path}:{line_number}")
        filename, _annotator, text = parts
        transcripts[Path(filename.strip()).stem] = text.strip()
    return transcripts


def merge_transcripts(
    manifest_rows: list[dict[str, str]], transcript_paths: Iterable[Path]
) -> list[dict[str, str]]:
    transcripts: dict[str, str] = {}
    for path in transcript_paths:
        incoming = _read_transcripts(path)
        overlap = transcripts.keys() & incoming.keys()
        conflicts = [key for key in overlap if transcripts[key] != incoming[key]]
        if conflicts:
            raise ValueError(f"Conflicting transcripts for IDs: {conflicts[:5]}")
        transcripts.update(incoming)

    merged = []
    for original in manifest_rows:
        row = dict(original)
        row["english_text"] = transcripts.get(row["english_id"], row["english_text"])
        row["spanish_text"] = transcripts.get(row["spanish_id"], row["spanish_text"])
        merged.append(row)
    return merged
