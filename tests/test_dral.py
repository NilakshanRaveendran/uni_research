import csv
from pathlib import Path

from bilingual_voice.dral import build_manifest, merge_transcripts


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_uses_trans_id_not_filename_guess(tmp_path: Path) -> None:
    fragments = tmp_path / "release" / "fragments-short"
    fragments.mkdir(parents=True)
    (fragments / "EN_001_2.10.wav").write_bytes(b"RIFF")
    (fragments / "ES_001_1.10.wav").write_bytes(b"RIFF")
    fields = ["id", "participant_id_unique", "lang_code", "trans_id"]
    _write_csv(
        tmp_path / "release" / "fragments-short.csv",
        fields,
        [
            {
                "id": "EN_001_2.10",
                "participant_id_unique": "speaker-1",
                "lang_code": "EN",
                "trans_id": "ES_001_1.10",
            },
            {
                "id": "ES_001_1.10",
                "participant_id_unique": "speaker-1",
                "lang_code": "ES",
                "trans_id": "EN_001_2.10",
            },
        ],
    )
    rows = build_manifest(tmp_path)
    assert len(rows) == 1
    assert rows[0]["english_id"] == "EN_001_2.10"
    assert rows[0]["spanish_id"] == "ES_001_1.10"
    assert rows[0]["english_speaker_id"] == "speaker-1"
    assert rows[0]["spanish_speaker_id"] == "speaker-1"
    assert rows[0]["same_speaker_pair"] == "true"


def test_merge_transcripts_by_audio_id(tmp_path: Path) -> None:
    manifest = [
        {
            "english_id": "EN_1",
            "spanish_id": "ES_1",
            "english_text": "",
            "spanish_text": "",
        }
    ]
    transcripts = tmp_path / "transcripts.csv"
    _write_csv(
        transcripts,
        ["audio_id", "text"],
        [
            {"audio_id": "EN_1", "text": "Hello"},
            {"audio_id": "ES_1", "text": "Hola"},
        ],
    )
    merged = merge_transcripts(manifest, [transcripts])
    assert merged[0]["english_text"] == "Hello"
    assert merged[0]["spanish_text"] == "Hola"


def test_merge_official_pipe_transcripts(tmp_path: Path) -> None:
    manifest = [
        {
            "english_id": "EN_1",
            "spanish_id": "ES_1",
            "english_text": "",
            "spanish_text": "",
        }
    ]
    transcripts = tmp_path / "EN_transcripts.txt"
    transcripts.write_text("EN_1.wav|NW|Hello there.\n", encoding="utf-8")
    merged = merge_transcripts(manifest, [transcripts])
    assert merged[0]["english_text"] == "Hello there."
