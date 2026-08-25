import csv
from pathlib import Path

import pytest

from bilingual_voice.pipeline import (
    PROSODY_COLUMNS,
    RESULT_FIELDS,
    LocalModels,
    _write_results,
    read_results,
    sample_seed,
)


def test_xtts_requires_explicit_license_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    models = LocalModels(model_root=tmp_path)

    with pytest.raises(RuntimeError, match="Coqui model license"):
        _ = models.tts


def test_every_prosody_column_written_is_readable_by_summary_and_plots() -> None:
    # metrics.summarize_rows and plots.create_plots both read "{stem}_{role}".
    # A stem written under any other convention is silently dropped from results.
    assert [stem for stem, _ in PROSODY_COLUMNS] == [
        "f0_mean",
        "f0_std",
        "energy",
        "duration",
        "speaking_rate",
    ]
    for stem, _ in PROSODY_COLUMNS:
        for role in ("source", "generated", "target_reference"):
            assert f"{stem}_{role}" in RESULT_FIELDS


def test_sample_seed_is_stable_across_processes_and_varies_by_direction() -> None:
    # Pinned: hash() is salted per process, so it must never be used here.
    assert sample_seed("EN_001_10__ES_001_10", "en-es") == 922575599
    assert sample_seed("EN_001_10__ES_001_10", "es-en") == 1750702897


def test_results_write_replaces_file_and_leaves_no_temp_behind(tmp_path: Path) -> None:
    target = tmp_path / "metrics.csv"
    _write_results([{"sample_id": "pair-a", "direction": "en-es", "status": "ok"}], target)
    _write_results([{"sample_id": "pair-b", "direction": "es-en", "status": "ok"}], target)

    rows = read_results(target)
    assert [row["sample_id"] for row in rows] == ["pair-b"]
    assert list(tmp_path.iterdir()) == [target]


def test_results_write_preserves_previous_checkpoint_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "metrics.csv"
    _write_results([{"sample_id": "pair-a", "direction": "en-es", "status": "ok"}], target)

    def explode(self: csv.DictWriter, rows: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(csv.DictWriter, "writerows", explode)
    with pytest.raises(OSError):
        _write_results([{"sample_id": "pair-b", "direction": "es-en", "status": "ok"}], target)

    assert [row["sample_id"] for row in read_results(target)] == ["pair-a"]
