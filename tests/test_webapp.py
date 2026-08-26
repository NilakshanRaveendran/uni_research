from pathlib import Path

from bilingual_voice.pipeline import LocalModels
from webapp.backend import main


def test_local_models_accept_finetuned_translation_checkpoints(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best-en-es"
    models = LocalModels(model_root=tmp_path / "models", mt_model_dirs={"en-es": checkpoint})

    assert models.mt_model_dirs["en-es"] == checkpoint.resolve()


def test_web_health_blocks_jobs_until_xtts_license_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)

    health = main.health()

    assert health["xtts_license_accepted"] is False
    assert health["ok"] is False
    assert health["mt_mode"] == "fine-tuned on DRAL"
