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


def test_language_check_accepts_matching_or_inconclusive_speech() -> None:
    from webapp.backend.dubbing import check_language

    check_language("es", 0.97, "es-en")
    check_language("ta", 0.30, "es-en")  # too uncertain to reject (e.g. a music intro)


def test_language_check_rejects_unsupported_and_wrong_direction() -> None:
    import pytest

    from webapp.backend.dubbing import UnsupportedLanguageError, check_language

    with pytest.raises(UnsupportedLanguageError, match="Tamil.*Only English and Spanish"):
        check_language("ta", 0.92, "es-en")
    with pytest.raises(UnsupportedLanguageError, match="Choose English → Spanish"):
        check_language("en", 0.95, "es-en")


def test_link_validation_and_friendly_errors() -> None:
    from webapp.backend.fetch import _friendly, valid_url

    assert valid_url("https://www.youtube.com/watch?v=abc")
    assert not valid_url("youtube.com/watch?v=abc")
    assert not valid_url("file:///etc/passwd")
    assert "needs a login" in _friendly("ERROR: [Instagram] abc: login required, use --cookies")
    assert "isn't supported" in _friendly("ERROR: Unsupported URL: https://example.com")
