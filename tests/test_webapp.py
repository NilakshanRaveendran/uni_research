from pathlib import Path

import pytest

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
    assert health["mt_mode"] == "pretrained en-es, fine-tuned es-en"


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


def test_music_heard_as_another_language_is_rechecked_not_rejected() -> None:
    from webapp.backend.dubbing import UnsupportedLanguageError, check_language

    # A music intro came back as Nynorsk at 76 %: undecided before separation...
    assert check_language("nn", 0.76, "en-es", final=False) is False
    # ...a real unsupported language is still refused at once, and so is the wrong direction.
    with pytest.raises(UnsupportedLanguageError):
        check_language("ta", 0.95, "en-es", final=False)
    with pytest.raises(UnsupportedLanguageError, match="Choose Spanish → English"):
        check_language("es", 0.60, "en-es", final=False)
    # On the separated vocals the verdict is final.
    with pytest.raises(UnsupportedLanguageError):
        check_language("nn", 0.76, "en-es")
    assert check_language("en", 0.99, "en-es") is True


def test_link_validation_and_friendly_errors() -> None:
    from webapp.backend.fetch import _friendly, valid_url

    assert valid_url("https://www.youtube.com/watch?v=abc")
    assert not valid_url("youtube.com/watch?v=abc")
    assert not valid_url("file:///etc/passwd")
    assert "needs a login" in _friendly("ERROR: [Instagram] abc: login required, use --cookies")
    assert "isn't supported" in _friendly("ERROR: Unsupported URL: https://example.com")


def _client():
    from fastapi.testclient import TestClient

    return TestClient(main.app, base_url="http://localhost")


def test_cross_site_writes_are_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main, "LICENSE_FILE", tmp_path / "accepted")
    client = _client()
    # a plain form POST from another site: no client header
    assert client.post("/api/license/accept").status_code == 403
    # the header, but sent from a foreign page
    foreign = {"x-bvt-client": "web", "origin": "https://evil.example"}
    assert client.post("/api/license/accept", headers=foreign).status_code == 403
    assert not (tmp_path / "accepted").exists()
    # the app's own page
    ok = {"x-bvt-client": "web", "origin": "http://localhost:3000"}
    assert client.post("/api/license/accept", headers=ok).status_code == 200
    assert (tmp_path / "accepted").exists()


def test_reads_need_no_header_but_foreign_hosts_are_refused() -> None:
    from fastapi.testclient import TestClient

    assert _client().get("/api/health").status_code == 200
    rebinding = TestClient(main.app, base_url="http://attacker.example")
    assert rebinding.get("/api/health").status_code == 400


def test_links_to_this_machine_or_the_local_network_are_rejected() -> None:
    from webapp.backend.fetch import valid_url

    for url in [
        "http://localhost:8000/api/health",
        "http://127.0.0.1/x",
        "http://[::1]/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.168.1.10/video.mp4",
        "http://10.0.0.5/v",
        "http://printer.local/v",
    ]:
        assert not valid_url(url), url
    assert valid_url("https://youtube.com/shorts/yTiooVPtA3s")
    assert valid_url("https://8.8.8.8/video.mp4")
