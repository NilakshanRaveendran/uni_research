from pathlib import Path

import pytest

from bilingual_voice.pipeline import LocalModels


def test_xtts_requires_explicit_license_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    models = LocalModels(model_root=tmp_path)

    with pytest.raises(RuntimeError, match="Coqui model license"):
        _ = models.tts
