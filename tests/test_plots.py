from pathlib import Path

from bilingual_voice.plots import create_plots


def test_plots_include_human_speaker_anchor_and_prosody(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        rows.append(
            {
                "direction": "en-es",
                "status": "ok",
                "speaker_similarity": str(0.65 + index * 0.05),
                "human_target_speaker_similarity": str(0.45 + index * 0.05),
                "f0_mean_target_reference": str(120 + index * 10),
                "f0_mean_generated": str(125 + index * 10),
            }
        )

    written = create_plots(rows, tmp_path)

    assert tmp_path / "speaker_similarity_by_direction.png" in written
    assert tmp_path / "prosody_f0_mean_generated_vs_reference.png" in written
    assert all(path.stat().st_size > 0 for path in written)
