from bilingual_voice.splits import assert_speaker_disjoint, assign_speaker_splits


def test_speaker_split_is_deterministic_and_disjoint() -> None:
    rows = [
        {
            "pair_id": f"pair-{speaker}-{item}",
            "english_speaker_id": f"speaker-{speaker}",
            "spanish_speaker_id": f"speaker-{speaker}",
        }
        for speaker in range(10)
        for item in range(2)
    ]
    rows.append(
        {
            "pair_id": "cross-speaker-pair",
            "english_speaker_id": "speaker-0",
            "spanish_speaker_id": "speaker-1",
        }
    )
    first = assign_speaker_splits(rows, seed=42)
    second = assign_speaker_splits(rows, seed=42)
    assert [row["split"] for row in first] == [row["split"] for row in second]
    assert_speaker_disjoint(first)
    assert {row["split"] for row in first} == {"train", "dev", "test"}
    speaker_zero = next(row["split"] for row in first if row["pair_id"] == "pair-0-0")
    speaker_one = next(row["split"] for row in first if row["pair_id"] == "pair-1-0")
    assert speaker_zero == speaker_one
