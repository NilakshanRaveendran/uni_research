from bilingual_voice.metrics import summarize_rows


def test_summary_retains_failure_rate_and_missing_references() -> None:
    rows = [
        {
            "direction": "en-es",
            "status": "ok",
            "translation_reference": "",
            "translated_text": "hola",
            "speaker_similarity": "0.7",
        },
        {
            "direction": "en-es",
            "status": "failed",
            "translation_reference": "",
            "translated_text": "",
            "speaker_similarity": "",
        },
    ]
    summary = summarize_rows(rows)
    assert summary["attempted"] == 2
    assert summary["failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["directions"]["en-es"]["corpus_bleu"] is None
