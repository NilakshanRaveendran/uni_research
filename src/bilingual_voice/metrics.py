"""Per-sample and corpus-level evaluation utilities."""

from __future__ import annotations

import math
from collections import defaultdict


def _wer_transform():
    import jiwer

    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def normalized_wer(reference: str, hypothesis: str) -> float | None:
    if not reference.strip():
        return None
    import jiwer

    transform = _wer_transform()
    return float(
        jiwer.wer(
            reference,
            hypothesis,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
    )


def normalized_corpus_wer(references: list[str], hypotheses: list[str]) -> float | None:
    if not references:
        return None
    import jiwer

    transform = _wer_transform()
    return float(
        jiwer.wer(
            references,
            hypotheses,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
    )


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    import sacrebleu
    from scipy.stats import pearsonr

    result: dict[str, object] = {
        "attempted": len(rows),
        "succeeded": sum(row.get("status") == "ok" for row in rows),
        "failed": sum(row.get("status") != "ok" for row in rows),
    }
    result["failure_rate"] = result["failed"] / result["attempted"] if rows else 0.0

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("direction", "unknown")].append(row)

    directions: dict[str, object] = {}
    for direction, group in grouped.items():
        successful = [row for row in group if row.get("status") == "ok"]
        with_references = [row for row in successful if row.get("translation_reference")]
        refs = [row["translation_reference"] for row in with_references]
        hyps = [row["translated_text"] for row in with_references]
        scores: dict[str, object] = {"attempted": len(group), "succeeded": len(successful)}
        if refs:
            scores["corpus_bleu"] = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
            scores["corpus_chrf"] = float(sacrebleu.corpus_chrf(hyps, [refs]).score)
            scores["translation_reference_count"] = len(refs)
        else:
            scores["corpus_bleu"] = None
            scores["corpus_chrf"] = None
            scores["translation_reference_count"] = 0

        with_source_references = [row for row in successful if row.get("source_reference_text")]
        if with_source_references:
            scores["corpus_asr_wer"] = normalized_corpus_wer(
                [row["source_reference_text"] for row in with_source_references],
                [row["asr_text"] for row in with_source_references],
            )
        else:
            scores["corpus_asr_wer"] = None
        scores["asr_reference_count"] = len(with_source_references)

        with_tts_asr = [row for row in successful if row.get("tts_asr_text")]
        if with_tts_asr:
            scores["corpus_tts_intelligibility_wer"] = normalized_corpus_wer(
                [row["translated_text"] for row in with_tts_asr],
                [row["tts_asr_text"] for row in with_tts_asr],
            )
        else:
            scores["corpus_tts_intelligibility_wer"] = None
        scores["tts_asr_count"] = len(with_tts_asr)

        for field in (
            "asr_wer",
            "tts_intelligibility_wer",
            "speaker_similarity",
            "human_target_speaker_similarity",
        ):
            values = []
            for row in successful:
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            scores[f"mean_{field}"] = sum(values) / len(values) if values else None
            scores[f"{field}_count"] = len(values)

        same_speaker = [row for row in successful if row.get("same_speaker_pair") == "true"]
        scores["prosody_reference_same_speaker_count"] = len(same_speaker)
        for feature in ("f0_mean", "f0_std", "energy", "duration", "speaking_rate"):
            generated_values = []
            reference_values = []
            for row in same_speaker:
                try:
                    generated = float(row.get(f"{feature}_generated", ""))
                    reference = float(row.get(f"{feature}_target_reference", ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(generated) and math.isfinite(reference):
                    generated_values.append(generated)
                    reference_values.append(reference)
            correlation = None
            if (
                len(generated_values) >= 2
                and len(set(generated_values)) > 1
                and len(set(reference_values)) > 1
            ):
                correlation = float(pearsonr(generated_values, reference_values).statistic)
            scores[f"target_reference_{feature}_pearson_r"] = correlation
            scores[f"target_reference_{feature}_count"] = len(generated_values)
        directions[direction] = scores
    result["directions"] = directions
    return result
