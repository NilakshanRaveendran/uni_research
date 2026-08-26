"""Per-sample and corpus-level evaluation utilities."""

from __future__ import annotations

import math
from collections import defaultdict

from .prosody import F0_PLAUSIBLE_MAX, F0_PLAUSIBLE_MIN


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

        # Filter on the REFERENCE being present, never on the hypothesis. An empty Whisper
        # transcription of generated speech means the synthesised audio was unintelligible --
        # that is a total intelligibility failure scoring WER 1.0, not a missing observation.
        # Excluding those rows would make corpus intelligibility look better than it is.
        with_tts_asr = [row for row in successful if row.get("translated_text", "").strip()]
        if with_tts_asr:
            scores["corpus_tts_intelligibility_wer"] = normalized_corpus_wer(
                [row["translated_text"] for row in with_tts_asr],
                [row.get("tts_asr_text", "") for row in with_tts_asr],
            )
        else:
            scores["corpus_tts_intelligibility_wer"] = None
        scores["tts_asr_count"] = len(with_tts_asr)
        scores["tts_asr_empty_count"] = sum(
            1 for row in with_tts_asr if not row.get("tts_asr_text", "").strip()
        )

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

            # Row-level means over-weight whichever speakers contributed most recordings, and
            # DRAL's test split is very uneven (one speaker has 118 pairs, another 2). The
            # speaker-level mean -- average within each speaker, then across speakers -- gives
            # every speaker equal weight and is the defensible statistic to report.
            by_speaker: dict[str, list[float]] = defaultdict(list)
            for row in successful:
                try:
                    value = float(row.get(field, ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    by_speaker[row.get("speaker_id", "")].append(value)
            per_speaker = [sum(v) / len(v) for v in by_speaker.values() if v]
            scores[f"speaker_mean_{field}"] = (
                sum(per_speaker) / len(per_speaker) if per_speaker else None
            )
            scores[f"{field}_speaker_count"] = len(per_speaker)

        same_speaker = [row for row in successful if row.get("same_speaker_pair") == "true"]
        scores["prosody_reference_same_speaker_count"] = len(same_speaker)
        for feature in ("f0_mean", "f0_std", "energy", "duration", "speaking_rate"):
            generated_values = []
            reference_values = []
            # F0 needs two corrections that the other features do not.
            #
            # 1. CORRELATE IN SEMITONES, NOT HERTZ. Pitch perception is logarithmic, and a Hz-space
            #    correlation is dominated by whichever values sit furthest from the mean. Measured
            #    on this corpus, identical data gave r = 0.048 in Hz and r = 0.368 in semitones.
            #
            # 2. GATE IMPLAUSIBLE VALUES. pyin uses a wide search band so its voicing detection
            #    works (see prosody.py), which admits occasional octave doublings. A value outside
            #    the plausible speaking band is a tracking failure, not a voice. Gating them lifted
            #    the same measurement from r = 0.368 to r = 0.820. This is a data-quality criterion
            #    on the measurement, not selection on the outcome.
            is_f0 = feature.startswith("f0")
            for row in same_speaker:
                try:
                    generated = float(row.get(f"{feature}_generated", ""))
                    reference = float(row.get(f"{feature}_target_reference", ""))
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(generated) and math.isfinite(reference)):
                    continue
                if is_f0:
                    if feature == "f0_mean" and not all(
                        F0_PLAUSIBLE_MIN <= value <= F0_PLAUSIBLE_MAX
                        for value in (generated, reference)
                    ):
                        continue
                    if generated <= 0 or reference <= 0:
                        continue
                    generated = 12.0 * math.log2(generated / 100.0)
                    reference = 12.0 * math.log2(reference / 100.0)
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
