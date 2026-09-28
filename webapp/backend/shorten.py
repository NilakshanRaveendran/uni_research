"""Shorter translations when a dubbed line will not fit (isochrony by n-best selection).

Professional dubbing shortens the words to fit the mouth rather than speeding the voice. MarianMT's
beam search already holds shorter paraphrases a little below its best one; this picks among them.

Measured on 209 unique sentences / 188 unique lines of the app's past jobs (both directions):
  * the app's beam-4 output is the top of the beam-8 n-best for 197/209 sentences;
  * delta 0.3 alone shortens 74% of sentences, mean -10.5% spoken characters over all sentences,
    and costs 4.9 back-translation chrF (corpus); judged on 40 changed sentences, 3 changed meaning;
  * with the round-trip check (a candidate's back-translation may score at most 15 chrF below the
    current one's) it shortens 65%, mean -7.8%, for -1.3 chrF, and none of those 3 get through;
  * on the 67 lines that needed WORLD it removes 10% of the characters (median WORLD factor 1.30 ->
    1.19; 16 need none). This is an estimate from each line's measured XTTS rate, not re-synthesised.
"""

from __future__ import annotations

import re

# Words whose loss flips or removes a negation. A shorter candidate must keep at least as many.
NEGATIONS = {
    "es": {"no", "ni", "nunca", "jamás", "jamas", "nadie", "nada", "ninguno", "ninguna", "ningún",
           "ningun", "tampoco", "sin"},
    "en": {"no", "not", "never", "nobody", "nothing", "none", "neither", "nor", "nowhere",
           "cannot", "without"},
}
_WORD = re.compile(r"[\wÀ-ÿ']+", re.UNICODE)
_DIALOGUE_DASH = re.compile(r"^\s*[-–—]\s*")  # OPUS subtitle training data: "- ¿De dónde eres?"

DEFAULT_DELTA = 0.3
MAX_ROUND_TRIP_CHRF_DROP = 15.0
# Spoken characters (letters and digits) XTTS says per second of speech at speed 1, after the
# silence is trimmed: median of 152 past takes (IQR 8.3-14.1), by OUTPUT language.
SPOKEN_CHARS_PER_S = {"es": 10.8, "en": 12.0}


def spoken_chars(text: str) -> int:
    """Letters and digits: what XTTS actually has to say (punctuation and spaces are not spoken)."""
    return sum(ch.isalnum() for ch in text)


def negations(text: str, language: str) -> int:
    words = [w.lower() for w in _WORD.findall(text)]
    count = sum(w in NEGATIONS.get(language, set()) for w in words)
    if language == "en":
        count += sum(w.endswith("n't") for w in words)
    return count


def keeps_meaning_markers(candidate: str, reference: str, language: str) -> bool:
    """A shorter line may not lose a question, a negation, a number or its sentence ending."""
    if not candidate.strip():
        return False
    if candidate.count("?") < reference.count("?"):
        return False
    if negations(candidate, language) < negations(reference, language):
        return False
    if set(re.findall(r"\d+", reference)) - set(re.findall(r"\d+", candidate)):
        return False
    # A hypothesis cut off mid-sentence: the reference ends a sentence, the candidate does not.
    cut_off = reference.rstrip()[-1:] in ".?!…" and candidate.rstrip()[-1:] not in ".?!…"
    return not cut_off


def nbest(model, tokenizer, sentences, beams: int = 8, max_new_tokens: int = 512, batch: int = 8):
    """Per sentence, the distinct beam-search candidates, best first, with their HF beam scores
    (sum of token log-probs / length, i.e. length_penalty 1: a length-normalised log-prob)."""
    import torch

    out = []
    for k in range(0, len(sentences), batch):
        chunk = list(sentences[k : k + batch])
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = {name: t.to(model.device) for name, t in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc, num_beams=beams, num_return_sequences=beams, max_new_tokens=max_new_tokens,
                length_penalty=1.0, output_scores=True, return_dict_in_generate=True,
            )
        texts = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        scores = gen.sequences_scores.tolist()
        for i in range(len(chunk)):
            seen, uniq = set(), []
            for text, score in zip(texts[i * beams : (i + 1) * beams], scores[i * beams : (i + 1) * beams]):
                text = _DIALOGUE_DASH.sub("", text).strip()
                if text and text not in seen:
                    seen.add(text)
                    uniq.append((text, float(score)))
            out.append(uniq)
    return out


def _chrf(hypothesis: str, reference: str) -> float:
    import sacrebleu

    return sacrebleu.sentence_chrf(hypothesis, [reference]).score


def eligible_candidates(candidates, current: str, language: str, delta: float = DEFAULT_DELTA):
    """Candidates within `delta` of the best score, shorter than `current`, keeping its markers."""
    if not candidates:
        return []
    top = max(score for _, score in candidates)
    base = spoken_chars(current)
    return [
        (text, score) for text, score in candidates
        if score >= top - delta and spoken_chars(text) < base
        and keeps_meaning_markers(text, current, language)
    ]


def pick_shorter(candidates, current: str, language: str, delta: float = DEFAULT_DELTA,
                 target_chars: int | None = None, source: str | None = None,
                 back_translations: dict | None = None,
                 max_chrf_drop: float = MAX_ROUND_TRIP_CHRF_DROP) -> str:
    """Choose a translation from an n-best list [(text, score), ...].

    Eligible: within `delta` of the best score, shorter (spoken characters) than `current`, keeping
    every question mark, negation, number and the sentence ending of `current`, and -- when
    `source` and `back_translations` ({text: back-translation}, covering `current`) are given --
    whose back-translation scores at most `max_chrf_drop` chrF below that of `current` against
    `source`. With `target_chars`, the best-scoring eligible candidate that reaches the target wins
    (shorten only as much as needed); otherwise, or when none reaches it, the shortest. Returns
    `current` when nothing is eligible.
    """
    eligible = eligible_candidates(candidates, current, language, delta)
    if eligible and source is not None and back_translations and current in back_translations:
        floor = _chrf(back_translations[current], source) - max_chrf_drop
        eligible = [(t, s) for t, s in eligible
                    if t in back_translations and _chrf(back_translations[t], source) >= floor]
    if not eligible:
        return current
    if target_chars is not None:
        reaching = [(t, s) for t, s in eligible if spoken_chars(t) <= target_chars]
        if reaching:
            return max(reaching, key=lambda row: row[1])[0]
    return min(eligible, key=lambda row: (spoken_chars(row[0]), -row[1]))[0]


def translate_fitted(model, tokenizer, sentences, currents, language: str, shorten,
                     target_chars=None, delta: float = DEFAULT_DELTA, back_translate=None,
                     beams: int = 8, max_new_tokens: int = 512):
    """Re-translate the sentences flagged in `shorten` from an n-best list; keep the rest as they are.

    `currents` are the app's normal (beam-4) translations of `sentences`; `target_chars` optionally
    gives a spoken-character budget per sentence. `back_translate(texts) -> texts` (the opposite-
    direction model) turns on the round-trip check; without it only the automatic markers guard.
    """
    result = list(currents)
    flagged = [i for i, flag in enumerate(shorten) if flag and sentences[i].strip() and currents[i]]
    if not flagged:
        return result
    lists = nbest(model, tokenizer, [sentences[i] for i in flagged], beams, max_new_tokens)
    backs: dict = {}
    if back_translate is not None:
        pool = sorted({currents[i] for i in flagged} | {
            t for i, cands in zip(flagged, lists)
            for t, _ in eligible_candidates(cands, currents[i], language, delta)})
        backs = dict(zip(pool, back_translate(pool)))
    for i, cands in zip(flagged, lists):
        target = None if target_chars is None else target_chars[i]
        result[i] = pick_shorter(cands, currents[i], language, delta, target,
                                 source=sentences[i] if backs else None, back_translations=backs)
    return result


def needs_shortening(translation: str, window: float, language: str, free_compression: float) -> bool:
    """Predicted not to fit: longer than XTTS can say in `window` with the compression that costs
    nothing audible (`free_compression`: XTTS speed times WSOLA's transparent range)."""
    rate = SPOKEN_CHARS_PER_S.get(language, 11.0)
    return spoken_chars(translation) > rate * max(window, 0.1) * free_compression


def budget_chars(window: float, language: str, free_compression: float) -> int:
    """Spoken characters that fit in `window` within the free compression."""
    return int(SPOKEN_CHARS_PER_S.get(language, 11.0) * max(window, 0.1) * free_compression)
