> **SUPERSEDED IN PART — read this first.**
>
> This document was written on 26 August 2026, before the F0 measurement was corrected
> (commits `312b42d` and `ea7542c`) and before the corpus study and the pipeline measurement were
> put on a single F0 search band. Every prosody-transfer number below that reads
> *r* = 0.09 or similar is a **metrics artefact**: F0 was correlated in hertz with no plausibility
> gate. Everything else — identity, BLEU/chrF, WER, fine-tuning, the dubbing bug — is unchanged.
>
> The corrected figures are in the dissertation (`thesis/main.pdf`, Chapter 4) and regenerate from
> `thesis/numbers.json`. In short: generated-versus-human F0 level is *r* = 0.698 / 0.754 pooled and
> **0.311 / 0.314 within-speaker**, against a human within-speaker ceiling of **0.849** — a real gap,
> but far smaller than this document says. Section 4.9 of the dissertation explains the correction.

---

# Final review — answers, the dubbing bug, and what to do next

---

# PART A — Your fourteen questions, answered simply

## 1. Where are all the models?

**Models you trained** (620 MB total, gitignored because too large for a repo):
```
artifacts/finetune/best-en-es/model.safetensors   310 MB
artifacts/finetune/best-es-en/model.safetensors   310 MB
```
Each folder also holds `config.json`, `vocab.json`, `source.spm`, `target.spm`,
`tokenizer_config.json`, `generation_config.json`. **All eight files are needed to reload it.**

**The ridge prosody model is not a file.** Its six coefficients live in
`artifacts/out_final/T5_coefficients.csv`. It refits in under a second.

**Models you downloaded** (4.6 GB, in `models/`):
| Folder | Contains |
|---|---|
| `models/cache` | Whisper small (461 MB) |
| `models/coqui` | XTTS-v2 (1.7 GB) |
| `models/huggingface` | MarianMT pretrained (2.4 GB) |
| `models/spkrec-ecapa-voxceleb` | ECAPA (cached elsewhere by speechbrain) |

## 2. The graphs, one by one

| # | Chart | In one sentence |
|---|---|---|
| **G1** | grouped bar | Humans transfer prosody across languages (*r*=0.93); your system does not (*r*=0.09). |
| **G2** | grouped bar | Fine-tuning raised BLEU +2.34 (en-es) and +3.23 (es-en). |
| **G3** | line | Train loss falls, dev loss rises after epoch 2 — overfitting, caught. |
| **G4** | grouped bar | Voice reaches 94.5% / 86.1% of what a human achieves across languages. |
| **G5** | histogram | Raw XTTS output is 1.52× too long — systematic, not random. |
| **G6** | 6-panel bar | The model beats the baseline only on pitch *range*, not level or timing. |
| **G7** | grouped bar | ASR errors cost 1.8–3.3 BLEU downstream — the cascade's weakness. |
| **G8** | **dot plot** | The 90% average hides a 69–128% per-speaker range. |
| **G9** | **contour** | Where human pairs actually sit — and it revealed two speaker clusters (male/female). |
| **G10** | **radar** | The human prosody "shape" is large; the system's collapses inward. |
| **G11** | **radial bar** | Scorecard: everything is 0.5–0.9 except prosody transfer at **0.10**. |
| **G12** | **line** | The 90% headline falls to 83% once you require recordings ≥3 s. |
| **G13** | **line (ECDF)** | The full distribution — the two curves nearly overlap, which means alone hide. |
| **G14** | **dot plot** | All six model fits vs three baselines, with significance marks. |
| **G15** | **dot plot** | *r*=0.927 is 71% between-speaker variance; within-speaker it is **0.787**. |

**The three that matter most: G1** (the headline), **G15** (the honesty correction), **G4** (identity).

## 3. How do the models work in this project?

Four models in a chain, each doing one job:

```
audio → [Whisper] → text → [MarianMT] → translated text → [XTTS-v2] → new audio
                                                              ↑
                                            source audio (for the voice)
                     [ECAPA] compares source voice vs generated voice → a score
```

- **Whisper** hears speech, writes words (plus timestamps in the web app).
- **MarianMT** translates the words.
- **XTTS-v2** reads the translation aloud, imitating the voice in the reference audio.
- **ECAPA** is measurement only — it never generates anything.
- **Ridge model** is analysis only — it predicts prosody numbers, it does not change audio.

## 4. Audio extraction and merging

**Extract** (`ffmpeg`):
```bash
ffmpeg -i video.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le source.wav
```
`-vn` drops video, `-ac 1` makes it mono, `-ar 16000` resamples to 16 kHz (what Whisper and ECAPA
expect), `pcm_s16le` is uncompressed WAV.

**Merge** (`ffmpeg`):
```bash
ffmpeg -i video.mp4 -i dubbed.wav -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest out.mp4
```
`-map 0:v:0` takes video from file 1, `-map 1:a:0` takes audio from file 2. **`-c:v copy` is the
important part**: the video is copied bit-for-bit, never re-encoded, so there is no quality loss and
it is fast. Only the audio is replaced.

## 5. How is the dubbed length the same as the original?

Two mechanisms:

1. **Per segment**, each synthesised clip is stretched or squeezed to exactly the duration of the
   segment it replaces, using WORLD analysis/resynthesis (`pyworld`). This changes duration without
   changing pitch — verified on a 150 Hz test signal where the pitch stayed 150.3 Hz across
   0.6×–2.0× scaling.
2. **Overall**, segments are placed on a silent timeline at their original start times, so the track
   is as long as the video. Confirmed: 65.28 s in → 65.28 s out.

**But this only works when the scale needed is within 0.5×–2.0×. That is the cause of your bug —
see Part B.**

## 6. Why the audio and video mismatch — DIAGNOSED

This is a real bug and your report is accurate. See **Part B** below for the full diagnosis and the
fix. Short version: **59% of segments in one of your jobs hit the 0.5× clamp**, so they stay too
long and bleed into the next segment, and where Whisper found no speech there are **silent gaps up
to 8.6 seconds**.

## 7. The dataset is ~25 GB but we used "limited" data — why?

**You did not subsample. You used every usable pair.** This is a misunderstanding worth clearing up:

| What the 25 GB contains | Used? |
|---|---|
| `DRAL-16kHz.tgz` archive (11 GB) | it *is* the extracted data, counted twice |
| `fragments-short` — 5,786 EN/ES clips | ✅ **all of it** |
| `fragments-long` — 4,496 clips | ❌ longer versions of the same content |
| `recordings` — 208 full conversations | ❌ unsegmented, no pair structure |
| Japanese files | ❌ out of scope |

**2,893 pairs is the complete set of EN/ES paired fragments in the release.** The quality gate then
removed 18.7% where F0 could not be measured reliably. So "more data" is not available inside DRAL —
it would require a different corpus.

## 8. Is this enough for submission?

**Technically, yes — you have more than most projects at this stage.** Working pipeline evaluated on
870 utterances, a fine-tuning experiment with proper checkpoint selection, a corpus study, a trained
model, a web application, 15 figures, provenance recorded.

**Two things are not finished:**
- **The report itself.** No amount of results substitutes for the written document.
- **Two claims need softening** (multiple comparisons, and the within/between decomposition). See
  Part C.

## 9. Keras or TensorFlow?

**Neither. This project uses PyTorch — version 2.13.0.**

Whisper, MarianMT (via Hugging Face `transformers`), XTTS-v2 (Coqui) and ECAPA (SpeechBrain) are all
PyTorch. The ridge model uses plain NumPy. **There is no TensorFlow and no Keras anywhere.**

If asked why: the entire modern speech ecosystem is PyTorch, and none of these four pretrained models
exists in a Keras form.

## 10. How the models work, where they save, what happens after training

**Before training:** MarianMT's ~74 million weights hold general English↔Spanish translation learned
from the OPUS corpus (books, subtitles, EU documents).

**During training:** for each batch of 8 sentence pairs — the model predicts the Spanish, the loss
measures how wrong it is, gradients flow backwards, AdamW nudges every weight slightly (learning rate
2e-5). Repeat 250 times per epoch. After each epoch, evaluate on the **dev** set.

**What we watched:** train loss kept falling (1.92 → 1.22) but dev loss started rising after epoch 2
(1.712 → 1.790). That is memorising, not learning. **So epoch 2 was kept.**

**After training:** `model.save_pretrained()` wrote the adjusted weights to
`artifacts/finetune/best-en-es/model.safetensors`. Nothing else changed on disk.

**How it is used afterwards:** loading that folder gives you a model that behaves like MarianMT but
prefers DRAL's conversational register. The web app loads it by default
(`BVT_MT_MODE=pretrained` switches back to the original for comparison).

## 11. How is this research, given it is mostly pretrained models?

The honest framing, and it *is* defensible:

**Integration is engineering. Measurement is research.** Your instrument is DRAL's same-speaker
parallel design, and it let you measure three things that could not be measured otherwise:

1. **A calibrated ceiling.** A cosine of 0.42 means nothing on its own. Against the *same person's*
   real recording in the other language (0.445), it becomes 94.5%. Almost no corpus permits this.
2. **The prosody gap, quantified.** Humans transfer pitch at *r*=0.787 within-speaker; the system
   achieves 0.090. That is the gap an entire Amazon architecture (VIPT) exists to close — and they
   had to work *without* parallel expressive data.
3. **A negative result with a mechanism.** Prosody prediction beats a constant offset only on pitch
   *range*, and the learned weight of −0.60 explains why: per-utterance variation regresses toward
   the speaker's typical range.

**The analogy to use:** a thermometer is not new physics, but measuring that a patient has a fever is
still a finding. You built a measurement apparatus and reported what it read — including the parts
that read badly.

**What it is not:** a new architecture, a state-of-the-art claim, or a listening study. Say that
plainly and the panel will respect the boundary.

## 12. Submit now, then perfect it in a month — is that sensible?

**Yes, and it is the right call.** Submit what is measured and honest. Then the month's work has a
clear target, because you now know exactly what is broken (Part B) and what is weak (Part C).

One condition: **the thesis must describe what exists now, not what you plan.** Put the improvements
in Future Work. A thesis that promises and a thesis that delivers are marked differently.

## 13 & 14. Where to improve, and which model to train next

See **Part C**. Short answer: **train a length-controlled translation model.** It is the same
12-minute fine-tune you already know how to run, it fixes the bug in Part B, and it is a genuine
research contribution.

---

# PART B — The dubbing bug, diagnosed from your own jobs

## What the data shows

From `webapp/jobs/3ae892f43977/result.json` — 39 segments, **23 clamped (59%)**, raw duration ratio
**2.17×**:

```
[ 21.76- 22.80] slot 1.04s  tts 6.68s  x0.50 -> 3.34s  CLAMPED  OVERFLOWS +2.30s
[ 56.24- 57.08] slot 0.84s  tts 7.61s  x0.50 -> 3.81s  CLAMPED  OVERFLOWS +2.97s
[ 60.72- 63.04] slot 2.32s  tts 13.42s x0.50 -> 6.71s  CLAMPED  OVERFLOWS +4.39s
[ 38.24- 40.48] ...                                    SILENT GAP 6.28s
[ 52.48- 54.96] ...                                    SILENT GAP 8.64s
```

## Cause 1 — clamping causes overlapping speech (your "mismatch")

A 0.84 s slot receives **7.61 s** of synthesised speech. To fit, it would need 0.11× scaling. The
clamp floor is **0.5×**, so it comes out at 3.81 s — **2.97 s too long**.

That overflow lands on top of the next segment. And because segments are *added* into the timeline
(`timeline[offset:end] += retimed`), overlapping audio **sums** — two voices speaking at once.
That is the garbled, out-of-sync audio you heard.

## Cause 2 — silent gaps (your "audio is muted")

Gaps of **6.28 s and 8.64 s** where Whisper detected no speech, so nothing was placed and the
timeline stays at zero. Either the original had speech Whisper missed, or it was music/noise which
the pipeline does not reproduce at all.

## Cause 3 — video is fine

Verified: 65.28 s in → 65.28 s out; 112.36 → 112.30. **The video is not being truncated.** What you
perceived as "the video is gone" is the silence plus desync above.

## The root cause is not the retiming — it is translation length

`raw_duration_ratio 2.17` means the translated speech is **more than twice** as long as the speech it
replaces. No retiming can absorb that without destroying the audio. The real problem is upstream:
**MarianMT produces a translation whose spoken length was never constrained.**

## Fixes, cheapest first

| Fix | Effort | Effect |
|---|---|---|
| **Fill silent gaps with the original audio** (quiet, ducked) instead of digital silence | ~20 lines | removes the dead stretches immediately |
| **Prevent overlap**: truncate a segment at the next segment's start instead of summing | ~5 lines | removes doubled voices |
| **Widen the clamp to 0.35×** and log it | 1 line | fewer overflows, some quality cost |
| **Merge adjacent short Whisper segments** before translating | ~25 lines | fewer absurd 0.84 s slots |
| **Length-controlled MT** — the real fix | one 12-min fine-tune | attacks the 2.17× at source |

---

# PART C — Final review: what to improve for the thesis and the journal

## Two claims you must soften before submitting

**1. Multiple comparisons.** You fitted six models and reported three p-values under 0.05. Under
Bonferroni (0.05/6 = 0.0083) **none survive**; the smallest is 0.0166. Holm and Benjamini-Hochberg
at 5% also reject everything. Write:

> No prosody result is family-wise significant. The F0-range results survive only under a 10% false
> discovery threshold. The claim rests on the sign and magnitude agreeing across both directions and
> on the coefficient mechanism, not on the p-values.

**2. The within/between decomposition.** Your *r* = 0.927 for F0 mean is **71% between-speaker
variance** — partly just "low-pitched people are low-pitched in both languages." Report both:

| Measure | Pooled | Within-speaker |
|---|---|---|
| F0 mean | 0.927 | **0.787** |
| Duration | 0.873 | **0.870** |
| F0 range | 0.307 | **0.264** |

Your duration finding survives intact, and the central contrast (0.787 vs 0.264) still holds.

## The one model to train next — length-controlled MT

**Why this and nothing else:**

- It **fixes the bug** in Part B at its source: the 2.17× length ratio and 59% clamp rate
- It is the **same 12-minute fine-tune** you already ran successfully
- It is a **real research contribution** — length-aware translation for dubbing is an active area
- It is **measurable**: BLEU, length-compliance rate, clamp rate, and final duration ratio

**How:** prepend a length-control token to each source sentence during fine-tuning — e.g. `<short>`,
`<normal>`, `<long>` based on the target/source character ratio in the training pair. At inference,
request `<short>` when the slot is tight. This is the standard isometric-MT recipe and needs no new
architecture.

**Expected result:** the clamp rate should fall well below 59%, and the duration ratio should
approach 1.0 *before* retiming — which means less DSP and better audio.

## Ranked improvement list for the final submission

| Priority | Work | Cost | Why |
|---|---|---|---|
| 1 | Soften the two claims above | 30 min | they are currently indefensible |
| 2 | Fix overlap + silent gaps in the app | 1–2 h | the bug you actually hear |
| 3 | Length-controlled MT fine-tune | half a day | fixes the cause, is a contribution |
| 4 | Merge short Whisper segments | 1 h | fewer impossible slots |
| 5 | Small listening study (5–10 listeners, 20 clips) | 1 day | the one required metric still missing |
| 6 | Frame-level F0 comparison (DTW-aligned) | 1 day | makes you comparable to VIPT's 0.30/0.40 |
| 7 | Re-run everything at the final commit | 1 h | provenance currently records an older commit |

## For the journal version specifically

The publishable core is **not** the pipeline. It is:

> *A same-speaker parallel corpus permits calibrated measurement of what cross-lingual speech
> synthesis preserves. We show speaker identity reaches 86–95% of a human cross-language ceiling
> while prosody correspondence collapses from r = 0.787 (human) to r = 0.090 (synthesised), and that
> the predictable component of cross-lingual prosody lies in pitch range rather than pitch level or
> timing.*

That is a focused, honest, defensible paper. Add the length-controlled MT experiment and you have a
second contribution: **a measured intervention on the failure you identified.**

What to leave out of the journal: the web app (mention as an artefact), the BLEU-61 reconciliation
(internal), and the fine-tuning register finding (a nice observation, not the headline).
