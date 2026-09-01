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

# Part 3 — The seven figures, and what each one proves

All generated from your real result files. Palette is Okabe-Ito, validated for colour-vision
deficiency (worst adjacent ΔE 11.4, above the 8 floor); every bar carries a direct value label so
the figures survive greyscale printing. No dual axes anywhere.

---

## G1 — `G1_prosody_transfer_gap.png` ★ **your single most important figure**

| Feature | Human EN vs human ES | Generated vs human target |
|---|---|---|
| F0 mean | **0.927** | **0.090** |
| F0 range | 0.307 | 0.079 |
| Duration | **0.873** | 0.491 |
| Speaking rate | 0.602 | 0.356 |

**What it proves.** When the *same human* performs content in both languages, their pitch level
carries across almost perfectly (r = 0.927) and their timing carries across strongly (r = 0.873).
When **your system** generates the target-language speech, the correlation with what the human
actually did collapses to **r = 0.090** for pitch.

**Say this:** "Prosody is a real, measurable, transferable property of the same speaker across
languages — and our pipeline does not transfer it. The generated pitch has essentially no
relationship to the human target."

**Why:** XTTS receives the reference audio for **timbre** only. It is never given the source's pitch
contour, so it invents its own intonation from the text.

**This is your contribution, not your failure.** It quantifies precisely the gap that Swiatkowski et
al. built an entire neural architecture (VIPT) to close — and they had to do it *without* parallel
expressive data, which is what DRAL gave you.

**Where it goes:** Results and Discussion, as the figure that motivates everything after it.

---

## G2 — `G2_pretrained_vs_finetuned.png`

BLEU and chrF, pretrained vs fine-tuned, both directions, n = 432.

**What it proves.** Fine-tuning improved reference-matching in both directions: +2.34 BLEU / +1.82
chrF for en-es, +3.23 / +1.97 for es-en.

**The essential caption:** the gain is **register adaptation, not translation adequacy** — DRAL's
Spanish is a re-enactment, so matching it means sounding more like DRAL. Pair this figure with the
`bicicleta → bici` examples from Part 2.

**Note that chrF moves less than BLEU** (+1.8 vs +2.3). chrF is character-based and more robust to
word-order and lexical substitution, which is consistent with the change being stylistic rather than
semantic. That is a sophisticated point to make if pressed.

---

## G3 — `G3_finetune_overfitting.png`

Train and dev loss per epoch, both directions, with the selected epoch marked.

**What it proves.** Textbook overfitting: train loss falls monotonically while dev loss bottoms at
epoch 2 and rises. **Both directions independently selected epoch 2 on dev.**

**Say this:** "We observed overfitting after two epochs on 1,998 training pairs, detected it on a
held-out development set, and selected the epoch-2 checkpoint. The test set was evaluated once."

This is your **methodology figure**. It demonstrates you understand train/dev/test discipline —
which is worth more to an examiner than the +2.34 BLEU itself.

---

## G4 — `G4_speaker_identity_vs_ceiling.png`

| Direction | Human ceiling | Generated | % of ceiling |
|---|---|---|---|
| en-es | 0.4452 | 0.4206 | **94.5%** |
| es-en | 0.4450 | 0.3832 | **86.1%** |

**What it proves.** Reported as a bare cosine, 0.42 sounds mediocre. Against the **human ceiling** —
the same speaker's real recording in the other language, which scores only 0.445 — it says the
system retains most of the identity that is *retainable* across a language switch.

**Why the ceiling exists at all:** because DRAL is same-speaker parallel. Most corpora cannot give
you this. It is a genuine methodological strength — emphasise it.

**The asymmetry is real.** The two ceilings are near-identical (0.4452 vs 0.4450), as they must be
since they compare the same two human recordings either way. So the 94.5% vs 86.1% gap is genuinely
in XTTS's output: **voice cloning works better into Spanish than into English.** Offer XTTS's
training-data balance as a *hypothesis*, not a claim — you have not tested it.

---

## G5 — `G5_duration_mismatch.png`

Histogram of generated ÷ human-target duration, n = 868, with the 1.0 line and the median marked.

**What it proves.** Raw XTTS output is systematically **1.52× too long** (generated 4.07 s vs human
2.67 s). Not random error — a consistent bias.

**Why it matters:** in a dubbed video this drifts out of sync within a few utterances. This figure is
the *problem statement* for your web app's retiming stage.

---

## G6 — `G6_prosody_model_vs_baselines.png`

Six panels: 2 directions × 3 targets, four systems each, with the speaker-clustered p-value.

**What it proves — read it carefully, because it is a mixed result:**

| Target | Direction | Ridge vs B2 | Speaker p | Significant? |
|---|---|---|---|---|
| F0 range | es-en | **−14.7%** | 0.017 | ✅ |
| F0 range | en-es | **−9.5%** | 0.042 | ✅ |
| Duration ratio | es-en | −5.3% | 0.035 | ✅ |
| Duration ratio | en-es | −0.1% | 0.194 | ❌ |
| F0 mean | en-es | −0.0% | 0.135 | ❌ |
| F0 mean | es-en | −0.0% | 0.670 | ❌ |

**The finding:** the predictable structure in cross-lingual prosody lives in **F0 variation**, not
in pitch level or (mostly) timing.

**The mechanism — quote this, it makes the result interpretable:** the learned weight on source F0
range is **−0.60** (en-es) and **−0.59** (es-en). Large and negative. The model learned that an
unusually wide source pitch range should be predicted **narrower** in the target — per-utterance
variation regresses toward the speaker's typical range rather than transferring one-to-one.

**For F0 mean, λ = 10⁶ was selected and all weights are ≈ 0 — the model *chose* to become the
baseline.** That is the regulariser working correctly, reporting honestly that no conditional signal
is available. It is evidence *against* overfitting, and a great answer to "did you overfit?"

**Two things that look like bugs and need one sentence each:**
- For log duration ratio, **B1 and B2 are mathematically identical** (copy-source is 0, so
  copy + offset = the mean). Hence p = 1.0000.
- **B0/B1/B2 MAE is direction-symmetric by construction** — swapping source and target mirrors the
  error distribution. Only the *ridge* rows genuinely differ between directions.

---

## G7 — `G7_cascade_error_propagation.png`

| Direction | MT on human transcript | MT inside the cascade | Cost |
|---|---|---|---|
| en-es | 23.98 | 22.14 | −1.84 BLEU (ASR WER 0.150) |
| es-en | 26.79 | 23.45 | −3.34 BLEU (ASR WER 0.221) |

**What it proves.** The classic weakness of cascaded systems, measured on your own data: ASR errors
propagate and cost translation quality downstream, and the cost is larger in the direction where ASR
is worse (Spanish source, WER 0.221).

**Why it is valuable:** it is a quantitative argument for end-to-end speech translation, made from
your own numbers rather than cited from a paper. Excellent Discussion and Future Work material.

---

## Suggested figure numbering for the report

| Report | File | Section |
|---|---|---|
| Figure 1 | *hand-drawn pipeline diagram* (you should draw this) | Methodology |
| Figure 2 | G5 duration mismatch | Results — synthesis |
| Figure 3 | G4 speaker identity vs ceiling | Results — synthesis |
| Figure 4 | **G1 prosody transfer gap** | Results — the headline |
| Figure 5 | G6 prosody model vs baselines | Results — model |
| Figure 6 | G3 fine-tuning curves | Results — fine-tuning |
| Figure 7 | G2 pretrained vs fine-tuned | Results — fine-tuning |
| Figure 8 | G7 cascade error propagation | Discussion |

If you must cut, cut G7 first (it is a bonus finding) and G2 second (its numbers are in a table
anyway). **Never cut G1, G3, or G4** — those are the headline, the methodology evidence, and the
identity result.
