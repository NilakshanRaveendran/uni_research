# ProsoMLP and ProsoCNN — prosody predictors for cross-lingual speech

Two small neural models that predict the prosody of a translated utterance from the
source utterance, trained from scratch on DRAL. They fill a gap in the cascaded
speech-to-speech pipeline (Whisper → MarianMT → XTTS), which preserves *speaker identity*
well but transfers almost no utterance-level *prosody*.

Released open source. Every number in this card is generated from the artefacts in this
repository by `python -m bilingual_voice.prosody_card build`, so it cannot drift from the
data.

## Architectures

Both models are written from scratch in PyTorch and trained from random initialisation.
**No pretrained weights are used in either model.**

### ProsoMLP — 6,284 parameters

A residual MLP over the source contour's DCT coefficients.
`Linear(16→64) → LayerNorm → GELU → Dropout(0.1) → Linear(64→64) → LayerNorm → GELU → Dropout(0.1) → Linear(64→12)`

- Trained on: `mlp-en-es, mlp-es-en` (5-seed ensemble, seeds [498, 499, 500, 501, 502])
- Optimiser: AdamW, lr 0.003, weight decay 0.01, batch 64, gradient clip 1.0
- Early stopping on dev loss, patience 40, max 400 epochs
- Loss: SmoothL1 on standardised residuals; contour coefficients weighted 0.3 relative to the three reported targets
- Device: CPU (at this size, MPS kernel-launch overhead exceeds the arithmetic)

### ProsoCNN — 4,796 parameters

A 1-D CNN reading the raw 64-point contour, bypassing the DCT compression.
`Conv1d(1→16,k=5) → GELU → Conv1d(16→16,k=5) → GELU → mean+max pool → concat 8 scalars → Linear(40→64) → GELU → Linear(64→12)`

- Trained on: `cnn-en-es, cnn-es-en` (5-seed ensemble, seeds [498, 499, 500, 501, 502])
- Optimiser: AdamW, lr 0.003, weight decay 0.01, batch 64, gradient clip 1.0
- Early stopping on dev loss, patience 40, max 400 epochs
- Loss: SmoothL1 on standardised residuals; contour coefficients weighted 0.3 relative to the three reported targets
- Device: CPU (at this size, MPS kernel-launch overhead exceeds the arithmetic)

### The output layer is initialised to zero — on purpose

Every target is parameterised as a **residual from baseline B2** (copy the source value,
plus one global offset fitted on train). With a zero-initialised head the prediction at
step 0 *is* B2, exactly. The model can only move away from the baseline if the dev loss
improves, so it is structurally impossible for it to start out worse than the published
baseline. A contract test asserts a fresh model outputs exactly zero.

## Training data

Speaker-component-disjoint 70/15/15 split, seed 498. Splits are formed over the union-find components of the speaker graph, so no speaker appears in more than one split.

- **en-es**: 1633 train / 322 dev / 348 test pairs; 14 test speakers
- **es-en**: 1633 train / 321 dev / 347 test pairs; 14 test speakers

Source: **DRAL** (Dialogs Re-enacted Across Languages), restricted to pairs where one
bilingual speaker performed both sides. DRAL's second-language side is a conversational
**re-enactment, not a literal translation** — so these models learn how a speaker's
prosody maps across their own two languages, which is exactly the quantity of interest,
but the pairs are not translation pairs and should not be described as such.

### Inputs (16), and the leak that was avoided

Eight DCT coefficients of the source contour, plus the source's pitch level, pitch span,
log duration, voiced ratio, word count and speaking rate, plus the **machine
translation's** word count and the log word ratio.

The target word count is taken from machine translation, **never from the human target
transcript**. The human target is the thing being predicted and its length is most of the
answer; a model given that number would post a duration score it could never reproduce in
deployment. MT output is available before synthesis, so using it is causally legitimate.

One residual mismatch is measured rather than assumed: MT is run over the gold source
transcript because train/dev have no ASR pass, while deployment translates Whisper output.
MT(gold) and MT(asr) word counts correlate at *r* = 0.611 (MAE 1.32 words), so the test
split additionally carries deployment-realistic ASR-derived features and both are reported
(`ProsoMLP` vs `ProsoMLP_deploy`).

## Results — test split

Mean absolute error against the human target. **B2** is the baseline the thesis reports;
**ridge_full** is the linear model given the *same 16 inputs* as ProsoMLP, so a neural win
is attributable to the architecture rather than to extra features.

### en-es

| target | B0 copy source | B2 copy plus offset | ridge t0 | ridge full | ProsoMLP | ProsoCNN |
|---|---|---|---|---|---|---|
| `f0_mean_st` | 1.3745 (+1.4%) | 1.3550 | 1.3548 (-0.0%) | 1.3548 (-0.0%) | 1.3472 (-0.6%) | 1.3031 (-3.8%) |
| `f0_std_st` | 0.9772 (-0.6%) | 0.9827 | 0.9317 (-5.2%) | 0.9333 (-5.0%) | 0.8778 (-10.7%) | 0.7864 (-20.0%) |
| `log_dur_ratio` | 0.1863 (+2.7%) | 0.1814 | 0.1814 (-0.0%) | 0.1812 (-0.1%) | 0.1754 (-3.4%) | 0.1758 (-3.1%) |

### es-en

| target | B0 copy source | B2 copy plus offset | ridge t0 | ridge full | ProsoMLP | ProsoCNN |
|---|---|---|---|---|---|---|
| `f0_mean_st` | 1.3497 (+1.4%) | 1.3305 | 1.3305 (-0.0%) | 1.3304 (-0.0%) | 1.3450 (+1.1%) | 1.2694 (-4.6%) |
| `f0_std_st` | 0.9630 (-0.5%) | 0.9682 | 0.8638 (-10.8%) | 0.9283 (-4.1%) | 0.8388 (-13.4%) | 0.8244 (-14.9%) |
| `log_dur_ratio` | 0.1865 (+2.7%) | 0.1815 | 0.1813 (-0.1%) | 0.1715 (-5.5%) | 0.1733 (-4.5%) | 0.1716 (-5.4%) |

### Statistical inference, and its limit

Errors are correlated within a speaker, and the test split holds only **14 speakers**. Two speaker-aware tests are reported: a Wilcoxon signed-rank test over
per-speaker mean absolute errors, and a bootstrap CI that resamples speakers rather
than recordings.

With 14 speakers the smallest attainable two-sided signed-rank *p* is **0.00012**. Holm correction is applied over a pre-specified family of 12
confirmatory comparisons (2 models x 3 targets x 2 directions); the correction over
all table rows is also reported as a conservative sensitivity check. **The binding
constraint on significance is the speaker count, not the effect size** — a property
of the corpus, not of the models.

## The evaluation ladder

Each arm imposes one system's predicted prosody on audio XTTS **already produced**, so the
words, the voice and the vocoder path are identical across arms and only the prosody
differs. No speech is re-synthesised.

| arm | what it is |
|---|---|
| `S0` | raw XTTS, untouched |
| `S0_identity` | WORLD analysed and resynthesised with no edit — the vocoder's own footprint |
| `S1_copy` | B2's prediction imposed. **This is the baseline the models must beat** |
| `S2_mlp` / `S2_cnn` | the from-scratch models' predictions imposed |
| `S3_dct` | the human target's own prosody, through the 8-coefficient DCT bottleneck |
| `S3_full` | the human target's own prosody at full contour resolution — the ceiling |
| `*_pitch` | as above but pitch ONLY, timing untouched — isolates the retiming's cost |

`S3 − S3_dct` is what the representation costs; `S3_dct − S2` is what the predictor costs;
`S2 − S1` is what the model buys.

### en-es

| arm | n | pitch MAE (st) | span MAE (st) | contour r | dur / target | ECAPA | WER |
|---|---|---|---|---|---|---|---|
| `S0` | 344 | 4.0461 | 3.2242 | 0.152 | 2.033 | 0.4398 | 0.0999 |
| `S0_identity` | 339 | 2.0641 | 1.2528 | 0.170 | 2.029 | 0.4118 | 0.1172 |
| `S1_copy` | 338 | 2.0172 | 1.0070 | 0.216 | 1.264 | 0.3592 | 0.1497 |
| `S2_mlp` | 339 | 1.8943 | 1.0022 | 0.253 | 1.266 | 0.3611 | 0.1436 |
| `S2_cnn` | 342 | 1.8728 | 0.8781 | 0.226 | 1.259 | 0.3587 | 0.1481 |
| `S2_cnn_pitch` | 345 | 1.9219 | 0.9734 | 0.212 | 2.028 | 0.3951 | 0.1247 |
| `S3_dct` | 337 | 1.7575 | 0.4921 | 0.576 | 1.261 | 0.3555 | 0.1370 |
| `S3_full` | 340 | 1.5509 | 0.5072 | 0.588 | 1.261 | 0.3599 | 0.1413 |
| `S3_full_pitch` | 345 | 1.5158 | 0.5310 | 0.588 | 2.029 | 0.3949 | 0.1284 |

### es-en

| arm | n | pitch MAE (st) | span MAE (st) | contour r | dur / target | ECAPA | WER |
|---|---|---|---|---|---|---|---|
| `S0` | 343 | 3.2697 | 1.7413 | 0.179 | 1.388 | 0.4032 | 0.1071 |
| `S0_identity` | 341 | 2.2666 | 1.1027 | 0.169 | 1.394 | 0.3769 | 0.1325 |
| `S1_copy` | 345 | 2.4178 | 0.9229 | 0.254 | 1.044 | 0.3347 | 0.1591 |
| `S2_mlp` | 343 | 1.9229 | 0.9306 | 0.310 | 1.043 | 0.3407 | 0.1806 |
| `S2_cnn` | 340 | 1.8501 | 0.8742 | 0.247 | 1.040 | 0.3378 | 0.1708 |
| `S2_cnn_pitch` | 343 | 1.8138 | 0.9583 | 0.250 | 1.393 | 0.3596 | 0.1378 |
| `S3_dct` | 337 | 1.7322 | 0.4191 | 0.620 | 1.029 | 0.3354 | 0.1656 |
| `S3_full` | 344 | 1.3816 | 0.5362 | 0.639 | 1.030 | 0.3358 | 0.1629 |
| `S3_full_pitch` | 345 | 1.4136 | 0.4837 | 0.653 | 1.392 | 0.3574 | 0.1457 |

## Limitations — read these before citing anything above

- **These models do not replace XTTS, Whisper or MarianMT.** They add a prosody-transfer component
  that the pretrained cascade does not have. Any claim that they beat a production TTS system would
  be false.
- **Not state of the art.** 6,284 and 4,796 parameters trained on 1,633 pairs per direction.
- **Injection costs speaker similarity and intelligibility.** ECAPA falls from 0.440 (raw XTTS) to
  ~0.359 and WER rises from 0.100 to ~0.148. Decomposed: the WORLD round-trip costs 0.028 ECAPA,
  the time-scaling costs a further 0.053, and the models' own pitch prediction costs +0.002 -- i.e.
  nothing. The oracle arm pays the same price, so the cost belongs to the injection method, not to
  prediction quality. Against the thesis's speaker-identity headline this is a regression from
  ~94.5% of the human ceiling to ~80.6%, and it must be reported as such.
- **Retiming is a net loss except for duration.** Pitch-only variants recover 0.02-0.04 ECAPA and
  most of the WER, and improve pitch accuracy in three of four cases. Retiming buys duration match
  and nothing else. Choose the configuration to match what the application needs.
- **Duration match is incomplete in en-es** (1.26x the human target) because XTTS overruns by 2.03x
  and the 0.5x time-scale clamp binds. es-en reaches 1.04x. Length-controlled MT is the real fix
  and is not attempted here.
- **Contour shape is only partly recovered.** ProsoMLP raises contour correlation above the
  copy-plus-offset baseline in both directions (0.343 vs 0.316 en-es; 0.360 vs 0.321 es-en) but
  ProsoCNN does not. The oracle reaches 0.588/0.639, so most of the available shape signal is still
  unreached. Cross-language contour correspondence in this corpus is only r = 0.24.
- **Duration is not a neural win.** A ridge given the same 16 inputs matches or beats both networks
  on es-en duration.
- **ProsoMLP is worse than baseline on es-en pitch level** (+1.09%). Reported, not hidden.
- **14 test speakers.** Effect sizes are large but no comparison survives Holm correction on the
  per-speaker signed-rank test. The speaker-clustered bootstrap CI excludes zero for pitch span in
  both directions for both models; that is the evidence the claims rest on.
- **No listening study.** Every number here is an objective measure. Whether any of it is audible
  to a listener has not been established.
- **The +/-3 semitone level cap binds on ~9% of utterances**, including for the oracle. The reported
  ceiling is therefore the ceiling *under a safety constraint* that protects speaker identity --
  which any deployed system would also need.
- **DRAL pairs are re-enactments, not translations**, so these models learn how one bilingual
  speaker's prosody maps across their own two languages. That is the quantity of interest, but the
  pairs are not translation pairs.
- **The audio-domain ladder was run twice.** The first run revealed that the injector was imposing
  contour shape without amplitude, discarding the models' best-supported prediction; that was fixed
  and the ladder re-run. The scalar results above are unaffected -- they score model predictions and
  the models were frozen before test was touched -- but the ladder is not a single-shot result.

## Reproducing

```bash
python -m bilingual_voice.contours    extract data/splits/dral_manifest_with_text.csv artifacts/contours --jobs 6
python -m bilingual_voice.mt_words    build   data/splits/dral_manifest_with_text.csv artifacts/mt_words.csv
python -m bilingual_voice.prosody_net prepare --direction en-es
python -m bilingual_voice.prosody_net overfit --arch mlp --direction en-es   # sanity gate
python -m bilingual_voice.prosody_net train   --arch mlp --direction en-es
python -m bilingual_voice.prosody_eval report --split dev
python -m bilingual_voice.prosody_ladder render  results/synthesis_test.csv --jobs 6
python -m bilingual_voice.prosody_ladder measure artifacts/ladder results/synthesis_test.csv
python -m bilingual_voice.prosody_ladder report  artifacts/ladder --synthesis-csv results/synthesis_test.csv
python -m bilingual_voice.prosody_figures all
```

Seeds are fixed (498-502). Training is CPU-only and takes seconds per model.
