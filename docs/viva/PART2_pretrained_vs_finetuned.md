# Part 2 — Pretrained vs fine-tuned: everything

This is the section a panel will push hardest on, so it is worth knowing cold.

---

## 1. What "pretrained" means

A pretrained model already had its weights set by someone else, on their data. You download it and
use it as-is. `Helsinki-NLP/opus-mt-en-es` was trained by the Helsinki-NLP group on the **OPUS**
corpus — millions of *written* sentence pairs from books, film subtitles, and EU documents.

Using it directly is called **zero-shot**: the model has never seen your data and you have not
adjusted it.

## 2. What "fine-tuning" means

You take those already-set weights as the *starting point* and continue training on your own data
with a small learning rate. The model adapts to your domain without forgetting what it learned.

You did **not** train from scratch. Training MarianMT from nothing needs millions of sentence pairs
and days of GPU time. You started from a finished model and adjusted it for ~12 minutes.

## 3. Why fine-tune at all — the three reasons

1. **Domain mismatch.** OPUS is written text. DRAL is spontaneous bilingual conversation with
   disfluencies (*"T- t- tell me"*), fillers (*"uh"*, *"eh"*), and regional Mexican Spanish. A model
   trained on EU documents is not matched to that.
2. **Your own objectives required it.** The progress report states: *"Pre-trained models like
   MarianMT will be fine-tuned for the domain-specific dialogue of films and conversational speech
   to improve contextual translation fidelity."*
3. **It is the only way to answer the question.** Whether adaptation helps cannot be asserted — it
   has to be measured against the pretrained baseline on the same test set.

## 4. The experimental setup

| | |
|---|---|
| Models | `Helsinki-NLP/opus-mt-en-es`, `opus-mt-es-en` |
| Training data | **1,998** DRAL train sentence pairs per direction |
| Dev (for selection) | **429** pairs |
| Test (evaluated once) | **432** pairs |
| Optimiser | AdamW, learning rate 2e-5 |
| Batch size | 8, gradient clipping 1.0 |
| Epochs run | 4 |
| **Epoch selected** | **2 — chosen on DEV, never on test** |
| Seed | 498 |
| Hardware | Apple M1 GPU (MPS) |
| Decoding (both systems) | beam = 4, max 128 tokens — **identical**, so the comparison is fair |

**Why 2e-5 and not larger:** fine-tuning needs a *small* learning rate. A large one would overwrite
the general translation knowledge — catastrophic forgetting. 2e-5 is roughly an order of magnitude
below typical from-scratch rates and is the standard range for adapting a pretrained transformer.

## 5. The results

| Direction | System | BLEU | chrF |
|---|---|---|---|
| en-es | pretrained | 23.98 | 49.87 |
| en-es | **fine-tuned** | **26.31** | **51.70** |
| en-es | *change* | *+2.34* | *+1.82* |
| es-en | pretrained | 26.79 | 50.30 |
| es-en | **fine-tuned** | **30.02** | **52.27** |
| es-en | *change* | *+3.23* | *+1.97* |

Both directions improved, on a held-out test set of unseen speakers, under identical decoding.

## 6. The overfitting evidence — know these numbers

| Epoch | en-es train | en-es dev | es-en train | es-en dev |
|---|---|---|---|---|
| 1 | 1.9222 | 1.7304 | 1.6138 | 1.3778 |
| **2** | 1.5694 | **1.7121** ← best | 1.3373 | **1.3610** ← best |
| 3 | 1.3676 | 1.7437 | 1.1651 | 1.3817 |
| 4 | 1.2207 | 1.7904 | 1.0322 | 1.4322 |

Read it across: **training loss falls monotonically** (1.92 → 1.22) while **dev loss bottoms at
epoch 2 and then rises** (1.712 → 1.790). The model has stopped learning the task and started
memorising the 1,998 training sentences.

**Both directions independently selected epoch 2.** That is the strongest thing you can say about
your methodology: the selection was made on a held-out development set, the test set was touched
once, and the same answer emerged twice.

## 7. The caveat you must volunteer before you are asked

**The +2.34 BLEU is register and style adaptation, not improved translation adequacy.**

DRAL's Spanish side is a **re-enactment**, not a literal translation. So matching the reference more
closely means *sounding more like DRAL*, not translating better. The evidence:

| Source | Pretrained | Fine-tuned |
|---|---|---|
| "in the bike or in the dam" | "en la **bicicleta** … **presa**" | "en la **bici** … **represa**" |
| "We help each other out." | "Nos ayudamos **mutuamente**." | "Nos ayudamos **el uno al otro**." |
| "tell me what day exactly" | "Dime **exactamente qué día**." | "Dime **qué día exactamente**." |

Colloquial contractions, regional lexis, and word order shifted toward the reference. **In several
cases the pretrained output is the better standalone translation.** The fine-tuned one is simply
closer to how these particular speakers talk.

Saying this yourself is far stronger than being caught by it. A student who identifies the
limitation of their own positive result is in a completely different position from one who defends
an overclaim.

## 8. Why only MarianMT — not Whisper or XTTS?

Expect this question. The answer:

- **Whisper** already achieves corpus WER 0.15 (English source) — there is little headroom, and
  fine-tuning ASR on 1,998 utterances of one corpus would likely *hurt* generalisation.
- **XTTS-v2** fine-tuning requires substantially more data and compute than a laptop provides, and
  the project's own guide explicitly places it behind the zero-shot baseline.
- **MarianMT is the cheapest meaningful experiment**: small enough to fine-tune in 12 minutes, with
  a clear domain mismatch to exploit and a well-understood metric to measure it.

That is a defensible engineering decision, not a shortcut — say it as such.

## 9. The number conflict you must reconcile

Your progress report quotes **BLEU ≈ 61**. This study measures the pretrained model at **23.98**.

Both are correct; they measure different things:
- **61** was **back-translation** BLEU — translate out and back, compare to the original. Much
  easier, because errors can cancel on the return trip.
- **23.98** is BLEU against an **independent human reference** (the DRAL Spanish re-enactment).

**Put one explicit sentence in the report distinguishing them**, or it reads as a regression you
introduced.

## 10. Cascade error propagation — a bonus finding

The fine-tuning comparison fed the models **human transcripts**. The full pipeline feeds them
**Whisper output**. Comparing:

| Direction | MT on human transcript | MT inside the cascade | Cost of ASR errors |
|---|---|---|---|
| en-es | 23.98 | 22.14 | **−1.84 BLEU** (ASR WER 0.150) |
| es-en | 26.79 | 23.45 | **−3.34 BLEU** (ASR WER 0.221) |

This quantifies the classic weakness of cascaded systems: **ASR mistakes propagate and cost
translation quality downstream**, and the cost is larger in the direction where ASR is worse. It is
a genuine argument for end-to-end approaches, and a good thing to raise in your discussion.
