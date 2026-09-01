> # ⛔ SUPERSEDED — DO NOT STUDY FROM THIS FILE
>
> **Read this whole box before reading anything below it.**
>
> This pack was written on 26 August 2026, **before** the F0 measurement was corrected (commits
> `312b42d` and `ea7542c`). Every prosody-prediction answer below is built on the superseded
> analysis in `artifacts/out_final/`. The dissertation reports the corrected analysis in
> `artifacts/out_wideband/`. **If you memorise the answers below you will contradict your own
> submitted thesis in front of the panel.**
>
> | This file rehearses | The thesis says | Occurrences here |
> |---|---|---|
> | ridge beats B2 by **−9.54 %** / **−14.65 %** | **−5.14 %** / **−3.89 %** | 12 + 12 |
> | speaker *p* = **0.0419** / **0.0166** | ***p* = 0.0012 / 0.0001** | 10 |
> | learned weight **−0.6034** / **−0.5916** | **−0.1906** / **−0.1429** | 3 |
> | test split = **357** pairs | **348** (347 in es-en) | 14 |
> | human F0 level *r* = **0.927** | **0.9519** pooled, **0.8488** within-speaker | 18 |
> | F0 range transfers at *r* = **0.307** | **0.263** pooled, **0.214** within-speaker | — |
>
> **The most dangerous single line is §3.6 (line 314), and its conclusion is now inverted.**
> It argues that *"no prosody result is family-wise significant"* — reasoning from the old
> *p* = 0.0166 that Holm stops immediately. With the corrected *p*-values of **0.0012** and
> **0.0001**, the two pitch-range results **do** survive Bonferroni, Holm **and**
> Benjamini–Hochberg at 5 %. If an examiner asks *"does your pitch-range result survive
> multiple-comparison correction?"*, the answer is **yes, in both directions** — the opposite of
> what this file trained you to say.
>
> One caution in the other direction: a **third** result also clears the Bonferroni threshold
> (en-es F0 *level*, *p* = 0.0040) but its effect size is **−0.02 %**, a CI of about
> [−0.0003, −0.0001] semitones. That is significance on a zero-sized effect. Do not present it as a
> third win — say so yourself before the panel finds it.
>
> **What is still accurate here:** everything not about prosody prediction — speaker identity and
> the human ceiling, BLEU/chrF and the fine-tune, WER, the cascade-error argument, the dubbing
> defects, and the general examiner-handling strategy.
>
> Study instead from `docs/viva/ANSWER_CARD.md`, which is generated from `thesis/numbers.json`.

---

# Part 4 — Examiner question pack

Fifty questions from five independent examiner perspectives, plus an adversarial pass on where
the project is genuinely weak. Every model answer is grounded in your real measured numbers.

**How to use this:** read the answer, then close the document and say it out loud. If you cannot
say it without reading, you do not know it yet. Prioritise the **TRAP** and **HARD** questions,
and Part 5 below.

---


## A · Research methodology examiner

*Focus: Research methodology: question formulation, objectives, dataset justification, validity threats, contribution, reproducibility, ethics, and whether the conclusions are supported by the evidence*


### 1.1  [warm-up]

> **Before we go near any results, state your research question in one sentence. Then tell me what the contribution of this project actually is.**


The question is: in a cascaded speech-to-speech pipeline between English and Spanish, in both directions, how much of a speaker's identity and prosody actually survives, and can prosody be predicted from source-side features? The contribution is not a new model. We fine-tuned only MarianMT, about 74 million parameters, and fitted a ridge regression; Whisper small, XTTS-v2 and ECAPA were used exactly as downloaded. The contribution is measurement. Because DRAL is same-speaker parallel, we could compute a human ceiling for speaker similarity, 0.4452 for English to Spanish and 0.4450 for Spanish to English, and show generation reaches 94.5% and 86.1% of it. And we produced a clear negative result: prosody does not transfer, generated-to-human F0 mean correlates 0.048 and 0.133 where the same humans correlate 0.927. Plus a working end-to-end dubbing application.


*Follow-up to expect:* If the student answers "we built a system", the examiner will press: "Engineering is not a research contribution — what did you learn that nobody could have learned without this corpus?" The recovery is to name the two measurement claims: the human ceiling for speaker similarity, and the near-zero prosody correlation against the 0.927 human-human baseline.


### 1.2  [warm-up]

> **Tell me exactly what data this project used. All of it.**


One corpus only: DRAL, Dialogs Re-enacted Across Languages, the 16 kHz release, and only the English and Spanish audio. That is 2,893 paired fragments, 96 unique speakers, 5,786 wav files, of which 2,834 pairs are same-speaker and 59 cross-speaker. No third language, no Common Voice, no external audio for pitch or anything else. DRAL fragments are re-enactments: a bilingual speaker performs a conversational fragment in one language, then re-performs the same content in the other, so pairs are matched in content and speaker but are not literal translations. We built pairs from the official trans_id metadata rather than filename substitution, because some valid pairs carry different conversation IDs. Splits are speaker-disjoint by union-find over connected speaker IDs, seed 498, 70/15/15: raw 2,022, 436 and 435, and after the quality gate and the same-speaker restriction 1,668, 327 and 357.


*Follow-up to expect:* The examiner will ask what the quality gate removed and why that is not cherry-picking. Answer with the numbers: voiced-frame ratio at least 0.30, duration between 0.4 and 20 seconds, mean F0 within 60 to 400 Hz, dropping 541 of 2,893 or 18.7%, applied before any modelling and identically to both languages, so it is not target-dependent selection.


### 1.3  [standard]

> **There are larger and better-known speech translation resources — CVSS, Common Voice, Fisher. Justify choosing DRAL over them.**


The research question is speaker identity and prosody preservation, and that question is only answerable if the same human speaks both languages. Common Voice is monolingual read speech with no parallel pairs, so there is no target-side human reference to compare a generated utterance against. CVSS's target speech is synthetic — the target side is TTS — so asking how well we preserved a human speaker's voice is meaningless there. DRAL's 2,834 same-speaker pairs give the one thing neither offers: a human upper bound. The ECAPA cosine between the same speaker's real English and real Spanish recordings is 0.4452 and 0.4450, and only against that can I claim 94.5% and 86.1% of ceiling rather than reporting a bare cosine. DRAL is also conversational rather than read. The price is scale: 2,893 pairs, and only 14 speakers in the test split.


*Follow-up to expect:* The examiner will push on the cost side: "So you chose a corpus that makes your statistics weak." The student should concede that directly and argue the trade was deliberate — no amount of Common Voice data would let you compute a same-speaker human ceiling, and without a ceiling the speaker-similarity numbers have no interpretable scale.


### 1.4  [standard]

> **You stated three objectives at proposal stage. Go through them and tell me honestly which were met.**


Objective one, bidirectional voice-to-voice preserving voice and emotional tone, is partly met. The pipeline ran 868 of 870 rows, a 0.23% failure rate, and voice reached 94.5% and 86.1% of the human ceiling — but emotional tone was not preserved, with generated-to-human F0 mean correlations of 0.048 and 0.133. Objective two, temporal alignment through diarization and pause segmentation, was not implemented as written. What exists instead is Whisper segment timestamps plus WORLD time-scaling in the web app, which moved the duration ratio from 0.785 to 1.001 on a clean clip — timing alignment achieved by a different and narrower mechanism, with no diarization. Objective three is met: the Next.js and FastAPI application produces a remuxed dubbed video. Lip-sync was never in scope. So: one partly met, one substituted, one met.


*Follow-up to expect:* The examiner will ask why the objectives were not simply rewritten to match what was built. The honest answer is that objectives were fixed at proposal and we report against them as stated, flagging the substitution — retro-fitting objectives to results would make the whole evaluation unfalsifiable.


### 1.5  [standard]

> **Suppose I hand this project to another student next year and tell them to reproduce your numbers. What can they reproduce exactly, and what can they not?**


Reproducible exactly: the splits, because union-find over the speaker graph is deterministic and the seed is 498; the quality gate, because the thresholds are numeric — voiced ratio 0.30, duration 0.4 to 20 seconds, F0 within 60 to 400 Hz, dropping 541 of 2,893; the MarianMT run, with AdamW, learning rate 2e-5, batch 8, gradient clipping 1.0, four epochs, 1,998, 429 and 432 pairs per direction; and the ridge, which is closed-form numpy with a stated lambda grid from 1e-3 to 1e6. Decoding was held identical for pretrained and fine-tuned, beam 4 and 128 tokens, so the +2.34 and +3.23 BLEU are not decoding artefacts. Not reproducible bit-exactly: XTTS-v2 synthesis, because it is stochastic and we ran on Apple MPS, so the 1.522 duration ratio and cosine values will move slightly. And DRAL access requires agreeing to the maintainers' terms.


*Follow-up to expect:* The examiner may ask whether the 0.4206 and 0.3832 similarity figures would survive a rerun. The student should say the point estimates will shift slightly but the reported speaker-cluster bootstrap CIs are what carry the claim, and the ceiling comparison is direction-symmetric by construction, so the 94.5 versus 86.1 asymmetry is the robust part.


### 1.6  [standard]

> **This project clones real people's voices. Walk me through the ethical position.**


Three strands. Data: DRAL is a released research corpus collected with participant consent for research use, and we used it under those terms, reporting only anonymised speaker IDs, never identities. Cloning: XTTS-v2 reproduces a real person's timbre from a few seconds of reference audio, and in every one of the 870 evaluation rows the speaker was cloned into their own other language — the same person, never a third party. Misuse: the web application accepts arbitrary uploaded video, so it could clone someone who never consented; that risk is real, it sits with the uploader, and I would state it as a limitation rather than hide it. Human subjects: we ran no listening study, so no ethics approval was required, and that is also the honest reason there is no MOS — recruitment plus approval was not feasible in the time available.


*Follow-up to expect:* The examiner will ask what safeguard the deployed app should carry. A good answer names something concrete and modest — a consent affirmation at upload, and audible or metadata watermarking of synthesised output — while conceding neither is implemented in this version.


### 1.7  [**HARD**]

> **Your test set is 357 pairs. Your prosody statistics treat that as your sample size. Convince me that is a sound inference.**


It is not, and that is why we did not do it alone. Those 357 pairs come from only 14 speakers, and the distribution is severely uneven — one test speaker contributes 236 of the 870 synthesis rows, another contributes 4 — so a naive per-recording test is largely measuring one person repeatedly. Every prosody result is therefore reported twice: naive per-recording Wilcoxon, and per-speaker Wilcoxon with speaker-cluster bootstrap confidence intervals. Speaker similarity is reported speaker-weighted, not row-weighted, giving 0.4206 and 0.3832. Under clustering, only three results survive: F0 range at -9.54% with speaker p=0.0419 and -14.65% with p=0.0166, both CIs excluding zero, and Spanish-to-English duration at -5.27%, p=0.0353. With hindsight I would have capped rows per speaker to balance the split and treated the effective n as 14, which is honestly very low power.


*Follow-up to expect:* The examiner will ask why the split was not simply made larger or balanced. The recovery is that speaker-disjointness was the non-negotiable constraint given only 96 speakers, and balancing would have cost test size — but stating the effective n as 14 in the results section, not just in the limitations, is the fix.


### 1.8  [**HARD**]

> **You report that fine-tuning improved BLEU by 2.34 and 3.23 points. Tell me precisely what that demonstrates.**


Not improved translation quality. DRAL Spanish is a re-enactment, not a translation of the English, so the reference is a paraphrase of the same content produced independently by the same speaker. Fine-tuning moved the model toward DRAL's spontaneous conversational register, so going from BLEU 23.98 to 26.31 and 26.79 to 30.02 is register and style adaptation, not adequacy. The qualitative evidence says so plainly: "bicicleta" becomes "bici", "mutuamente" becomes "el uno al otro", "exactamente qué día" reorders to "qué día exactamente", and "presa" becomes "represa". In several cases the pretrained output is the better standalone translation and scores lower purely because it does not match the re-enactor's wording. So the defensible claim is that fine-tuning aligns output style with real bilingual conversational speech; claiming better translation would need a genuine translation reference we do not have.


*Follow-up to expect:* The examiner may then produce the earlier progress report quoting BLEU 61 and ask which number is a lie. Neither: 61 was back-translation BLEU, a far easier task where the model is scored against text it started from, while 23.98 is against an independent human reference — both correct, different metrics, and the student should say so without hedging.


### 1.9  [**HARD**]

> **Your title claims the system preserves speaker identity. On the evidence in front of you, is that conclusion supported?**


Only in the timbre dimension, and the title over-claims. Timbre is supported: speaker-weighted ECAPA cosine 0.4206 into Spanish and 0.3832 into English against human ceilings of 0.4452 and 0.4450, so 94.5% and 86.1%. Because those two ceilings are essentially identical, as they must be, the asymmetry is a real finding — cloning works better into Spanish than into English. Everything else about identity is not preserved. Generated-versus-human F0 mean correlates 0.048 and 0.133 where the same humans correlate 0.927; F0 range 0.017 and 0.141; and generated audio averages 4.07 seconds against a 2.67-second human target, 1.522 times too long. The mechanism explains it: XTTS conditions on the reference for timbre only and is never given the source pitch contour. The honest title would say speaker timbre, not speaker identity.


*Follow-up to expect:* If the student points to the web app's retiming as a fix, the examiner should press hard. The correct concession: the WORLD retiming exists only in the application, the 870-row evaluation used raw XTTS with no correction, so 1.522 is the research measurement and the 0.785 to 1.001 improvement is a separate demonstration on one clean clip, not an evaluation result.


### 1.10  [**TRAP**]

> **The strongest p-value anywhere in your prosody tables is 0.0014, for F0 mean prediction in the English-to-Spanish direction. Take me through that result.**


That is the one number in the table I would refuse to call a finding. It is the naive per-recording Wilcoxon, which assumes 356 independent items when there are 14 speakers; the speaker-level test on the identical data gives p=0.135, so it does not survive clustering. The effect size makes it moot anyway — the ridge is 0.02% better than baseline B2, which is nothing. And the selected regularisation was lambda equal to 1e6 in both directions. Because the ridge fits the residual from copy-source, lambda tending to infinity reproduces B2 exactly, so the model nests the baseline and chose to collapse into it. Read properly, that row is a clean null: the six source features carry no conditional information about target F0 level beyond one global offset. The genuine results are F0 range, -9.54% and -14.65%, significant at speaker level.


*Follow-up to expect:* The examiner will ask why an uninformative p-value was reported at all. The answer is that reporting both tests side by side is precisely what exposes the inflation from speaker clustering — dropping the naive figure would conceal the methodological problem rather than solve it, and the same logic explains reporting p=1.0000 where B1 and B2 are mathematically identical for log duration ratio.


---


## B · Machine-learning examiner

*Focus: Training, fine-tuning and regularisation: what was actually learned, with what optimiser, and how it was kept honest*


### 2.1  [warm-up]

> **You use four neural models in this system. Which of them did you actually train, and which did you simply download and run?**


Only two things were fitted in this project. The first is MarianMT, which was genuinely fine-tuned: Helsinki-NLP opus-mt-en-es and opus-mt-es-en, about 74 million parameters each, continued training on 1,998 DRAL sentence pairs per direction with AdamW, learning rate 2e-5, batch size 8, four epochs, seed 498, on Apple MPS. That took 12.4 minutes for en-es and 10.8 for es-en. The second is the ridge prosody model, which is a closed-form least-squares fit in numpy, six source features, three targets, six fits per direction, under a second to run. Everything else was used exactly as downloaded with no weight updates at all: Whisper small for ASR, XTTS-v2 for synthesis and voice cloning, and ECAPA speechbrain for speaker embeddings.


*Follow-up to expect:* If the answer is vague, the examiner will ask what the contribution is if three of four models are off-the-shelf. The student should point to the fine-tuning result, the corpus study on 2,352 pairs, and the speaker-clustered evaluation design as the contributions, not the model weights.


### 2.2  [warm-up]

> **You describe the ridge regression as a model you trained. Is that the same kind of training as the MarianMT fine-tuning?**


No, and I would not want to blur them. MarianMT fine-tuning is iterative gradient training: roughly 250 AdamW steps per epoch over 1,998 pairs, four epochs, a learning rate of 2e-5, gradient clipping at 1.0, and a loss curve that can and does overfit. The ridge model has no epochs, no learning rate, no batches and no stochasticity; it is a closed-form normal-equation solve on standardised features with an unpenalised intercept, and it finishes in under a second. Its only hyperparameter is the ridge penalty lambda, selected on the dev split over a grid from 1e-3 to 1e6 using the one-standard-error rule. So it is a statistical fit, not deep learning, and I only claim deep-learning training for MarianMT.


*Follow-up to expect:* The examiner may ask whether the report is careless in calling both "models". The student should concede the wording could be tightened and offer the phrasing "a fine-tuned neural translator and a regularised linear prosody predictor".


### 2.3  [standard]

> **Why did you fine-tune only the translation model? Why not fine-tune Whisper or XTTS on DRAL as well?**


Mainly because of what the data can support. On the text side I have 1,998 clean sentence pairs per direction, which is enough to shift a 74-million-parameter translator's register. On the audio side, after the quality gate and the same-speaker restriction I have 1,668 training pairs from a speaker-disjoint split of 96 speakers in total. Fine-tuning XTTS-v2 or Whisper, both trained on thousands of hours, on that amount of re-enacted conversational speech would very likely degrade them rather than help. They also already perform: TTS intelligibility WER is 0.0808 and 0.0866, speaker similarity reaches 94.5 and 86.1 per cent of the human ceiling, and Whisper's WER is 0.1502 and 0.2212. The compute limit, one M1 laptop, is the honest secondary reason.


*Follow-up to expect:* Expect the examiner to press on the poor prosody transfer, F0 mean r=0.048 and 0.141 for range. The student must say that is an architectural limit, XTTS is never given the source pitch contour, so no amount of fine-tuning on 1,668 pairs would fix it.


### 2.4  [standard]

> **Justify your training hyperparameters. Why AdamW, why a learning rate of 2e-5, why batch size 8, and why four epochs?**


AdamW is the standard optimiser for transformer fine-tuning; its decoupled weight decay means the penalty is not rescaled by the adaptive step size, which matters when you take only a few hundred updates from pretrained weights. 2e-5 is the conventional fine-tuning rate, one to two orders of magnitude below pretraining rates, small enough to adapt the model without destroying what it already knows. Batch 8 was set by what fits comfortably on Apple MPS, and with 1,998 pairs it gives about 250 optimiser steps per epoch, which is enough movement to matter. Gradient clipping at 1.0 protects against a spike on a long sentence. Four epochs was chosen simply to be long enough to see the dev curve turn, and it turned at epoch 2. I should be honest: I ran no hyperparameter search, these are standard defaults, and the only value I genuinely selected was the epoch.


*Follow-up to expect:* The examiner may ask what would change with a proper search. The student should name learning rate and epoch count as the two worth searching, and note that with a dev set of only 429 pairs a wide search risks overfitting the dev split itself.


### 2.5  [standard]

> **Take me through your fine-tuning loss curves and tell me exactly which checkpoint you shipped, and how you chose it.**


For en-es, training loss falls monotonically, 1.9222, 1.5694, 1.3676, 1.2207, while dev loss goes 1.7304, 1.7121, 1.7437, 1.7904, so the dev minimum is epoch 2. For es-en it is the same shape: train 1.6138 down to 1.0322, dev 1.3778, 1.3610, 1.3817, 1.4322, again a minimum at epoch 2. That is textbook overfitting: after epoch 2 the model is fitting the 1,998 training sentences rather than the task. I shipped the epoch-2 checkpoint in both directions, selected purely on dev. The test split of 432 pairs was touched once, with identical decoding for both systems, beam 4 and a 128-token limit, giving 26.31 against 23.98 BLEU for en-es and 30.02 against 26.79 for es-en.


*Follow-up to expect:* The examiner may ask why the two directions agree on epoch 2 and whether that is suspicious. The student should say it is expected, since both directions use the same corpus size, learning rate and batch size, so the point at which capacity starts memorising is similar.


### 2.6  [standard]

> **Your prosody predictor is ridge regression. Why not a small neural network? Surely that would model the relationship better.**


Two reasons. First, the data: 1,668 training pairs, six features, three targets, and a test split of 357 pairs from only 14 speakers, one of whom contributes 236 of the 356 evaluated rows. In that regime extra capacity buys variance, and any gain a network reported would be impossible to audit against speaker confounding. Second, and more important, ridge is fitted on the residual from copy-source, so it exactly nests baseline B2: as lambda tends to infinity the six weights vanish and you recover copy-source-plus-one-global-offset. That means the lambda chosen on dev is itself a test of whether conditional signal exists. For F0 mean it came out at 1e6, the top of the grid, so the model told me there is none. A neural network cannot make that statement cleanly.


*Follow-up to expect:* If the answer stops at "not enough data", the examiner will ask what would make a network justified. The student should say more speakers rather than more rows, since the binding constraint is 14 test speakers, not 1,668 training pairs.


### 2.7  [**HARD**]

> **You report that lambda was selected as 1e6 for F0 mean in both directions. What does that number actually mean, and what does it tell you about your model?**


The grid ran from 1e-3 to 1e6 and the one-standard-error rule chose the ceiling of the grid in both directions. Because the fit is on the residual from copy-source, that limit shrinks all six weights to zero and the model collapses onto baseline B2, copy the source value plus one global offset. So the ridge chose to become the baseline. That is exactly why the reported change is minus 0.02 per cent for en-es and minus 0.01 per cent for es-en, and why the naive per-recording p of 0.0014 collapses to a per-speaker p of 0.135. I report it as a genuine negative result: given those six source features there is no per-utterance information about target F0 mean beyond a global offset. It is consistent with the corpus finding that pitch level transfers at r=0.927, so copying is already near-optimal.


*Follow-up to expect:* The examiner may ask why the naive test gave p=0.0014 at all. The student should explain that per-recording tests treat 356 correlated rows from 14 speakers as independent, so one dominant speaker's small systematic residual manufactures significance.


### 2.8  [**HARD**]

> **You continued training a general-purpose translator on 1,998 sentences of conversational re-enactment. How do you know you did not cause catastrophic forgetting?**


Strictly, I do not know, because I never evaluated the fine-tuned models on an out-of-domain general test set such as Flores or a WMT set. That is a real limitation and it is the experiment I would add first. What I can say is that the update was deliberately small: four epochs at 2e-5, with the epoch-2 checkpoint, roughly five hundred optimiser steps, and the en-es dev loss only moved from 1.7304 to 1.7121. The qualitative changes also look like register drift rather than collapse: bicicleta becomes bici, mutuamente becomes el uno al otro, presa becomes represa. In several cases the pretrained output is the better standalone translation, which is precisely what mild forgetting of the formal register looks like, not a broken model.


*Follow-up to expect:* Expect the examiner to ask how to protect against forgetting if training longer. The student should name a held-out general benchmark as a stopping criterion, plus lower learning rates, fewer updated layers, or mixing in general-domain data.


### 2.9  [**HARD**]

> **You have about 74 million parameters and 1,998 training sentences. That is tens of thousands of parameters per sentence. Why is that not hopeless?**


Because I am not estimating 74 million parameters from 1,998 sentences. The opus-mt models arrive already carrying the translation function, learned from tens of millions of sentence pairs, so fine-tuning takes about 250 small AdamW steps per epoch at 2e-5 from an already good point in weight space. The effective degrees of freedom actually being fitted are far smaller than the parameter count, and early stopping is itself a regulariser: the dev minimum at epoch 2 marks where capacity starts going into memorising the training set instead of adapting. The evidence that it generalised is that the gains held on an unseen, speaker-disjoint test split of 432 pairs, plus 2.34 BLEU and 1.82 chrF for en-es and plus 3.23 and 1.97 for es-en. Epochs 3 and 4 show what happens when capacity does overwhelm the data.


*Follow-up to expect:* The examiner may ask what the capacity is being spent on if not translation. The student should say register and lexical-style adaptation, and volunteer that DRAL Spanish is a re-enactment, so the BLEU gain is style match rather than improved adequacy.


### 2.10  [**TRAP**]

> **Your F0-range fits beat baseline B2 by 9.54 and 14.65 per cent, with a large negative weight of about minus 0.60 on source F0 range. It is clearly learning something real, so why not lower lambda, add features, or use a higher-capacity model and get more out of it?**


Because those two numbers survive precisely because of the regularisation discipline. Lambda was chosen on dev with the one-standard-error rule, which deliberately takes the most penalised model within one standard error of the dev optimum, and the result then holds under per-speaker Wilcoxon tests, p=0.0419 for en-es and 0.0166 for es-en, with speaker-cluster bootstrap intervals excluding zero. With 14 test speakers and one contributing 236 of 356 rows, a lower lambda or a bigger model would raise the naive per-recording figures while failing the speaker test, which is exactly the F0-mean pattern: naive p=0.0014, speaker p=0.135. The achievable ceiling is also low by nature, since human English and Spanish F0 range correlate at only r=0.307, and the minus 0.60 weight is regression toward the speaker's typical range, not a rich function waiting to be learned.


*Follow-up to expect:* If the student takes the bait and says more capacity would help, the examiner will ask them to predict what happens to the per-speaker p-value. The recovery is to state that dev and naive metrics would improve while the speaker-clustered test would lose significance, which is the definition of a speaker-confounded gain.


---


## C · Statistics examiner

*Focus: Statistical inference and validity: speaker clustering, baseline choice, effect size versus significance, selection effects, multiplicity, and replicability*


### 3.1  [warm-up]

> **Your prosody test set has 357 pairs. How many independent observations do you actually have, and why does the distinction matter?**


The 357 pairs come from only 14 speakers, so I have something much closer to 14 independent units than 357. Two recordings from the same speaker share the same vocal tract, the same habitual F0 level and the same speaking style, so their errors are correlated. A naive per-recording test treats all 357 as independent, so it divides by a standard error that is far too small and reports significance that is not there. The split is also very uneven: one test speaker contributes 236 synthesis rows and another only 4, so a per-row test is largely a statement about one person. That is exactly why the primary inference in the report is a per-speaker Wilcoxon on n equals 14, with speaker-cluster bootstrap confidence intervals, and the naive p-values are shown only for contrast.


*Follow-up to expect:* If the student only says "the data are correlated", the examiner will ask for the number: what is the design effect, or how much did a p-value actually move? Point at F0 mean en-es, where naive p equals 0.0014 became speaker p equals 0.135 — a conclusion reversal, not a rounding change.


### 3.2  [warm-up]

> **You define three baselines, B0, B1 and B2. Which one is the comparator that your ridge model has to beat, and why is beating the other one not interesting?**


B2 is the comparator. B1 predicts the training mean and ignores the input entirely, so it is trivially weak: a speaker's English and Spanish F0 mean correlate at r equals 0.927 in the corpus study, so simply copying the source value already beats the mean by a wide margin. B0 is that copy, and B2 is copy plus one global fitted offset per direction, which absorbs the systematic cross-language shift of plus 0.249 semitones. B2 matters for a structural reason as well: the ridge is fitted on the residual from copy-source, so it nests B2 exactly — as lambda tends to infinity the ridge reproduces B2. Any genuine claim of conditional, per-utterance signal therefore has to show up as an improvement over B2, and I only report the comparison against B2 as the headline.


*Follow-up to expect:* The examiner will press on log duration ratio, where B1 versus B2 gives p equals 1.0000. Say plainly that this is an algebraic identity — the copy value is zero, so copy plus offset is the mean — so the p-value is vacuous and should be read as "the same estimator twice", not as evidence of equivalence.


### 3.3  [standard]

> **For F0 mean in the English-to-Spanish direction you report a naive p of 0.0014 and a per-speaker p of 0.135. Reconcile those, and tell me what you conclude.**


They are not in conflict; they are answering different questions. The naive Wilcoxon over 356 correlated rows has a very small effective standard error, so a completely consistent but microscopic advantage gets ranked as significant. The effect size is minus 0.02 per cent MAE reduction against B2 — two hundredths of one per cent. Once I aggregate to the 14 speakers, p is 0.135 and the cluster bootstrap interval covers zero. The decisive corroboration is not statistical at all: the one-standard-error rule selected lambda equals 1e6 in both directions, which means the ridge shrank itself back to B2. The model itself found no conditional signal in the six source features for F0 mean. So I report it as a null result, and I treat the naive p as an artefact of the wrong unit of analysis.


*Follow-up to expect:* The examiner will ask why the report shows the naive p-value at all if it is misleading. Answer that it is shown deliberately, as the demonstration of how much clustering matters, and because a reader comparing with papers that do naive per-utterance testing needs to see both numbers.


### 3.4  [standard]

> **Describe exactly what your speaker-cluster bootstrap resamples, and what a row-level bootstrap would have got wrong.**


It resamples the 14 test speakers with replacement, not the 357 rows. When a speaker is drawn, all of that speaker's rows come with them — 236 for one speaker, 4 for another — the paired difference in mean absolute error between the ridge and B2 is recomputed inside each resample, and the interval is the 2.5 and 97.5 percentiles over resamples. That keeps the within-speaker correlation and the unevenness inside the resampling distribution instead of assuming them away. A row-level bootstrap would break speakers apart, treat 357 correlated rows as 357 draws and reproduce the same over-confidence as the naive test. Two results survive this: F0 range en-es at minus 9.54 per cent with speaker p equals 0.0419, and es-en at minus 14.65 per cent with p equals 0.0166, both with intervals excluding zero.


*Follow-up to expect:* Expect a challenge on coverage: fourteen clusters is few, so the bootstrap distribution is coarse and the intervals are only approximately valid. Concede it — say the cluster bootstrap is the honest choice rather than a precise one, and that the mechanism and the sign agreement across directions carry more weight than the interval endpoints.


### 3.5  [standard]

> **You have five statistically framed prosody comparisons. Which of them do you actually believe are findings, and on what grounds?**


One clear finding, one modest one, three nulls. F0 mean is minus 0.02 and minus 0.01 per cent, practically zero whatever the p-value says, and the ridge chose lambda equals 1e6 in both directions, so I call it nothing. English-to-Spanish duration at minus 0.07 per cent is also nothing. The finding is F0 range: minus 9.54 per cent en-es and minus 14.65 per cent es-en, with a mechanism I can name — the learned weight on the source F0 standard deviation is minus 0.6034 and minus 0.5916, a large negative, meaning an unusually wide source range is predicted narrower in the target. The corpus study explains why: F0 range transfers at only r equals 0.307 across languages against 0.927 for level, so copying is the wrong rule for range. Spanish-to-English duration at minus 5.27 per cent is real but modest.


*Follow-up to expect:* The examiner will ask you to separate effect size from significance in one sentence. Say that a 0.02 per cent improvement with a p of 0.0014 is a sample-size artefact, while a 9.54 to 14.65 per cent error reduction with a named coefficient and an independent corpus-level reason is an effect that happens also to be significant.


### 3.6  [standard]

> **You fitted six ridge models — three targets in two directions — and you report three p-values below 0.05: 0.0419, 0.0353 and 0.0166. What happens to those when I correct for multiple comparisons?**


Most of them do not survive, and I should say so. With a family of six, Bonferroni needs p below 0.0083, and none of my three reach that. Holm is the same verdict: the smallest, 0.0166, times six is 0.0996, so the procedure stops immediately. Benjamini-Hochberg at a false discovery rate of 5 per cent also rejects nothing, because 0.0419 would need to be below 0.025 and 0.0166 below 0.0083. At a 10 per cent false discovery rate all three pass, since 0.0419 is below 0.05 at rank three. So the correct claim is that no prosody result is family-wise significant, and the F0 range results are only survivors under a tolerant FDR threshold. What I actually rely on is the sign and magnitude agreeing across both directions plus the coefficient mechanism, not the p-values.


*Follow-up to expect:* The examiner may ask whether the two F0 range tests are independent evidence or two tickets in the same lottery. Answer that they are neither: they are the same mechanism tested in two directions on the same 14 speakers, so they are correlated tests, which is why replication of the sign matters more than multiplying the p-values.


### 3.7  [**HARD**]

> **Your quality gate discarded 541 of 2,893 pairs, 18.7 per cent. Convince me that dropping nearly a fifth of the data has not biased your results in your own favour.**


The overlapping reasons are voiced ratio 529, nan F0 280, implausible F0 280 and duration 97. Every one of those is a condition where the F0 measurement itself is undefined or outside the 60 to 400 hertz search range, so those rows would contribute measurement error rather than prosody. Crucially the gate is defined on the input recordings and applied before the split and identically to B0, B1, B2 and the ridge, so it cannot tilt the ridge-versus-B2 comparison, which is a paired difference on the same surviving rows. The same logic is why I kept the 8 pairs pinned at the F0 search bound rather than removing them: dropping rows because their error is large would be target-dependent selection, and that would be real bias. What the gate does cost me is external validity — my claims cover voiced fragments of 0.4 to 20 seconds only, not backchannels.


*Follow-up to expect:* Expect: "Show me the sensitivity analysis." Concede honestly that the evaluation was not re-run with a relaxed gate, name that as the missing check, and note the supporting evidence that the excluded material is genuinely degenerate — the only two pipeline failures out of 870 rows were the sub-0.33-second backchannels "Mm-hmm." and "Yeah."


### 3.8  [**HARD**]

> **You report mean absolute error. Prosody errors are heavy-tailed and one test speaker supplies 236 of your rows. Why should I trust a mean of a heavy-tailed quantity, and what would a median tell me instead?**


You are right that the mean is the vulnerable statistic. F0 range has a mean of 2.251 semitones with a standard deviation of 1.295, so the distribution is skewed and a handful of very wide-range utterances can move a mean absolute error, and with 236 rows from one speaker against 4 from another, a per-row mean is close to a per-speaker mean of the wrong speaker. Two things protect the inference. The test statistic is the Wilcoxon signed rank, which uses only ranks and so is insensitive to how extreme the tail is, and it is computed on the 14 speaker-level differences, so every speaker gets weight one. What I did not do is report median absolute error or a trimmed mean beside the MAE, and that is a fair gap. I would expect the minus 9.54 and minus 14.65 per cent to shrink but keep their sign.


*Follow-up to expect:* The examiner will ask why you expect only shrinkage rather than reversal. Argue from the coefficient: a weight of minus 0.6 on source F0 standard deviation shifts every wide-range utterance, so the gain is systematic shrinkage across the distribution, not a few outliers being rescued.


### 3.9  [**HARD**]

> **Suppose next year another group ran your exact pipeline on a fresh set of DRAL speakers. Which of your numbers would you bet on reproducing, and which would you expect to vanish?**


I would bet on the direction of the F0 range result and not on its size. The reasons are that both directions agree in sign, minus 9.54 and minus 14.65 per cent, that the learned weights are almost identical at minus 0.6034 and minus 0.5916, and that the mechanism is visible in the corpus independently of the test set: F0 range transfers at r equals 0.307 while level transfers at 0.927, so shrinking an extreme source range is the right rule for anyone's speakers. I would not defend the exact p-values of 0.0419 and 0.0166 from 14 clusters, and I would call the Spanish-to-English duration gain of minus 5.27 per cent provisional. The F0 mean nulls would reproduce, because the ridge selected lambda equals 1e6 and became B2 by itself. Nothing here transfers to real dubbing, where source and target speakers differ.


*Follow-up to expect:* Expect: "So what would make your claim replicable?" Say the honest answer is more speakers rather than more utterances — the binding constraint is 14 clusters, not 357 rows — plus a pre-registered single target and a cross-speaker test set.


### 3.10  [**TRAP**]

> **Your corpus study finds English and Spanish F0 mean correlating at r equals 0.927 across 2,352 pairs. That is a strong cross-linguistic result about pitch transferring between languages. Defend it.**


I would not defend that reading, because the correlation is largely guaranteed by the design. Every one of those 2,352 pairs is the same bilingual speaker re-enacting the fragment in both languages, so the variance driving r is between-speaker variance — the standard deviation is 6.663 semitones in English and 6.638 in Spanish, mostly anatomy and sex differences — while the actual language effect is a shift of plus 0.249 semitones with an interval of 0.146 to 0.351. So r equals 0.927 mostly says a person's voice is their voice. The informative number is the contrast with F0 range, where r is only 0.307 and the difference of minus 0.025 is not significant: even within one speaker, per-utterance pitch variation does not carry over. That contrast is also why copy-source, and hence B2, is such a hard baseline to beat for level.


*Follow-up to expect:* The examiner may push further: does the same by-construction problem inflate the ECAPA speaker similarity numbers? Answer yes in the same sense — the human ceilings of 0.4452 and 0.4450 exist only because DRAL is same-speaker parallel — but note that the ceiling is what makes the 94.5 versus 86.1 per cent asymmetry interpretable rather than inflated.


---


## D · Phonetics / speech examiner

*Focus: Prosody and acoustic measurement: what prosody is, how F0, range, duration and rate were extracted and operationalised, why energy was excluded, what ECAPA cosine and the human cross-language ceiling actually mean, and why timbre transferred while prosody did not.*


### 4.1  [warm-up]

> **Let's start simply. When you say your system preserves prosody, what do you actually mean by prosody, and which parts of it did you measure?**


Prosody is the layer of speech above the individual phoneme: the melody, the rhythm and the loudness pattern. I measured three utterance-level dimensions, plus one derived rate. Pitch level as mean F0 in semitones, pitch variation as the standard deviation of the semitone contour, and duration in seconds, with speaking rate in words per second on top. On the 2,352 same-speaker DRAL pairs, English averages 8.874 semitones against Spanish 9.123, F0 range 2.251 against 2.227 semitones, duration 2.740 against 2.856 seconds, and rate 3.407 against 3.042 words per second. Loudness I left out deliberately, and I can explain why. So to be precise: I measured utterance-level prosodic statistics, not the full contour, and that scope limit matters for what I can claim.


*Follow-up to expect:* Expect: "So you did not measure intonation, only its variance?" Concede it directly — a single standard deviation says how much the pitch moved, never where or in what shape, so utterance-level statistics are the coarse end of prosody modelling.


### 4.2  [warm-up]

> **All your pitch numbers are in semitones rather than hertz. Why?**


Because hertz is linear but pitch perception is roughly logarithmic. A semitone is twelve times the log base two of the frequency over a fixed reference, so equal semitone steps are equal perceived intervals: 100 to 200 hertz and 200 to 400 hertz are both one octave, twelve semitones, even though one is a 100 hertz jump and the other 200. My search range of 60 to 400 hertz is nearly three octaves, about 33 semitones, and it contains both low male and high female voices. In hertz, a 20 hertz standard deviation means something completely different for a 90 hertz speaker and a 220 hertz speaker; in semitones the same 2.251 figure means the same perceptual wobble for both. It also makes the corpus delta interpretable: plus 0.249 semitones is Spanish sitting about one and a half percent higher in hertz, which is tiny.


*Follow-up to expect:* Expect: "What reference did you use, and does it matter?" Say the reference only shifts every value by a constant, so it cancels in deltas and correlations, but it must be identical across both languages or the comparison is meaningless.


### 4.3  [standard]

> **You define F0 range as the standard deviation of the semitone contour. Justify that choice, and tell me what it throws away.**


I wanted one number per utterance that survives noisy pitch tracking. A true min-to-max range is decided by two frames, so a single halving or doubling error hands you a twelve-semitone range from nothing. A standard deviation over all voiced frames averages that error down, and because it is in semitones it is comparable across speakers with different base pitch. The values are modest and stable: English 2.251 plus or minus 1.295 semitones, Spanish 2.227, with the language difference minus 0.025 and a confidence interval of minus 0.086 to 0.037, so not significant. What it throws away is everything about shape and position — a rising boundary tone and a mid-utterance emphasis of the same size are identical to my measure. Frame-level F0 correlation, as Swiatkowski and colleagues report, is a genuinely different and finer measurement.


*Follow-up to expect:* Expect: "If the language difference in F0 range is not significant, what is your ridge model learning?" Answer that the means match but transfer is weak, r equals 0.307, so the variance is there to model, and the learned weight of about minus 0.60 on source range is regression toward the speaker's typical range, not a language offset.


### 4.4  [standard]

> **Talk me through how you extracted F0, and why you restricted the search to 60 to 400 hertz.**


I used pyin, the probabilistic YIN tracker, which gives a per-frame F0 and a voicing decision. The 60 to 400 hertz window is a deliberate compromise: wide enough to cover the low male and high female speakers in 96 DRAL speakers, but narrow enough that a candidate and its octave cannot both sit inside the range for most voices, which is what suppresses halving and doubling errors. Had I used something like 40 to 800, one octave-jumped frame would inflate the standard deviation badly. I then gated on the result: mean F0 inside 60 to 400, voiced ratio at least 0.30, duration 0.4 to 20 seconds. That removed 541 of 2,893 pairs, 18.7 percent, with overlapping reasons — voiced ratio 529, NaN F0 280, implausible F0 280, duration 97. Eight pairs sit at the search bound and I kept them, because dropping rows by their error size would be target-dependent selection.


*Follow-up to expect:* Expect: "Those eight bound-pinned pairs are probably wrong values — why keep them?" Hold the line: removing them because their error looks large would filter the test set using the quantity being predicted, which biases the result in your favour; leaving them in is the conservative choice.


### 4.5  [standard]

> **What exactly is the voiced-frame ratio, and why did you gate on it at 0.30?**


It is the fraction of analysis frames that pyin declares voiced — roughly, how much of the fragment is actually periodic speech rather than silence, breath, laughter or noise. It matters because every pitch statistic I report is computed only over voiced frames, so if a fragment has ten voiced frames the mean and especially the standard deviation are essentially noise. At 0.30 it was the single largest gate: 529 of the 541 dropped pairs failed it, often the same rows that also gave NaN F0 for exactly this reason. I also feed it into the ridge model as s_voiced_ratio, as a proxy for how densely packed with speech a fragment is. The same issue bit synthesis at the other end: XTTS refused two reference clips under 0.33 seconds, the backchannels "Mm-hmm" and "Yeah", which is 2 of 870 rows, 0.23 percent.


*Follow-up to expect:* Expect: "Why 0.30 and not 0.5?" Be honest that it was a judgement call balancing statistic stability against corpus loss, not a tuned parameter, and that at 0.30 you already lose 18.7 percent of pairs — a stricter gate would shrink an already small 14-speaker test set.


### 4.6  [standard]

> **You model pitch and duration but not loudness. The textbooks list intensity as one of the three pillars of prosody. Why did you drop it?**


Because in this corpus absolute energy is a recording measurement, not a prosodic one. I measured a 29 decibel spread in RMS level within a single conversation, so a change of that size can be entirely microphone distance, gain or channel, and any model I fitted would mostly be predicting the recording setup. Predicting an absolute RMS target would also be useless downstream, since the output level of XTTS is set by the synthesiser and the final mix, not by my prediction. So I dropped it, and I would rather say that plainly than report a target I cannot defend. The honest framing is that this is a scope limitation, not a claim that intensity does not matter: the correct version is a channel-normalised or within-utterance relative dynamic, and that requires level normalisation I did not do.


*Follow-up to expect:* Expect: "Could you not just normalise per file and use that?" Concede yes in principle, and say the reason you did not is that per-file normalisation destroys exactly the between-utterance loudness variation you would be trying to model, so it needs per-conversation calibration you had no metadata for.


### 4.7  [**HARD**]

> **Your generated audio averages 4.07 seconds against a human target of 2.67. Where does that factor of one and a half come from, and what did you do about it?**


There are two separate effects and it would be wrong to conflate them. The linguistic one is small: across 2,352 human pairs the Spanish-over-English duration ratio is 1.068 on average, median 1.047, so about five percent. The 1.522 factor in the pipeline is a synthesis artefact — XTTS-v2 is given text and a reference voice with no duration constraint at all, so it simply speaks slowly, and machine-translation output length adds to it. In the research evaluation I applied no correction, so 1.522 is an honest uncorrected baseline. In the web app I fix it in the timing dimension using WORLD, pyworld, time-scaling each segment into its original Whisper slot: the raw ratio of 0.785 on a clean clip became 1.001. On a synthetic 150 hertz signal the durations scaled from 0.6 to 2.0 times while measured F0 stayed 150.3 hertz, confirming duration changes only.


*Follow-up to expect:* Expect: "Your clean clip was 0.785, shorter than the slot, but your corpus figure is 1.522 — which is it?" Explain that they are different measurements: 1.522 is the mean over 868 fragment-level rows with no correction, while 0.785 is one segment-level clip in a segmented pipeline, and per-segment scaling is clamped to 0.5 to 2.0 with clamps counted.


### 4.8  [**HARD**]

> **You clone the voice successfully, yet your generated F0 correlates with the human target at r equals 0.048. How can identity transfer and prosody fail at the same time?**


Because they are different quantities, and XTTS only receives one of them. The reference audio conditions the model on speaker timbre — the spectral signature of the voice — and it is never given the source pitch contour, so pitch is regenerated from text. That is exactly what the numbers say: generated-versus-human F0 mean correlates 0.048 for English to Spanish and 0.133 the other way, F0 range 0.017 and 0.141, duration 0.364 and 0.618, whereas the two human recordings of the same content by the same speaker correlate 0.927, 0.307 and 0.873. Meanwhile ECAPA similarity is 0.4206 against a human ceiling of 0.4452, 94.5 percent. So this is timbre transfer, not prosody transfer. And I must own the architectural gap: my ridge model predicts target prosody but was never wired into the synthesiser, so its predictions were never used.


*Follow-up to expect:* Expect: "Then what was the prosody model for?" Say it is a measurement study establishing what is predictable — F0 range genuinely improves on the copy-plus-offset baseline by 9.54 and 14.65 percent — and that the obvious next step is injecting those predictions as F0 scaling through WORLD, which you did not implement.


### 4.9  [**HARD**]

> **You found Spanish longer and slower and you explain it with syllable-timing versus stress-timing. Does your data actually support that explanation?**


It is consistent with it, but I would not claim I demonstrated it. What I measured is that Spanish fragments are 0.116 seconds longer, with a confidence interval of 0.087 to 0.145, a ratio of 1.068 mean and 1.047 median, and slower at 3.042 against 3.407 words per second, a difference of minus 0.365. The standard account fits: English is stress-timed with heavy vowel reduction in unstressed syllables, Spanish is syllable-timed with full vowels, so Spanish spends more syllables and more time on the same content. But my rate is words per second, not syllables per second, so I cannot separate more syllables per word from slower articulation — the syllable-timing story is an interpretation of a word-level measurement. It is also telling that rate correlates only 0.602 across the pair while duration correlates 0.873, so rate is the less speaker-stable property.


*Follow-up to expect:* Expect: "How would you test it properly?" Say you would recount rate in syllables per second, or phones per second from a forced aligner, and check whether the Spanish disadvantage disappears — if it does, it is syllable count; if it survives, it is articulation speed.


### 4.10  [**TRAP**]

> **A cosine similarity of 0.42. That is less than half. By any normal reading your voice cloning failed. Defend it.**


The absolute number is not interpretable, and that is the point. ECAPA cosine is not calibrated so that one means the same voice; the meaningful question is what the same human achieves when they speak the other language. Because DRAL is same-speaker parallel, I could measure that ceiling directly: 0.4452 for English to Spanish and 0.4450 for Spanish to English, between one bilingual speaker's own real recordings. My generated audio scores 0.4206 and 0.3832, so 94.5 and 86.1 percent of ceiling. The two ceilings agreeing to three decimals is the sanity check that they are the same quantity, which makes the eight-point asymmetry real: cloning works better into Spanish. Two caveats — it is speaker-weighted over only 14 test speakers, one with 236 rows and one with 4, and ECAPA was trained for speaker identity, so it largely measures timbre, not prosody.


*Follow-up to expect:* Expect: "So your metric cannot tell a good clone from a bad one in absolute terms?" Agree, and say that is precisely why you report a ratio to a measured human ceiling rather than a raw cosine, and why a listening study would be the proper complement — which you could not run for ethics and recruitment reasons.


---


## E · Systems / software examiner

*Focus: Software engineering and systems: cascade architecture versus end-to-end, error propagation, the web application, latency and deployability, reproducibility and testing*


### 5.1  [warm-up]

> **Before we get into the results, walk me through your system as an architecture. What are the stages, and which of them did you actually train?**


It is a four-stage cascade. ffmpeg extracts the audio, Whisper small transcribes it with segment timestamps, MarianMT translates each segment, XTTS-v2 re-synthesises it cloning the speaker from the reference audio, and in the web app pyworld's WORLD vocoder time-scales each segment back into its original slot before ffmpeg remuxes with the video stream copied. Only two things were trained. MarianMT was fine-tuned, about 74 million parameters, four epochs with epoch two selected on dev, 12.4 minutes for English-to-Spanish on Apple MPS. And a ridge regression prosody model, which is a closed-form numpy fit taking under a second, so a statistical fit rather than deep learning. Whisper, XTTS-v2 and the ECAPA speaker embedder are used exactly as downloaded, and ECAPA is only an evaluation metric, not part of the pipeline.


*Follow-up to expect:* If ECAPA is not in the pipeline, what actually conditions XTTS on the speaker's voice? The student must say the reference wav conditions timbre only, and that the source pitch contour is never given to XTTS, which is why generated-to-human F0 correlation is only 0.048.


### 5.2  [warm-up]

> **Why FastAPI on the backend and Next.js on the frontend? Justify the stack rather than describing it.**


The hard constraint is that Whisper, MarianMT, XTTS-v2 and pyworld are all Python, and XTTS holds large weights and takes seconds per segment, so the models have to be loaded once into a long-lived Python process rather than reloaded per request. FastAPI gives us that, plus typed request and response models and async endpoints, so an upload can be accepted while the work continues. Next.js with TypeScript is the client because the user-facing part is really upload, progress and playback, and typed contracts catch mismatches at compile time. Honestly, though, the framework is not the interesting decision. At roughly 13x realtime, where a 15 second clip takes about 3 minutes, the design consequence is that this must be a job with progress reporting rather than a synchronous HTTP call, and that is what shaped the split.


*Follow-up to expect:* Could you not have done this in Flask or Streamlit? Concede that a demo would work in either; the argument is typed contracts and async handling of a three-minute request, not raw performance.


### 5.3  [standard]

> **Your app asks Whisper for segment timestamps. Why do you need them? What breaks if you simply translate and synthesise the whole transcript in one go?**


Because dubbing is an alignment problem, not only a translation problem. Each Whisper segment carries a start and end time, and that is the slot the new audio has to fit. Without it you have one long translated utterance with nothing anchoring it to the picture, and the drift is large. In the research run raw XTTS output averaged 4.07 seconds against a 2.67 second source, 1.522 times too long, so a single block would run roughly half the clip length past the end. Some expansion is inherent even with perfect translation: in DRAL the Spanish side is on average 6.8% longer than the English, ratio mean 1.068. Per-segment slots let us correct locally, and on a clean test clip the raw duration ratio of 0.785 became 1.001 after retiming.


*Follow-up to expect:* What if Whisper's boundaries are wrong? The student should concede we measured only WER, 0.1502 and 0.2212, never boundary accuracy, and that Objective 2's diarisation and pause segmentation, which was meant to address exactly this, was not implemented.


### 5.4  [standard]

> **Quantify error propagation for me. What does the cascade cost you, stage by stage?**


We measured it directly rather than asserting it. MarianMT on human reference transcripts scores BLEU 23.98 English-to-Spanish and 26.79 Spanish-to-English, but the same model inside the cascade, translating Whisper output, scores 22.14 and 23.45. So ASR error costs about 1.8 BLEU in one direction and 3.3 in the other. The asymmetry tracks the recogniser: Whisper's corpus WER is 0.1502 on English source and 0.2212 on Spanish, so the noisier Spanish recognition does more downstream damage. TTS then adds its own layer, with measured intelligibility WER of 0.0808 and 0.0866, and prosody is lost outright at that stage, F0 mean correlation with the human target being 0.048 and 0.133 against a human-to-human ceiling of 0.927. That is the honest cost of a cascade: each stage is individually respectable and the composition is not.


*Follow-up to expect:* Which stage would you fix first, and why? The expected answer is XTTS prosody conditioning, because 0.048 against 0.927 is a far larger gap than 3.3 BLEU.


### 5.5  [standard]

> **What actually fails in your pipeline, and what does the system do when it fails?**


In the 870-row research run, 868 succeeded and 2 failed, a 0.23% failure rate, and both failures had the same cause: XTTS refuses a reference clip shorter than 0.33 seconds, and the two cases were backchannels, one "Mm-hmm" and one "Yeah". The other guard is in the app, where the time-scale factor is clamped to between 0.5x and 2.0x, and we count the clamps and display the count so the user knows a segment was not fully fitted. What is missing I should state plainly: there is no retry, no fallback and no per-segment recovery, so a refused segment currently produces no audio. The right fix is to pass the original audio through for that slot, which is also the correct behaviour for a backchannel. And 0.23% is a figure on curated DRAL fragments, not on real video.


*Follow-up to expect:* How does it behave on a clip with two speakers? Concede there is no diarisation, since Objective 2 was not implemented, so every segment is cloned from whichever reference is supplied and multi-speaker correctness is simply unaddressed.


### 5.6  [standard]

> **Where exactly does the retiming stage live, and why is it not in your 870-row evaluation? That looks like you are reporting one system and demonstrating another.**


The retiming lives only in the web app, inside the FastAPI backend, after XTTS and before the ffmpeg remux, where pyworld time-scales each segment to its Whisper slot. It is deliberately absent from the research evaluation, because those 870 rows measure the uncorrected system: the 1.522x over-length, 4.07 seconds generated against a 2.67 second target, is the baseline deficit we are trying to quantify. Retiming before measuring would have hidden the finding. The converse is the fair criticism, and I accept it: the retiming is evidenced only at small scale, one clean clip moving from 0.785 to 1.001, plus a controlled 150 Hz signal where durations scale exactly 0.6x to 2.0x and measured F0 stays at 150.3 Hz. Re-running all 870 rows with retiming enabled is the experiment we have not done.


*Follow-up to expect:* So you cannot claim the app improves duration realism at population level? Agree: the provable claims are that retiming changes duration only and fits the slot on the cases tested; the population claim is unmeasured.


### 5.7  [**HARD**]

> **Direct speech-to-speech translation models exist. Defend your choice of a cascade instead of an end-to-end system.**


Two reasons, one about data and one about diagnosis. On data, we have 2,893 paired fragments, 2,352 after the quality gate, from 96 speakers, which is nowhere near enough to train a direct speech-to-speech model; the cascade lets us reuse pretrained Whisper and XTTS and fine-tune only 74 million MarianMT parameters in about 12 minutes. On diagnosis, a cascade is measurable stage by stage: I can say ASR costs 1.8 to 3.3 BLEU, and that XTTS is where prosody dies, F0 correlation 0.048 against a human-to-human 0.927. An end-to-end model gives one number and no attribution. The genuine cost is that there is no gradient path from the output audio back to the translation, so timing and prosody cannot be jointly optimised, and that is precisely the failure the 0.048 describes.


*Follow-up to expect:* So an end-to-end model would beat you on prosody? Likely yes, because it can condition on source acoustics; the student should argue the cheaper move is to feed XTTS the F0 the ridge model already predicts, not to rebuild the system.


### 5.8  [**HARD**]

> **You report about 13x realtime on an M1. Is this deployable, and if not, what is the path?**


At 13x realtime a 15 second clip takes about 3 minutes, so a five minute video is roughly an hour. That is an offline batch job, not an interactive service, and I would deploy it as a queued job with progress rather than pretend otherwise. The helpful structural fact is that the work is per-segment and therefore close to embarrassingly parallel: segments are independent through translation, synthesis and retiming, and today we run them serially on CPU-class hardware. So the obvious wins are GPU inference for XTTS, batching segments, and computing the speaker conditioning latent once per speaker rather than per segment. But I have to be honest that we never profiled per stage, so I cannot tell you whether XTTS is 80% or 95% of that 13x. Producing that profile is the first thing I would do.


*Follow-up to expect:* Give me a number you would target. Do not invent one: say the deployable target is under 1x realtime for a queued service, and that only the missing profile determines whether GPU XTTS reaches it.


### 5.9  [**HARD**]

> **If I hand your repository to another student, what guarantees they get your numbers back? And separately, what did you actually test?**


On reproducibility: seed 498 throughout, splits made speaker-disjoint by union-find over connected speaker IDs at 70/15/15, and the quality gate stated numerically, voiced-frame ratio at least 0.30, duration between 0.4 and 20 seconds, mean F0 within 60 to 400 Hz, dropping 541 of 2,893 pairs, 18.7%, with reasons logged. Crucially the pretrained and fine-tuned MarianMT are decoded identically, beam 4 and 128 maximum tokens, so the plus 2.34 and plus 3.23 BLEU cannot be a decoding artefact. The ridge fit is closed-form numpy, so no library randomness at all. On testing, the checks that earned their keep were property-style: the 150 Hz signal proving retiming changes duration only, F0 staying 150.3; and the ridge nesting baseline B2 as lambda grows, confirmed when lambda 1e6 was actually selected. There is no unit-test suite or CI for the web app.


*Follow-up to expect:* Then how do you know the web app is correct? Concede it is verified only by running it end to end and listening; the retimer is the single component with a real test, while the segment-slot mapping and ffmpeg remux have no automated coverage.


### 5.10  [**TRAP**]

> **Your app time-scales every segment to fit its original slot, and you get a duration ratio of 1.001. That means you have solved prosody preservation in the app, does it not?**


No, and I would resist that reading. Retiming forces each segment to the source slot; it does not predict the duration a human speaker would produce. Those are different targets: in DRAL the Spanish re-enactment is on average 6.8% longer than the English, ratio 1.068, and Spanish is slower at 3.042 against 3.407 words per second. So fitting the source slot actively works against the natural expansion. It is the right engineering choice for dubbing and the wrong claim for prosody. Second, retiming provably touches duration only, since on the 150 Hz test the pitch stayed 150.3 Hz, which means pitch is untouched: generated-to-human F0 correlations are 0.048 and 0.133 against a human-to-human 0.927. And the ridge duration model, the only component that predicts a target duration, is not in the app at all.


*Follow-up to expect:* So what is your prosody claim? The corpus finding that pitch level transfers at r=0.927 while variation transfers at only r=0.307, plus the ridge F0-range gain over B2 of 9.54% and 14.65% with speaker-level p of 0.0419 and 0.0166; the app claim is timing alone.


---


# Part 5 — Where you are genuinely weak

This is the adversarial pass: the questions most likely to actually damage you, that the five
examiners above under-weighted. Read this section twice.


## Your strongest accurate one-sentence defence

> By exploiting the same-speaker parallel structure of DRAL to establish a human cross-language upper bound, this project builds a working bidirectional English-Spanish voice-cloning dubbing cascade and then measures it strictly enough — speaker-clustered, with nested baselines — to establish a specific dissociation that the field's utterance-level literature had not quantified: speaker timbre transfers to 94.5 and 86.1 per cent of the human ceiling while prosody does not transfer at all (F0 correlations of 0.05 to 0.14 against a human-human 0.93), and within pitch itself, level should be copied across languages whereas per-utterance range should be shrunk toward the speaker's habitual range rather than copied.


## Volunteer these before you are asked

Being caught is far worse than conceding. Say each of these yourself, early.


1. "Emotional tone" is in our stated Objective 1 and we never measured it — and our own correlations argue against it. Restate the delivered scope on the objectives slide as speaker timbre plus, in the app, utterance timing.

2. Objective 2 was not implemented: there is no diarization and no pause-based segmentation. Whisper segments are ASR/VAD boundaries, not speaker turns. There is also no lip-sync; the video stream is copied unchanged.

3. There are two systems in this thesis. The 870-row evaluation is raw XTTS with no correction — that is why 1.522x is a baseline number. The retiming exists only in the app, and the app is validated on one clip and a synthetic tone, not evaluated.

4. Nothing in the delivered pipeline consumes a prediction from the prosody model. It is a specification for a controller we did not build.

5. Ridge regression is not deep learning. Whisper, XTTS-v2 and ECAPA were used exactly as downloaded, unmodified. The only trained component is the MarianMT fine-tune in two directions.

6. The BLEU gain is register and style adaptation, not improved translation adequacy, because DRAL Spanish is a re-enactment rather than a translation. In several cases the pretrained output is the better standalone translation — "bicicleta" to "bici", "presa" to "represa", "mutuamente" to "el uno al otro". Show these before anyone asks what the 2.34 BLEU bought.

7. The "BLEU 61" in the earlier progress report was back-translation BLEU, a materially easier task. The 23.98 here is against an independent human reference. Both are correct; they are different metrics.

8. The test split has 357 pairs but only 14 speakers, and one speaker supplies 236 of the 870 synthesis rows. That is why every prosody comparison is reported with per-speaker Wilcoxon and a speaker-cluster bootstrap alongside the naive test.

9. For F0 mean, ridge selected lambda equal to 1e6 in both directions — the model chose to become baseline B2. Present that as a negative result about conditional predictability, not as a footnote. And note that the naive p=0.0014 does not survive speaker clustering (p=0.135).

10. The F0-range result is regression to the mean. Say so first, and show that the fitted residual weight of about minus 0.60 is what the corpus correlation of r=0.307 predicts, as a consistency check between two chapters.

11. Six ridge fits produced three p-values below 0.05. State the multiple-comparison position yourself, including what Holm at m=6 does to 0.0419, 0.0353 and 0.0166.

12. For log_dur_ratio, baselines B1 and B2 are mathematically identical, hence p=1.0000. Say it so nobody reads it as a bug or as a win.

13. The TTS WER of 0.081 and 0.087 is text fidelity against the MT hypothesis the synthesiser was given, not intelligibility, and it is not comparable to the 0.150 and 0.221 measured on human speech.

14. There is no external system baseline anywhere in this work, and no impostor (different-speaker) control for the ECAPA cosine. Compute the impostor distribution before the viva and bring it; if you cannot, declare the omission before it is found.

15. The same-speaker design is what makes the human ceiling possible and is also why these results do not generalise to real dubbing, where the source and target speakers differ.

16. The quality gate dropped 18.7 per cent of pairs, with reasons; 8 pairs were retained with F0 pinned at the search bound, and they were kept deliberately because filtering rows by their error size would be target-dependent selection.

17. Energy was excluded as a target because RMS varies by 29 dB within a single conversation, which makes absolute RMS a microphone measurement rather than a prosody measurement.

18. PESQ and STOI are inapplicable — cross-lingual synthesis has no time-aligned reference — and no MOS study was run, for ethics and recruitment reasons. Say this before being asked why perceptual metrics are missing.

19. The synthesis evaluation runs on the full 435-pair test split while the prosody evaluation runs on the gated 356/357 rows. Label which table sits on which population, and explain that the gate is an F0-measurability gate rather than a quality filter.


## The eight dangerous questions


### D1. Objective 1 does not say you preserve the voice. It says you preserve the voice AND the emotional tone. Show me your emotion result.

**Why this is dangerous:** "Emotional tone" is written into the stated objective, and there is no emotion metric anywhere in the project: no classifier, no annotation, no valence/arousal measure, no listening test. Worse, every measurement of the channel that actually carries emotion points the other way — generated-vs-target F0 mean r=0.048/0.133 against a human-human ceiling of 0.927, and F0 range r=0.017/0.141 against 0.307. So this is not an unmeasured nice-to-have; the project holds direct quantitative evidence AGAINST its own stated objective. Any bluff here ('XTTS picks up emotion from the reference clip') is instantly refutable: XTTS is conditioned on the reference for timbre and is never given the source pitch contour, and the correlation table proves it.


**Best honest answer:** Concede without hedging, then out-argue the panel on the strength of the negative result. "We did not measure emotion, and I will go further: our own numbers are evidence that emotional tone is not preserved. Generated-to-target F0 mean correlates at 0.048 and 0.133 where the same human's own two recordings correlate at 0.927; F0 range at 0.017 and 0.141 against 0.307. The mechanism is explicit — XTTS conditions on the reference audio for timbre only, it is never given the source F0 contour, so there is no path for the emotional signal to travel. Objective 1 as written was too broad and I would restate it as preserving speaker timbre, and in the app, utterance timing. What we can defend is that we quantified the failure instead of asserting success, and we could only quantify it because DRAL is same-speaker parallel and gave us a human upper bound to fall short of. Two further points on measurability: DRAL is neutral-to-mildly-expressive conversational re-enactment, so even a correct emotion evaluation on this corpus would barely cover the emotion space, and a proper claim needs an emotion-labelled corpus and a listening study, neither of which we had. The prosody predictor is the first step toward closing the gap — it specifies what the target F0 mean and range should be — but it is not connected to the synthesiser."


**Prepare before the panel:** Restate the objective on the slide, visibly, so the scoping reads as honesty rather than concealment. Put the four generated-vs-target correlations and the two human ceilings on one slide next to each other. Be able to explain in one sentence why XTTS structurally cannot transfer emotion (speaker-encoder conditioning, no pitch-contour input, no emotion embedding). Have the future-work path costed: predicted F0 and duration passed to a controllable TTS or applied post-hoc with WORLD, plus an emotion-labelled corpus for a real metric.


### D2. Point to the place in your system where a prediction from your prosody model is actually used.

**Why this is dangerous:** Nowhere. The 870-row evaluation used raw XTTS; the web app's retiming targets come from Whisper segment timestamps, not from the ridge model's predicted log-duration ratio. The single largest piece of statistical work in the thesis has no consumer, which lets the panel reframe the project as two half-projects stapled together — a descriptive study of someone else's corpus, and an integration of three downloaded models — with no causal link between them. This is the classic 'so what' and it is fatal if answered by waffle.


**Best honest answer:** Own the disconnection, then defend the study as a specification rather than a component. "Correct: no prediction is consumed anywhere. The prosody model is not a module of the delivered pipeline, it is the feasibility study we needed before building the module — is target prosody predictable from the source at all? For F0 mean the answer is no: ridge selected lambda equal to 1e6 in both directions, which means it chose to collapse to the copy-source-plus-offset baseline it nests, so there is no conditional signal beyond copying. That is a usable result — copy is already the right rule. For F0 range copy is provably wrong: shrinking toward the speaker's habitual range beats copy-plus-offset by 9.54 and 14.65 per cent MAE, with per-speaker p of 0.042 and 0.017 and cluster bootstrap CIs excluding zero. So the deliverable of that study is a design rule for a controller: copy the pitch level, shrink the pitch variation. I agree the controller is unbuilt." Then be concrete about the wiring so it is clearly a next step and not a hole you never noticed: the predicted log-duration ratio is exactly the quantity the app's retiming currently takes from Whisper timestamps, and the predicted F0 range is a WORLD-side scaling of the semitone contour — both are one function call from existing code. And say why it was not done: with no MOS study and with PESQ/STOI inapplicable to cross-lingual synthesis, we had no way to validate whether the intervention improved anything, so shipping it would have been unmeasured.


**Prepare before the panel:** Draw the architecture with a dotted arrow from the prosody model into the TTS stage, labelled 'specified, not implemented'. Name the two exact insertion points. Have the counterfactual ready — wiring it in without a listening study would have produced an unevaluable change. If there is any time before the viva, apply the predicted F0-range scaling to a handful of test outputs so you can at least say you have heard it.


### D3. What is the ECAPA cosine between two different speakers on your data? Because until I know that, 0.42 and "94.5 per cent of ceiling" tell me nothing.

**Why this is dangerous:** This is the missing negative control, and its absence undermines the project's headline identity claim. Every identity number is a ratio between two quantities of unknown scale: 0.4206/0.3832 over a same-speaker cross-language ceiling of ~0.445. That ceiling is low by ECAPA standards for a single human, so the impostor distribution may not sit far below 0.42 — and if different-speaker cosine on this data turns out to be, say, 0.33, the claim that speaker identity is preserved largely evaporates. If the student cannot produce the floor, the panel is entitled to assume the worst, and this is one of the very few numbers in the project that could be computed in an afternoon and was not.


**Best honest answer:** The genuinely strong answer is to compute it before the viva and bring three numbers: impostor mean, cross-language same-speaker ceiling, same-language same-speaker ceiling. If it truly cannot be computed: "The floor is the one control we did not report and I accept the absolute claim is not licensed without it. What survives without a floor is a relative claim that is internally valid: the ceiling is 0.4452 for en-es and 0.4450 for es-en — near-identical, as it must be, since both are the cosine between the same speaker's real English and real Spanish. Against matched ceilings, the gap between 0.4206 and 0.3832 is a genuine direction-dependent property of the synthesis: cloning works better into Spanish than into English, and that comparison does not depend on knowing the floor. What I would not now say is 'identity is preserved'; I would say the clone reaches 94.5 and 86.1 per cent of the distance the same human's own cross-language recordings achieve, and that a verification-style evaluation with an impostor distribution and an EER is required before the stronger claim." Add the diagnostic point that shows you understand your own metric: a same-speaker cross-language cosine of only 0.445 tells you ECAPA is itself substantially language- and channel-sensitive, which is exactly why raw cosine values must be read against both a floor and a ceiling and never in absolute terms.


**Prepare before the panel:** Run it. The embeddings are already on disk: mean and SD of cosine over all non-matching speaker pairs in the 14-speaker test split, within and across language, plus the same-speaker same-language ceiling. One slide with floor, two ceilings, and the two generated values; an EER and where 0.42 falls on the ROC if you can manage it. Also prepare the honest sentence about what fraction of the 0.445 shortfall is language versus channel.


### D4. Your F0-range result is regression to the mean and nothing else. Your own corpus study reports r=0.307 between English and Spanish F0 range with equal standard deviations of 1.295, so the optimal slope on the source is about 0.31 and the residual weight should be about minus 0.69 — you fitted minus 0.60. What did the model learn that your correlation had not already told me?

**Why this is dangerous:** It attacks the one result the student is entitled to call a finding, and the arithmetic is hard to escape: the 'learned' weight is very nearly the mechanical consequence of a correlation the student reported themselves in a different chapter. If the student oversells it as having learned a cross-lingual prosody mapping, they lose credibility on everything else in the thesis. The other examiners pushed toward more capacity; this pushes the opposite way and is the more dangerous direction, because conceding it appears to leave the modelling chapter with no positive result at all.


**Best honest answer:** Say 'regression to the mean' yourself, before the panel finishes, and convert the agreement into a validation argument. "Yes — the mechanism is shrinkage and I will not describe it as more than that. I would put it more strongly than you did: the two analyses are quantitatively consistent. With both range SDs at 1.295 and r=0.307, the least-squares slope is 0.307 and the residual weight on the source should be about minus 0.69; on speaker-disjoint held-out speakers we fitted minus 0.6034 and minus 0.5916. Two independent estimates from different splits agreeing to that tolerance is evidence the effect is stable, not evidence that it is empty. It is worth reporting because copy-source is what every cascade dubbing system implicitly does — XTTS handed a wide-range reference will not shrink it — and we show copy is the wrong estimator for pitch variation and price the error at 9.54 and 14.65 per cent MAE with speaker-clustered significance. The claim is 'per-utterance pitch variation does not transfer across languages and should be shrunk toward the speaker's habitual range', not 'we learned a mapping'." Then give the contrast that makes the pair of results genuinely informative: the identical pipeline recommends opposite treatments for two components of the same acoustic parameter — copy the pitch level (r=0.927, lambda drove to 1e6, ridge became the baseline) and shrink the pitch variation (r=0.307, shrinkage wins). That dissociation is the result, and neither number alone would have shown it.


**Prepare before the panel:** Put 1-r next to the fitted weight for both directions on one slide and present it as a consistency check you ran deliberately. Rehearse saying the words 'regression to the mean' first. Have the multiplicity answer loaded in the same breath — six fits, three p-values under 0.05, and what Holm at m=6 does to 0.0166, 0.0353 and 0.0419 — because these two questions will arrive together.


### D5. Your synthetic speech scores WER 0.081 and 0.087. Real human speech through the same ASR scores 0.150 and 0.221. Explain to me how your TTS is nearly twice as intelligible as a human being.

**Why this is dangerous:** As reported, the number invites a false reading, and a panel will use it to test whether the student understands their own metrics rather than just their pipeline. The two figures have different references and different signal conditions: the human WER is Whisper against a human transcript of spontaneous, disfluent conversational re-enactment; the TTS WER is Whisper against the very MT string XTTS was instructed to say. It is a closed loop. On top of that the generated audio is 1.522 times slower than the human original, which makes ASR easier, and 1 and 2 empty transcriptions were folded in. If the student defends the comparison instead of dismantling it, every other metric in the thesis becomes suspect.


**Best honest answer:** Refuse the comparison and relabel your own metric. "They are not comparable, and I should have named them differently in the table. The 0.081 figure is a synthesis-fidelity check: the reference is the exact MT hypothesis XTTS was conditioned on, so it measures whether the synthesiser produced the words it was given. That matters, because truncation, dropped segments, repetition loops and empty output are XTTS's characteristic failure modes and we did observe 1 and 2 empty transcriptions. It is not intelligibility. The 0.150 and 0.221 figures are a genuinely harder task — Whisper on spontaneous conversational speech against an independent human transcript with disfluencies and backchannels. Two further reasons the synthetic number is flattered: the audio is clean, single-speaker, non-overlapping, and it is 1.522 times slower than the original, which is easier for an ASR. The correct statement is that the TTS reproduces its input text reliably, not that it is more intelligible than a human." Pivot to what the number does buy: it bounds the synthesis stage's contribution to cascade error, which is what licenses attributing the measured cascade loss — BLEU 23.98 to 22.14 and 26.79 to 23.45 — to ASR rather than to TTS.


**Prepare before the panel:** Rename the row in the results table to something like 'TTS text-fidelity WER (reference: MT hypothesis)' and flag the relabelling unprompted. Have the 1.522x slowdown ready as an explanation. If you can, add the matched control — Whisper on the human TARGET audio against its human transcript — so there is a like-for-like number on the slide.


### D6. Set your internal baselines aside. What other system did you compare against — any published speech-to-speech model, any dubbing tool, any ablation of your own cascade — on the same test set?

**Why this is dangerous:** The answer is none. Every comparison in the thesis is internal: pretrained versus fine-tuned MarianMT, B0/B1/B2 versus ridge, generated versus human ceiling. With no external reference point, '94.5 per cent of ceiling' and '0.23 per cent failure rate' cannot be judged good or bad, and the claimed contribution shrinks to 'we integrated three downloaded models'. It is made worse by an obvious free ablation that was not run: Whisper has a built-in X-to-English translate task, which for the es-en direction would eliminate the MT stage and its error propagation entirely — the student's only trained component solves, in one direction, a problem their downloaded ASR already solves.


**Best honest answer:** Concede the gap plainly, defend the substitute on its own terms, and be precise about what you would have run — vagueness here is what does the damage. "There is no external system baseline and that is the largest gap in the evaluation design. What we used instead is a human upper bound rather than a system lower bound: because DRAL is same-speaker parallel, we can measure what the same person's own voice scores across languages and express the system as a fraction of that. In one respect that is a stronger reference than a competing system, because it is a fixed ceiling rather than a moving target, and it is the only reason the direction asymmetry — 94.5 against 86.1 with ceilings of 0.4452 and 0.4450 — is interpretable at all. But it says nothing about whether another system would do better. The comparisons I should have run, in priority order: first, Whisper's own X-to-English translate task for es-en, which removes the MT stage and is nearly free — it cannot serve en-es, so it is not a replacement for a bidirectional system, but it is the obvious ablation and I did not run it; second, SeamlessM4T as a published direct-S2ST reference; third, YourTTS as a cloning reference, with the caveat that its published prosody numbers are frame-level F0 correlations and are not comparable to my utterance-level statistics." Also point out the one place you ARE externally anchored: pretrained opus-mt is a published system, so BLEU 23.98 and 26.79 are real external reference points for the MT stage, and the fine-tuned deltas are measured against them under identical decoding.


**Prepare before the panel:** Run the Whisper-translate ablation for es-en if there is any time at all — it is effectively a flag change and yields a BLEU number on the same 432 test sentences, which converts a hole into a result. Otherwise state the compute cost and why it was cut. Prepare one slide titled 'what we did not compare against, and what we used instead', so the concession is yours rather than the panel's.


### D7. Objective 3 is the only objective you claim you fully met, and its entire evidence base is one clean clip and a synthetic 150 hertz tone. So which system am I examining — the 870-row pipeline, or the app?

**Why this is dangerous:** Two problems compound. First, the evaluated system and the demonstrated system are different: the 870 rows are raw XTTS with no retiming; the app has retiming. Second, the app's evidence is n=1 real clip plus a correctness check on a synthetic signal — no WER, no speaker similarity, no clamp statistics on real data, no multi-speaker test. And because Objective 2 was never implemented, multi-speaker input is undefined behaviour: Whisper segments are ASR/VAD segments, not speaker turns, so a segment spanning a turn change gets one blended reference voice. Essentially every real dubbing use case is multi-speaker. This is the point where 'not implemented' and 'not evaluated' meet on the objective the student is most confident about, and confidence is exactly what makes it dangerous.


**Best honest answer:** Separate the two claims yourself and do not defend the app as evaluated. "You are examining two things and that should have been on my first slide. The 870-row study is the measurement, deliberately with no correction applied, which is why the 1.522x duration inflation is a clean baseline figure for raw XTTS rather than a bug. The app is the engineering demonstration for Objective 3, and it is validated, not evaluated: on a controlled 150 hertz signal we confirmed the WORLD time-scaling changes duration only — scale factors 0.6 to 2.0, measured F0 stable at 150.3 hertz — and on one real clip the duration ratio moved from 0.785 to 1.001. That is a correctness check on the retiming operator, not an evaluation of dubbing quality, and I am not claiming otherwise. On multi-speaker input: there is no diarization, so the app assumes a single speaker. It takes per-segment reference audio, which incidentally follows a speaker change that falls on a segment boundary, but a segment containing a turn change will produce a blended voice. We never tested it and I will not claim it works." Then pivot to the smallest honest fix and, crucially, to the cost you have not yet priced: retiming can be run over the existing 870 rows as a post-process to give a corrected duration ratio at n=870, and forcing a 1.522x-long signal into its original slot means routine compression near 0.66x, so the speaker similarity and text-fidelity WER should be re-measured after retiming — the naturalness cost of the correction is currently unmeasured.


**Prepare before the panel:** One slide, two columns: 'measured system' and 'demonstrated system', with what each was evaluated on. Run the retiming over the 870 rows if at all possible — it is a post-process on audio you already have, and it turns n=1 into n=870, converting the weakest claim in the thesis into a new result. Know the clamp rate and be able to describe what a clamped segment sounds like. Take a two-speaker clip through the app before the viva so its behaviour is not discovered live.


### D8. Three of you are submitting this. There are exactly two things trained: a twelve-minute fine-tune of a downloaded model, and a closed-form fit that runs in under a second. Partition the work for me — what is each person's individually assessable contribution, and is there enough here for three?

**Why this is dangerous:** This is the question the technical examiners skip and the moderator always asks. It exposes that the 'deep learning' in the title amounts to continued training of an off-the-shelf 74M model for 12.4 and 10.8 minutes, that the second model is not deep learning at all, and that Whisper, XTTS and ECAPA were used exactly as downloaded. 'We all worked on everything' reads as either an inflated headcount or an uneven split being concealed, and it invites the panel to divide the visible work by three and find each share thin. It is also the moment a student can accidentally claim a teammate's work and be caught by a follow-up they cannot answer.


**Best honest answer:** Reject the premise that GPU minutes measure the work, then give a hard partition with named owners. "Training time is the wrong unit — the fine-tune is twelve minutes because 74 million parameters on 1,998 sentences converges in two epochs, which is a fact about the problem, not about the effort. The assessable artefacts are: pairing DRAL through the official trans_id metadata rather than filename substitution, which recovers valid pairs whose conversation IDs differ; the speaker-disjoint split by union-find over connected speaker IDs; the quality gate with full drop accounting, 541 of 2,893 with overlapping reasons; the corpus study over 2,352 same-speaker pairs; the speaker-clustered inference machinery — per-speaker Wilcoxon plus speaker-cluster bootstrap — which is where most of the statistical care sits and which overturned one of our own conclusions, naive p=0.0014 down to speaker p=0.135; the 870-row end-to-end evaluation harness with the human-ceiling protocol; and the full-stack application with a verified WORLD retiming stage." Assign each to a named student, say honestly which were shared, answer at depth on your own and say plainly "that was X's work, I can describe it but they own it" on the others — that reads as integrity, not ignorance. On the label, concede it: two fine-tunes of off-the-shelf models plus a ridge fit is modest deep-learning novelty; the contribution is evaluation and integration, and the title should have said so.


**Prepare before the panel:** Bring a one-page contribution matrix, artefact by owner, agreed by all three and signed off by the supervisor. Each student must be able to reproduce their own numbers from the repo and name one design decision they personally made and why — for example, choosing the one-standard-error rule for lambda. Agree in advance who fields which topic, so you never contradict each other in the room. And check that nobody's slides claim a component they did not build.
