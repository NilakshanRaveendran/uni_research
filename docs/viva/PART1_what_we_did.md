# Part 1 — Direct answers to your four questions

Every number here comes from your actual result files, re-read today. Nothing is from memory.

---

## Q1. Did we train or fine-tune anything?

**Yes — exactly two things. Everything else is used exactly as downloaded.**

| Component | Trained by us? | What it is | Time |
|---|---|---|---|
| Whisper `small` | ❌ **No** | ASR, used as downloaded | — |
| **MarianMT** | ✅ **YES — fine-tuned** | ~74 M-parameter translation model, continued training on your DRAL text | 12.4 min (en-es), 10.8 min (es-en) |
| XTTS-v2 | ❌ **No** | voice cloning / TTS, used as downloaded | — |
| ECAPA (`spkrec-ecapa-voxceleb`) | ❌ **No** | speaker embeddings, used as downloaded | — |
| **Ridge prosody model** | ✅ **YES — fitted** | 6 weights, closed-form, *statistical* not deep learning | < 1 second |

### The distinction that matters for the panel

**MarianMT fine-tuning is genuine deep-learning training.** Gradients, an optimiser (AdamW,
lr 2e-5), batches of 8, four epochs, gradient clipping at 1.0, running on the Mac's GPU (MPS).
The model's ~74 million internal weights were modified and the result saved to disk.

**The ridge model is a statistical fit, not deep learning.** Six weights, solved by one closed-form
matrix equation — no epochs, no gradient descent. It is legitimately a *trained model* (fitted on
train, selected on dev, evaluated once on test) but you must **not** call it deep learning.

If asked "where is the Deep Learning in your title?" the honest answer is: the pretrained
components are all deep neural networks, and MarianMT was additionally fine-tuned by us. The ridge
model is a deliberately simple statistical layer on top.

---

## Q2. What is "synthesis" and what did the synthesis run actually do?

**Synthesis = generating speech.** A computer producing an audio waveform.

The **synthesis run** pushed every test pair through the complete system and saved the resulting
audio. Per row, five model invocations:

1. **Whisper** listens to the source audio → text
2. **MarianMT** translates that text
3. **XTTS-v2** speaks the translation in the source speaker's cloned voice ← *this is the synthesis*
4. **Whisper again** re-transcribes the generated audio (to measure intelligibility)
5. **ECAPA** computes two voice fingerprints (to measure identity)

Plus prosody extraction on three files (source, generated, human target).

**Scale:** 435 test pairs × 2 directions = **870 rows**, each producing a real `.wav` file.

**Result: 868 succeeded, 2 failed, failure rate 0.23%.** Both failures were XTTS refusing reference
audio shorter than 0.33 s — the two clips were the backchannels *"Mm-hmm."* and *"Yeah."*

**No training happens during the synthesis run.** It *uses* the models; it never modifies them.

### Why it took ~5 hours when training took 12 minutes

Three reasons compound:

1. **Audio is vastly bigger than text.** All 1,998 training sentence pairs are ~166 KB. The
   generated audio is ~150 MB. One second of speech is 24,000 numbers.
2. **Five models per row, not one.** A training step ran one model once; each synthesis row runs
   five plus feature extraction.
3. **Generation is inherently sequential.** In training the correct answer already exists, so 8
   sentences are checked simultaneously on the GPU. In generation nothing exists yet — XTTS must
   invent the waveform one chunk at a time, each depending on the previous. Nothing parallelises.
   And XTTS runs on the **CPU** here (the stable configuration on Apple Silicon) while training
   used the **GPU**.

So: teaching the model was fast; making it produce 870 audio files was slow.

---

## Q3. How many datasets are there?

**One. DRAL.** That is the only corpus you collected, processed, split, or evaluated on.

- DRAL 16 kHz release
- **2,893** paired EN/ES fragments
- **96** unique speakers
- **5,786** wav files
- **2,834** same-speaker pairs (one bilingual person performs *both* languages)
- Total speech ≈ 4.3 hours

**But be precise in the viva, because there is a second layer.** The four pretrained models were
each trained by their authors on *other* corpora, which you did not touch:

| Model | Trained by its authors on |
|---|---|
| Whisper | ~680,000 h of multilingual web audio (OpenAI) |
| MarianMT | OPUS — books, subtitles, EU documents (Helsinki-NLP) |
| XTTS-v2 | Coqui's multilingual TTS corpora |
| ECAPA | VoxCeleb (thousands of speakers) |

So the correct answer is: **"One dataset for our experiments — DRAL. The pretrained components
carry knowledge from their own training corpora, which we did not modify."** That distinction is
exactly the kind of thing a panel checks you understand.

---

## Q4. Did we use any other audio besides English and Spanish — for pitch, waveforms, or anything?

**No. Verified today, not assumed.**

- All **870** synthesis rows reference only `EN_*` and `ES_*` audio files
- The prosody feature table contains strictly `en_*` and `es_*` columns — no third prefix exists
- Directions are only `en-es` (435) and `es-en` (435)

**Nothing extra was used for pitch or waveform analysis.** Every F0, energy, duration and
speaking-rate number comes from the *same* English and Spanish DRAL recordings that the rest of the
study uses.

Two clarifications worth having ready:

- The DRAL release *does* contain other material — `fragments-long`, full `recordings`, and some
  Japanese files. **None of it was used.** Only `fragments-short` EN/ES pairs.
- Pitch was measured with **librosa's `pyin`** algorithm, which is a signal-processing method
  applied to your own audio — not a model, and not trained on anything.

---

## The one-line summary of what was built

A cascaded English↔Spanish speech-to-speech system (Whisper → MarianMT → XTTS-v2) evaluated on
870 held-out utterances; a fine-tuned MarianMT compared against its pretrained baseline; a corpus
study of how prosody corresponds across languages in 2,352 same-speaker pairs; a ridge model
predicting target-language prosody from source-language prosody; and a web application that dubs an
uploaded video and retimes each segment so the dub keeps the original rhythm.
