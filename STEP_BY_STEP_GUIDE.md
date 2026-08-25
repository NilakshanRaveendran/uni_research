# Step-by-Step Execution Guide
## Bilingual Voice Transcription & Synthesis Preserving Speaker Identity (ES ↔ EN)

**Who this is for:** you, starting from zero, finishing the research in ~4 weeks.
**Companion files:** `s2st_pipeline.py` / `s2st_pipeline.ipynb` (runnable code), `ROADMAP_4WEEKS.md` (strategy).
**Two tracks:**
- **Track A (required): zero-shot pipeline** — Phases 1–9. This alone completes the project.
- **Track B (optional): training/fine-tuning** — Phase 10. Do it *only after* Track A works end-to-end. It upgrades your report from "applied pipeline" to "pipeline + fine-tuning experiments".

---

## PHASE 1 — Environment setup (Day 1, ~1 hour)

### 1.1 Get a GPU runtime
- Open [Google Colab](https://colab.research.google.com) → New notebook → `Runtime → Change runtime type → T4 GPU` (Colab Pro: A100/L4 if offered).
- Alternative: [Kaggle Notebooks](https://www.kaggle.com/code) gives ~30 GPU-hours/week (P100/T4) — good as your second account for batch runs.

### 1.2 Install everything (one cell)
```python
!pip -q install -U openai-whisper transformers sentencepiece sacremoses
!pip -q install -U coqui-tts                    # XTTS-v2 (import name: TTS)
!pip -q install -U resemblyzer jiwer sacrebleu soundfile librosa
!apt -qq install -y ffmpeg
```

### 1.3 Verify the GPU and accept XTTS terms
```python
import os, torch
os.environ["COQUI_TOS_AGREED"] = "1"     # XTTS non-commercial licence (fine for academic)
print(torch.cuda.get_device_name(0))     # should print T4 / L4 / A100
```

### 1.4 Mount Google Drive (so models/outputs survive disconnects)
```python
from google.colab import drive
drive.mount('/content/drive')
WORK = "/content/drive/MyDrive/s2st_project"
import os; os.makedirs(WORK, exist_ok=True)
```

**✅ Checkpoint:** GPU prints, no install errors. If `coqui-tts` clashes with torch → `!pip install "transformers>=4.40"` and restart runtime.

---

## PHASE 2 — Load models & test each stage in isolation (Day 1–2)

Run cells **[2]–[5]** of `s2st_pipeline.py`. Then test each stage on ONE clip before chaining anything. Get a test clip: record yourself saying two Spanish sentences (phone voice memo → WAV), or grab one Common Voice ES clip.

### 2.1 Test ASR alone
```python
text, lang, segs = transcribe("test_es.wav")
print(lang, "→", text)        # expect "es" + a correct Spanish transcript
```

### 2.2 Test MT alone
```python
print(translate(text, "es", "en"))   # expect fluent English
print(translate("The weather is nice today.", "en", "es"))  # reverse works too
```

### 2.3 Test voice-cloning TTS alone (the new part!)
```python
synthesize("Hello, this is a test of voice cloning.", "en",
           speaker_wav="test_es.wav", out_path="clone_test.wav")
from IPython.display import Audio; Audio("clone_test.wav")
```
**Listen:** it should sound like the *speaker of test_es.wav* speaking English. This is the moment the project's title becomes real.

**✅ Checkpoint:** all three stages work individually. Debug here, not in the full chain.

---

## PHASE 3 — First end-to-end run (Day 3) 🎯

```python
r = speech_to_speech("test_es.wav")           # cell [6]+[8]
print("SRC:", r["src_text"]); print("TGT:", r["tgt_text"])
Audio(r["out_wav"])
```
Then flip direction with an English clip — no code changes needed (Whisper auto-detects the language).

Add the metrics (cell [7]):
```python
print("SECS :", secs(r["in_wav"], r["out_wav"]))    # want > 0.6 typically
print("UTMOS:", utmos(r["out_wav"]))                # want > 3.0
print("rtWER:", roundtrip_wer(r))                   # want < 0.15
```

**✅ Milestone (end of Week 1, Day 3):** one Spanish clip in → English out in the same voice, with numbers. **Save these WAVs — they're demo material.**

---

## PHASE 4 — Build the evaluation dataset (Day 4–5)

You need ~200–500 test utterances. No training data required for Track A.

### 4.1 CVSS ES→EN subset (your main benchmark)
CVSS pairs Spanish source speech with English reference translations; **CVSS-T** targets are voice-transferred (ideal anchor for speaker preservation).
1. Download **Common Voice v4 Spanish** (source audio CVSS is built on): https://commonvoice.mozilla.org/datasets — select "Common Voice Corpus 4", language Spanish. (Requires free account; it's a big tar — extract only the clips you need.)
2. Download CVSS ES translations: https://github.com/google-research-datasets/cvss (`cvss_c_es_en` text + target speech tars).
3. Keep the **test split**, take the first ~300 utterances. Build the layout cell [9] expects:
```
data/es/clipXXXX.wav          # Spanish source audio (from Common Voice, converted to wav 16k)
data/refs.tsv                 # "clipXXXX.wav<TAB>English reference translation"
```
Convert mp3→wav: `!ffmpeg -i clip.mp3 -ar 16000 -ac 1 clip.wav`

### 4.2 EN→ES eval set (smaller, reference-free)
- 50–100 English clips from **LibriSpeech test-clean** (https://www.openslr.org/12) or Common Voice EN.
- Text references for BLEU: **FLORES-200** ES sentences (https://github.com/facebookresearch/flores).
- Score this direction with SECS + UTMOS (+ BLEU vs FLORES when you feed FLORES text). State the speech-reference asymmetry openly in the report.

### 4.3 Demo clips
3–5 short (15–30 s) movie/interview clips per language + 2–3 volunteer recordings. Clean, single-speaker, no music underneath.

**✅ Checkpoint:** `batch_eval("data/es", refs_tsv="data/refs.tsv", limit=10)` runs 10 clips without crashing.

---

## PHASE 5 — Full batch evaluation = your results chapter (Week 2)

### 5.1 Main run (cloning ON)
```python
rows = batch_eval("data/es", refs_tsv="data/refs.tsv")   # full 200–500; checkpoints every 20
```
~10–20 s per clip on T4 → budget a few hours; the CSV checkpointing means a timeout never loses work. Copy `outputs/metrics.csv` to Drive.

### 5.2 Baseline run (cloning OFF — the killer comparison)
Re-synthesize the same translations with ONE fixed voice for all clips (cell [10]), then compute SECS the same way.
**Expected story:** cloning SECS ≈ 0.6–0.75 vs single-voice SECS ≈ 0.3–0.45. That gap **is** your headline result — it quantifies "speaker identity preservation".

### 5.3 EN→ES run
Same batch loop over your English set (auto-direction handles it).

### 5.4 Plots (matplotlib, straight from the CSV)
1. SECS distribution: cloning vs single-voice (two histograms/violins)
2. UTMOS distribution
3. WER + BLEU bars per direction
4. Scatter: SECS vs UTMOS (shows no quality-identity trade-off)

### 5.5 Optional SOTA baseline
Request access to **SeamlessExpressive** (gated, ~1 day): https://huggingface.co/facebook/seamless-expressive — run your eval subset through it for one comparison row. Skip without guilt if access is slow.

**✅ Milestone (end of Week 2):** `metrics.csv` for 3 systems (cloning / single-voice / optionally Seamless), 4 plots, mean ± std tables.

---

## PHASE 6 — Human evaluation (Week 3, start Day 1 — recruiting takes time)

1. Pick **12–20 clips** covering both directions, good and bad SECS cases.
2. Google Form, per clip: play source + output, rate 1–5 on **Naturalness**, **Speaker similarity** ("same person?"), **Overall quality**. Host audio via Drive links.
3. Recruit **5–6 bilingual listeners** (classmates). Sessions ≤ 15 min (matches your ethics section).
4. Analyze: mean ± std per axis; correlate human speaker-similarity with SECS (validates the automatic metric); simple inter-rater agreement (e.g., Krippendorff's alpha or mean pairwise Spearman).

---

## PHASE 7 — Prosody/emotion evidence (Week 3, ~2 days)

No training needed — measure that prosody *correlates* between source and output:
```python
import librosa, numpy as np
def prosody(path):
    y, sr = librosa.load(path, sr=16000)
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    return {"f0_mean": np.nanmean(f0), "f0_std": np.nanstd(f0),
            "energy_std": rms.std(), "dur_s": len(y)/sr}
```
Run on every (source, output) pair → report Pearson correlation of f0_mean, f0_std, speaking rate (words/s from Whisper timestamps). Positive correlations = "prosodic characteristics transfer".
**Optional emotion knob:** run **OpenVoice v2** (MIT, https://github.com/myshell-ai/OpenVoice) on 3–4 samples to demo explicit style control — one paragraph + one figure in the report.

---

## PHASE 8 — Stretch: video dubbing demo (Week 3, only if on schedule)

1. `ffmpeg -i clip.mp4 -ar 16000 -ac 1 audio.wav` — extract audio.
2. Run `speech_to_speech`; you already have segment timestamps from Whisper.
3. Time-stretch output to source duration: `ffmpeg -i out.wav -filter:a atempo=<ratio> out_fit.wav` (atempo accepts 0.5–2.0; chain two for more).
4. Background music: separate with `!pip install demucs` → `!demucs --two-stems=vocals audio.wav`, mix your synthesized voice over the `no_vocals` stem.
5. Remux: `ffmpeg -i clip.mp4 -i dubbed.wav -map 0:v -map 1:a -c:v copy dubbed.mp4`.

Ship ONE 30–60 s dubbed clip. Call it "duration-matched dubbing", not lip-sync.

---

## PHASE 9 — Write, package, present (Week 4)

- **Results & Discussion:** metric tables (mean ± std), the 4 plots, cloning-vs-baseline SECS story, MOS results + SECS↔MOS correlation, prosody correlations.
- **Methodology:** update to cascade + zero-shot cloning; redraw the architecture figure (use the pipeline diagram: Whisper → NLLB → XTTS-v2 with reference-clip conditioning).
- **Limitations:** EN→ES reference asymmetry · PESQ/STOI dropped (no time-aligned cross-lingual reference exists — cite this reason) · emotion transfer = analysis not control · duration-matched dubbing, not lip-sync.
- **Future work:** direct S2ST (Translatotron 2, UnitY, SeamlessExpressive), trained style transfer (StyleS2ST), textless S2ST (MSLM).
- **Repo:** clean notebook that reproduces everything top-to-bottom, README with demo audio links, `metrics.csv` committed.
- **Final verification:** re-run 3 held-out clips fresh; every number in the report must be reproducible from the CSV.

---

## PHASE 10 — OPTIONAL TRAINING TRACK (only after Phase 5 works)

You said you can train — good. Here is what is *worth* training in your window, in priority order. Each adds a genuine "we fine-tuned X and measured the delta" experiment to the report. **Never start training before the zero-shot pipeline + eval harness works, because the harness is how you measure whether training helped.**

### 10.1 Fine-tune Whisper on Spanish (BEST value · ~1–2 days · fits Colab T4) ⭐
- **What:** LoRA/PEFT fine-tune `whisper-small` or `whisper-medium` on Common Voice ES (10–20 h subset).
- **Why:** measurable WER improvement on accented/noisy Spanish; strengthens *your* pipeline stage (you own ASR).
- **How:** follow the HF recipe — https://huggingface.co/blog/fine-tune-whisper — swap dataset to `mozilla-foundation/common_voice_17_0`, `language="es"`. Use `peft` + 8-bit to fit a T4 (batch 8, grad-accum 2, ~3000 steps ≈ 6–8 h).
- **Report:** WER before vs after on your held-out ES test set; then run the *whole pipeline* with the fine-tuned ASR and show downstream BLEU/rtWER movement.

### 10.2 Fine-tune XTTS-v2 on target speakers (the "identity" experiment · ~2–3 days · needs ≥16 GB VRAM)
- **What:** speaker-adaptive fine-tune of XTTS-v2 on 5–15 min of ONE speaker's clean speech, vs zero-shot cloning of the same speaker.
- **Why:** directly interrogates the project's core claim — "how much identity does zero-shot capture vs adaptation?" SECS(zero-shot) vs SECS(fine-tuned) is a beautiful table.
- **How:** the maintained fork ships a recipe: https://github.com/idiap/coqui-ai-TTS → `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`. Prepare `metadata.csv` (`wav_path|text|speaker`) from your volunteer recordings; ~needs A100/L4 (Colab Pro) or reduce batch to 1–2 on T4. Train the GPT component only (default recipe), few thousand steps.
- **Data:** record one volunteer reading ~100 sentences (≈15 min speech), 16 kHz mono, quiet room. Transcribe with Whisper to make the metadata.
- **Report:** SECS/UTMOS/MOS for {zero-shot, fine-tuned} on held-out sentences of that speaker.

### 10.3 Fine-tune the MT model (cheap · ~half a day)
- **What:** LoRA fine-tune NLLB-600M (or your MarianMT) on ~50–100 k ES↔EN pairs (Tatoeba / OPUS OpenSubtitles — conversational register matches speech).
- **How:** standard HF `Seq2SeqTrainer` recipe: https://huggingface.co/docs/transformers/tasks/translation — small LR (1e-5…5e-5), 1–2 epochs.
- **Report:** BLEU/chrF before vs after on CVSS + FLORES; note domain-shift effect (subtitles ≈ spoken language).

### 10.4 What NOT to train (say why in the report — it shows judgment)
- ❌ **TTS from scratch** (Tacotron/VITS/XTTS): needs hundreds of hours + weeks of GPU.
- ❌ **Speaker encoder on VoxCeleb:** ECAPA/Resemblyzer are already trained on it; retraining adds nothing.
- ❌ **Direct S2ST (Translatotron-style):** research-lab scale (thousands of GPU-hours).

### 10.5 Training logistics on Colab
- Checkpoint to Drive **every epoch** (`save_steps`), assume disconnection *will* happen.
- Keep an eval cell that runs your Phase-5 metrics on 20 clips — run it after every training session; if a metric regresses, stop.
- Log with `wandb` (free) or a simple CSV — you'll need loss curves for the report figures.
- Time-box hard: any fine-tune that hasn't beaten zero-shot after 2 days of effort becomes one honest sentence in Limitations, and you move on.

---

## Weekly map (both tracks)

| Week | Track A (required) | Track B (optional training) |
|---|---|---|
| 1 | Phases 1–3: env, stage tests, first cloned output, metrics on 1 clip; start Phase 4 data | — (don't) |
| 2 | Phase 5: full batch eval + baseline + plots | 10.1 Whisper fine-tune in parallel (2nd Colab/Kaggle account) |
| 3 | Phases 6–7 (+8 stretch): human study, prosody, dubbing | 10.2 XTTS speaker adaptation OR 10.3 MT fine-tune (pick ONE) |
| 4 | Phase 9: write, repo, slides, verify | fold training deltas into Results |

## Success criteria (know when you're done)

- [ ] One-call `speech_to_speech()` works both directions
- [ ] metrics.csv ≥ 200 utterances, cloning + single-voice baseline
- [ ] SECS gap (cloning vs baseline) reported with mean ± std
- [ ] UTMOS + roundtrip WER + BLEU/chrF per direction
- [ ] MOS study: ≥5 listeners, 3 axes, inter-rater agreement
- [ ] Prosody correlation table (f0, energy, rate)
- [ ] (Optional) ≥1 fine-tuning experiment with before/after delta
- [ ] Repo reproduces every reported number
- [ ] Report limitations stated honestly (EN→ES asymmetry, PESQ/STOI dropped)
