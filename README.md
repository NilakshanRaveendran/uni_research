# Bilingual Voice Translation (English ↔ Spanish)

Local Apple-Silicon research pipeline using DRAL, Whisper, MarianMT, XTTS-v2,
ECAPA speaker embeddings, corpus BLEU/chrF, WER, and prosody measurements.

The earlier `s2st_pipeline.py` and notebook are retained as historical Colab
prototypes. New experiments should use the `bvt` commands documented here.

## Research design

- **Dataset:** DRAL 16 kHz, paired English/Spanish re-enactments from bilingual speakers.
- **Split:** speaker-disjoint train/dev/test partition (70/15/15).
- **ASR:** Whisper `small`, evaluated against human source transcripts when available.
- **MT:** MarianMT in each direction, evaluated with corpus BLEU and chrF.
- **TTS baseline:** pretrained XTTS-v2 zero-shot voice cloning on CPU.
- **Identity:** ECAPA cosine similarity between source and generated speech, calibrated
  against the source-to-human-target similarity for the paired DRAL recording.
- **Intelligibility:** Whisper transcription of generated speech versus text sent to XTTS.
- **Prosody:** F0, energy, duration, and speaking-rate measurements, including the real
  paired target-language DRAL recording as a human reference.

Failures remain in the master CSV and are included in the reported failure rate.

## 1. System setup (Apple Silicon)

```bash
brew install python@3.11 ffmpeg sox git-lfs
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[ml,dev]'
```

`coqui-tts` does not bundle PyTorch in current releases. The `ml` extra installs
PyTorch, torchaudio, and the required torchcodec audio backend explicitly. XTTS starts on CPU because that is the stable
baseline on Apple Silicon. MarianMT can be moved to MPS with `--mt-device mps`
after the CPU smoke test succeeds.

Review the [Coqui model license](https://coqui.ai/cpml) before accepting it. For
non-interactive model loading:

```bash
export COQUI_TOS_AGREED=1
```

## 2. Dataset

Download the public-domain 16-kHz archive from the
[official DRAL page](https://www.cs.utep.edu/nigel/dral/) and place it at:

```text
data/raw/DRAL-16kHz.tgz
```

Then extract and inspect it:

```bash
tar -xzf data/raw/DRAL-16kHz.tgz -C data/raw
bvt inspect-dral data/raw
```

Do not pair files by replacing `EN_` with `ES_`: some valid pairs use different
conversation/channel IDs. The builder follows the official `trans_id` links.

## 3. Manifest and speaker-disjoint splits

```bash
bvt build-manifest data/raw data/processed/dral_manifest.csv
bvt split-manifest data/processed/dral_manifest.csv data/splits/dral_manifest.csv \
  --seed 498
```

The manifest columns are:

```text
pair_id,english_speaker_id,spanish_speaker_id,same_speaker_pair,
english_id,spanish_id,english_audio,spanish_audio,
english_text,spanish_text,split
```

The inspected 16-kHz release contains 2,834 same-speaker pairs and 59 pairs whose
linked English and Spanish fragments have different unique-speaker IDs. Both IDs
are retained. Split assignment groups connected speaker IDs before partitioning,
preventing either side of those 59 pairs from leaking across splits. Primary
target-reference prosody correlations use the same-speaker subset.

The official audio release may not contain human transcripts. Empty transcript
fields are intentional: ASR WER and translation BLEU must be reported as unavailable
until real references are merged. Never use Whisper output as its own ASR reference.

To merge either UTF-8 CSV (`audio_id,text`) or the official DRAL challenge
`filename|annotator|text` transcript files:

```bash
bvt merge-transcripts data/splits/dral_manifest.csv \
  data/splits/dral_manifest_with_text.csv \
  --transcripts data/raw/DRAL16kHz/metadata/EN_transcripts.txt \
                data/raw/DRAL16kHz/metadata/ES_transcripts.txt
```

## 4. Smoke test and full evaluation

Start with 10 pairs in both directions:

```bash
bvt run data/splits/dral_manifest_with_text.csv \
  --split test --limit 10 --directions en-es es-en \
  --output results/smoke_metrics.csv

bvt summarize results/smoke_metrics.csv results/smoke_summary.json
bvt plot-results results/smoke_metrics.csv plots/smoke
```

Run the full held-out split only after reviewing generated audio and smoke-test failures:

```bash
bvt run data/splits/dral_manifest_with_text.csv \
  --split test --directions en-es es-en \
  --output results/final_metrics.csv --resume
bvt summarize results/final_metrics.csv results/final_summary.json
bvt plot-results results/final_metrics.csv plots/final
```

Models download on first use. Each completed row is checkpointed immediately.
`--resume` skips completed sample/direction keys but retries failures.

## 5. Verification

```bash
python -m pytest
python -m ruff check src tests webapp/backend
python -m compileall -q src
```

## Fine-tuned MarianMT experiment

MarianMT was fine-tuned independently in both directions using only the DRAL training split.
Checkpoint selection used development loss, and the held-out test split was used once for the
final comparison. The resulting gains were +2.336 BLEU for EN→ES and +3.228 BLEU for ES→EN.
These gains measure adaptation to DRAL's conversational re-enactment style, not necessarily a
general improvement in translation adequacy. Reproduce the experiment with
`python -m bilingual_voice.finetune`.

## Write-up: dissertation and manuscript

Two LaTeX documents report the committed result files. `thesis/scripts/thesis_numbers.py` writes
`thesis/numbers.json` from the CSVs and JSON in `results/` and `artifacts/` (it regenerates
byte-identically), and `thesis/scripts/thesis_figures.py` draws every figure from it. The numbers in
the `.tex` sources were transcribed from those files by hand, not inserted by a script, so a few
table cells carry double rounding; `docs/viva/DEFECT_REGISTER.md` lists every known discrepancy. The
reported pipeline results come from `results/synthesis_final.csv` (re-measured with
`bilingual_voice.remeasure`), not from the `results/final_metrics.csv` the commands in section 4
produce.

| Path | What it is |
|---|---|
| `thesis/` | UWU final-year dissertation. `uwuthesis.sty` implements SECTION-H of the dissertation guidelines (A4, 3.85 cm left margin, Times New Roman 12, 1.5 spacing, page numbers bottom-centre, APA references via `apacite`). |
| `journal/` | IEEE manuscript, unmodified `IEEEtran` class from the official IEEE template, `IEEEtran.bst` references. Journal format by default; change one class option for the conference format. |

```bash
python thesis/scripts/thesis_numbers.py     # -> thesis/numbers.json
python thesis/scripts/thesis_figures.py     # -> thesis/figures/
./thesis/build.sh                           # -> thesis/main.pdf
./journal/build.sh                          # -> journal/main.pdf
```

Both builds need a TeX distribution with `apacite`, `IEEEtran` and the packages listed at the top
of `thesis/uwuthesis.sty`.

## Prosody measurement: two frequency bands, not one

`prosody.py` and `analysis.py` share one definition of each band, and keeping them apart matters:

- **Search band, 65-1000 Hz** — passed to `librosa.pyin`. It must be wide, because pyin's
  voiced/unvoiced decision depends on the number of frequency candidates; narrowing it suppresses
  voicing detection entirely on quiet recordings.
- **Plausibility band, 60-400 Hz** — applied downstream in `metrics.py` and `analysis.py` as a
  data-quality gate. A mean F0 outside the adult speaking range is a tracking failure, not a voice.

F0 level is correlated in **semitones**, never in hertz. Correlating in hertz without the gate
understates the generated-versus-human correspondence several-fold; this is documented in Section
4.10 of the dissertation. One exception is known: the system's F0-*range* correlation in Table 4.4
(0.035 / 0.040) was computed from the standard deviation in hertz, ungated, while the human column
uses the semitone standard deviation. `thesis/scripts/remeasure_f0_range.py` measures the system
side the same way as the human side (0.088 / 0.112 within-speaker).

## Local website

The local web application accepts a short video or audio file, runs Whisper → MarianMT → XTTS-v2,
retimes generated segments, and returns dubbed media with segment details and measurements.

```bash
./webapp/start_local.sh
```

Open [http://localhost:3000](http://localhost:3000). Full setup and operating notes are in
[`webapp/README.md`](webapp/README.md).
