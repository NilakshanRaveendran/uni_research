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

## Local website

The local web application accepts a short video or audio file, runs Whisper → MarianMT → XTTS-v2,
retimes generated segments, and returns dubbed media with segment details and measurements.

```bash
./webapp/start_local.sh
```

Open [http://localhost:3000](http://localhost:3000). Full setup and operating notes are in
[`webapp/README.md`](webapp/README.md).
