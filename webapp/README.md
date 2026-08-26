# Bilingual Voice Dubbing — web app

Upload a video, choose a direction, get it back dubbed in the original speaker's voice with the
audio remuxed into the original picture. This implements **Objective 3** of the project
("integrate the synthesized dialogue with the original video, producing complete translated video
outputs"), which the research CLI did not cover.

## Install and run it locally

From the project root, install the declared web dependencies once:

```bash
source .venv/bin/activate
python -m pip install -e '.[ml,dev,web]'
cd webapp/frontend && npm install && cd ../..
```

Review the [Coqui model license](https://coqui.ai/cpml). If you accept it, enable XTTS
for this terminal before starting the app:

```bash
export COQUI_TOS_AGREED=1
```

The one-command launcher starts both services:

```bash
./webapp/start_local.sh
```

Open **http://localhost:3000**. Stop services started by the launcher with:

```bash
./webapp/stop_local.sh
```

For debugging, the equivalent manual setup uses two terminals. **Backend:**

Two terminals. **Backend:**

```bash
cd ~/Documents/research
COQUI_TOS_AGREED=1 .venv/bin/python -m uvicorn webapp.backend.main:app --port 8000
```

**Frontend:**

```bash
cd ~/Documents/research/webapp/frontend
npm install    # first time only
npm run dev
```

The frontend proxies `/api/*` to port 8000. The backend selects the completed DRAL-fine-tuned
MarianMT checkpoints by default; set `BVT_MT_MODE=pretrained` before launch to run the original
translation baseline instead.

## What happens to an upload

| Stage | Tool | Detail |
|---|---|---|
| 1 | `ffmpeg` | extract audio → 16 kHz mono WAV |
| 2 | Whisper `small` | transcribe **with segment timestamps** |
| 3 | MarianMT | translate each segment |
| 4 | XTTS-v2 | synthesise each segment, conditioned on the full source audio for voice |
| 5 | WORLD (`pyworld`) | **time-scale each segment to fit its original slot** |
| 6 | `ffmpeg` | remux: original video stream copied, audio replaced |

### Why stage 5 exists — this is the prosody preservation

Raw XTTS output does not match the timing of the speech it replaces. Measured on the DRAL test
split, generated speech runs about **1.5× longer** than the human reference; on continuous video,
Whisper segments include pauses that XTTS does not reproduce, so segments come out **too short**
instead. Either way the dub drifts out of sync within a few utterances.

Stage 5 fixes it by stretching or compressing each synthesised segment to exactly the duration of
the segment it replaces, using WORLD analysis/resynthesis so formants (and therefore speaker
identity) survive the operation. On the test clip:

```
raw TTS duration ratio   0.785     ← before retiming
final duration match     1.001     ← after retiming
speaker similarity       0.4220    ← measured on the retimed result
```

Scaling is clamped to **0.5×–2.0×**; beyond that the audio degrades audibly. Clamped segments are
counted and shown in the UI rather than hidden.

## Speed

Roughly **13× the clip length** on an M1: a 15-second clip takes ~3 minutes. Speech synthesis is
inherently sequential — each audio chunk depends on the previous one — and XTTS runs on CPU here
because that is the stable configuration on Apple Silicon. One job runs at a time (a global lock);
running two would make both slower.

Models load once on the first request (~30 s) and are reused for every later job.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | readiness, ffmpeg presence, allowed directions |
| `POST` | `/api/jobs` | multipart: `file`, `direction` (`es-en` \| `en-es`) → `202 {job_id}` |
| `GET` | `/api/jobs/{id}` | status, progress 0–100, message, result when done |
| `GET` | `/api/jobs/{id}/video` | dubbed MP4 (or WAV for audio-only uploads) |
| `GET` | `/api/jobs/{id}/audio` | dubbed audio track alone |

Accepts `.mp4 .mov .mkv .webm .m4v .avi .wav .mp3 .m4a .flac`, up to 200 MB. Audio-only uploads
work — the dubbed WAV becomes the deliverable since there is no picture to remux into.

Use a short clip with one clearly audible speaker. The current app creates a speech-only dubbed
track; it does not separate and preserve background music or handle multiple speakers separately.

## Outputs

Each job writes `webapp/jobs/<job_id>/`:

```
input.mp4                     the upload
source.wav                    extracted 16 kHz mono audio
segments/seg_NNNN.wav         raw XTTS output per segment
segments/seg_NNNN_retimed.wav after time-scaling
dubbed.wav                    assembled full-length track
dubbed.mp4                    final video
result.json                   per-segment metrics and totals
```

Keeping the raw *and* retimed segment audio makes the effect of stage 5 directly audible — useful
for the report and for a demo.

## Failure handling

A segment that fails (empty translation, synthesis error) records the error and is skipped; the rest
of the job continues and the failure is counted in `failed_segments` and shown in the UI. One bad
utterance never loses the whole job.

## Files

```
webapp/backend/dubbing.py   pipeline: extract, segment, translate, synthesise, retime, remux
webapp/backend/main.py      FastAPI service, background worker, single-job lock
webapp/frontend/app/page.tsx    bento-grid UI (upload, direction, progress, result, metrics, segments)
webapp/frontend/app/globals.css design tokens and grid
```

Reuses `LocalModels` from `src/bilingual_voice/pipeline.py` — the same Whisper, MarianMT, XTTS and
ECAPA instances the research CLI uses, so the app and the evaluation cannot drift apart.
