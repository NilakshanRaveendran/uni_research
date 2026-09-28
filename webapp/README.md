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

Open **http://localhost:3000**. The page hot-reloads but the backend does not, so after pulling
changes run the launcher again: it restarts the backend if its Python code has changed since it
started. Stop services started by the launcher with:

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

The frontend proxies `/api/*` to port 8000. By default (`BVT_MT_MODE=mixed`) English→Spanish uses
the pretrained MarianMT model and Spanish→English the DRAL fine-tune: on the app's own jobs the
fine-tune copies English words into Spanish output ("¿Accent? No tengo acento.") while for
Spanish→English it drops fewer sentences. `BVT_MT_MODE=finetuned` or `pretrained` uses one kind in
both directions. Lines are translated sentence by sentence, which stopped MarianMT dropping whole
sentences and question marks.

## What happens to an upload

| Stage | Tool | Detail |
|---|---|---|
| 1 | `ffmpeg` | extract audio → 16 kHz mono WAV |
| 2 | Whisper `small` | transcribe **with segment timestamps** |
| 3 | MarianMT | translate each segment, one sentence at a time; a line that cannot fit gets a shorter wording |
| 4 | ECAPA + XTTS-v2 | group segments by speaker, then synthesise each segment from a reference cut from **that speaker's own speech** |
| 5 | XTTS speed + WSOLA | **fit each line into the time it has, with a pause before the next line** |
| 6 | `ffmpeg` | remux: original video stream copied, audio replaced |

### Why stage 5 exists — fitting the dub to the video

Raw XTTS output does not match the timing of the speech it replaces. It speaks far slower than
people in real video (1.5–1.9 words/s against 3.1–3.8 in a measured YouTube Short), pads every
sentence with 0.417 s of silence and pauses up to ~4 s between sentences. Across 26 jobs, Spanish
lines came out a median 1.85× their slot and 42 % over 2×. Each line is therefore fitted in order:

1. **Cut the dead air**: edge silence and XTTS's padding go, inner pauses are capped at 0.25 s
   (35 dB threshold; ~15 % of the audio, no words lost in a Whisper check).
2. **Clone from a clean reference**: each speaker's reference is lightly denoised (spectral gate,
   at most 12 dB) before XTTS clones it. XTTS copies the reference's noise into every line and its
   runaway takes cluster on noisy references: 9 of 48 takes ran away before, 0 after, and word
   errors fell from 0.107 to 0.025.
3. **Re-sample runaway takes**: a take slower than 60 % of the language's normal rate is
   synthesised again with another seed (up to twice) and the fastest take kept.
4. **Shorter wording for lines that cannot fit**: when a line needs more compression than is
   free (XTTS 1.25× × WSOLA 0.85), each sentence is re-chosen from MarianMT's beam-8 candidates:
   close to the best score, keeping every question, negation and number, and checked by
   translating back. About 10 % fewer characters on such lines, no meaning errors in 35 judged.
5. **XTTS speed up to 1.25×**, which costs nothing measurable; at 1.5× XTTS starts dropping words.
6. **WSOLA for the rest**, never below 0.67× and never stretching. WSOLA splices the voice's real
   waveform; WORLD, used before, resynthesised it through a vocoder and damaged it even when it
   changed nothing (octave errors on 13.5 % of frames, a buzzy tone) — the "synthetic" sound.
7. **A real pause between lines**: 0.2 s before a different speaker, 0.1 s before the same one
   (the lower quartile of natural turn gaps in DRAL), and a line runs at most 0.3 s past the speech
   it replaces. Anything still left over is faded out; it is never summed with the next voice.

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
| `POST` | `/api/jobs/url` | form: `url`, `direction` → the worker downloads the video with yt-dlp (public http(s) hosts only, 10 min, 200 MB) |
| `POST` | `/api/license/accept` | records that the Coqui XTTS licence was accepted in the page |

State-changing requests must carry the header `X-BVT-Client: web` (the page sends it). This stops
another website open in the same browser from driving the local API with a plain form post.

Accepts `.mp4 .mov .mkv .webm .m4v .avi .wav .mp3 .m4a .flac`, up to 200 MB. Audio-only uploads
work — the dubbed WAV becomes the deliverable since there is no picture to remux into.

Each speaker is cloned separately. Segments are grouped by voice (ECAPA embeddings,
average-linkage clustering calibrated on DRAL), and every speaker gets a reference of up to 25 s of
their own speech, cut from the separated vocal stem when separation runs and from the original mix
otherwise. A speaker with under ~4 s of speech in the whole clip shares the closest voice, because
that is too little audio for a distinguishable clone. Two people who sound very alike may be merged,
one actor recorded very differently in two scenes may be split, and someone who only ever speaks in
very short lines is sometimes voiced as the person they answer. To fix the number of voices, stop the
backend and start it with the count set (the launcher does not restart a running backend for a
changed setting):

```bash
./webapp/stop_local.sh && BVT_SPEAKERS=2 ./webapp/start_local.sh
```

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
