"""
============================================================================
 Bilingual Voice Transcription & Synthesis Preserving Speaker Identity
 Spanish <-> English  |  Zero-shot voice-cloning cascade  |  Colab / Kaggle
----------------------------------------------------------------------------
 Pipeline:  speech --Whisper--> text --NLLB--> text --XTTS-v2--> speech
            (translated speech is spoken in the ORIGINAL speaker's voice)
 Evaluation: SECS (speaker similarity) + UTMOS (naturalness) + WER + BLEU
----------------------------------------------------------------------------
 This file is written as Colab "cells" (# %% markers). You can:
   - Open the companion .ipynb directly in Google Colab, OR
   - Paste each # %% block into a Colab cell.
 Team: Nilakshan (TTS/speaker) · Janukshan (ASR) · Piranya (MT/data)
============================================================================
"""

# %% [1] INSTALL  (run once per Colab session; ~3-5 min)
# ---------------------------------------------------------------------------
# !pip -q install -U openai-whisper transformers sentencepiece sacremoses
# !pip -q install -U coqui-tts                       # XTTS-v2 (import name: TTS)
# !pip -q install -U resemblyzer jiwer sacrebleu soundfile librosa
# !apt -qq install -y ffmpeg
# NOTE: coqui-tts >=0.27 does NOT bundle torch; Colab already has torch. If you
# hit a torch/transformers version clash, pin:  !pip install "transformers>=4.40"


# %% [2] IMPORTS & DEVICE
# ---------------------------------------------------------------------------
import os, glob, csv, math
import numpy as np
import torch
import soundfile as sf

os.environ["COQUI_TOS_AGREED"] = "1"   # accept XTTS non-commercial terms non-interactively
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# %% [3] CONFIG
# ---------------------------------------------------------------------------
WHISPER_SIZE = "large-v3"      # "medium" is ~2x faster if VRAM is tight
NLLB_MODEL   = "facebook/nllb-200-distilled-600M"
XTTS_MODEL   = "tts_models/multilingual/multi-dataset/xtts_v2"

# NLLB language codes
LANG2NLLB = {"es": "spa_Latn", "en": "eng_Latn"}
# XTTS language codes
LANG2XTTS = {"es": "es",       "en": "en"}

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# %% [4] LOAD MODELS (once)
# ---------------------------------------------------------------------------
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS

print("Loading Whisper...")
asr_model = whisper.load_model(WHISPER_SIZE, device=DEVICE)

print("Loading NLLB...")
mt_tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(DEVICE)

print("Loading XTTS-v2...")
tts = TTS(XTTS_MODEL).to(DEVICE)
print("All models loaded.")


# %% [5] STAGE FUNCTIONS
# ---------------------------------------------------------------------------
def transcribe(wav_path, force_lang=None):
    """Whisper: audio -> (text, detected_lang). Returns segments for later sync."""
    result = asr_model.transcribe(wav_path, language=force_lang, verbose=False)
    return result["text"].strip(), result["language"], result.get("segments", [])


def translate(text, src, tgt):
    """NLLB: text -> translated text. src/tgt in {'es','en'}."""
    mt_tok.src_lang = LANG2NLLB[src]
    enc = mt_tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    bos = mt_tok.convert_tokens_to_ids(LANG2NLLB[tgt])
    gen = mt_model.generate(**enc, forced_bos_token_id=bos, max_length=512)
    return mt_tok.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _split_sentences(text, max_chars=220):
    """XTTS handles ~250 chars best; chunk long text on sentence boundaries."""
    import re
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(cur) + len(s) < max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts or [text]


def synthesize(text, tgt_lang, speaker_wav, out_path):
    """XTTS-v2: translated text -> speech IN THE speaker_wav's VOICE."""
    chunks = _split_sentences(text)
    audio = []
    for ch in chunks:
        wav = tts.tts(text=ch, speaker_wav=speaker_wav, language=LANG2XTTS[tgt_lang])
        audio.append(np.asarray(wav, dtype=np.float32))
    audio = np.concatenate(audio) if len(audio) > 1 else audio[0]
    sr = tts.synthesizer.output_sample_rate
    sf.write(out_path, audio, sr)
    return out_path, sr


# %% [6] END-TO-END SPEECH-TO-SPEECH (bidirectional, auto-direction)
# ---------------------------------------------------------------------------
def speech_to_speech(in_wav, out_wav=None, force_src=None):
    """
    Full pipeline. Language is auto-detected by Whisper, so ES->EN and EN->ES
    both work with the same call. The ORIGINAL clip is used as the voice reference.
    Returns a dict with texts + output path (handy for evaluation).
    """
    out_wav = out_wav or os.path.join(OUT_DIR, "s2s_" + os.path.basename(in_wav))
    src_text, src_lang, segs = transcribe(in_wav, force_lang=force_src)
    src_lang = "es" if src_lang.startswith("es") else "en"
    tgt_lang = "en" if src_lang == "es" else "es"
    tgt_text = translate(src_text, src_lang, tgt_lang)
    out_path, sr = synthesize(tgt_text, tgt_lang, speaker_wav=in_wav, out_path=out_wav)
    return {"in_wav": in_wav, "src_lang": src_lang, "tgt_lang": tgt_lang,
            "src_text": src_text, "tgt_text": tgt_text, "out_wav": out_path,
            "segments": segs}


# %% [7] EVALUATION  --  the metrics your report needs
# ---------------------------------------------------------------------------
from resemblyzer import VoiceEncoder, preprocess_wav
_spk_encoder = VoiceEncoder(device=DEVICE if DEVICE == "cpu" else "cuda")

def secs(ref_wav, gen_wav):
    """Speaker Encoder Cosine Similarity: identity preservation (higher = better)."""
    e1 = _spk_encoder.embed_utterance(preprocess_wav(ref_wav))
    e2 = _spk_encoder.embed_utterance(preprocess_wav(gen_wav))
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

_utmos = None
def utmos(wav_path):
    """Predicted naturalness MOS (no reference needed)."""
    global _utmos
    if _utmos is None:
        _utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong",
                                trust_repo=True).to(DEVICE)
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(_utmos(t, sr).item())

def wer(reference, hypothesis):
    import jiwer
    return float(jiwer.wer(reference, hypothesis))

def bleu(reference_text, hypothesis_text):
    import sacrebleu
    return float(sacrebleu.sentence_bleu(hypothesis_text, [reference_text]).score)

def roundtrip_wer(result):
    """WER of the whole chain: re-ASR the OUTPUT speech, compare to target text."""
    hyp, _, _ = transcribe(result["out_wav"], force_lang=result["tgt_lang"])
    return wer(result["tgt_text"].lower(), hyp.lower())


# %% [8] SINGLE-CLIP DEMO
# ---------------------------------------------------------------------------
# r = speech_to_speech("sample_es.wav")
# print("SRC (%s): %s" % (r["src_lang"], r["src_text"]))
# print("TGT (%s): %s" % (r["tgt_lang"], r["tgt_text"]))
# print("Output :", r["out_wav"])
# print("SECS   : %.3f  (speaker similarity, source vs output)" % secs(r["in_wav"], r["out_wav"]))
# print("UTMOS  : %.2f  (naturalness of output)" % utmos(r["out_wav"]))
# from IPython.display import Audio, display
# display(Audio(r["in_wav"]));  display(Audio(r["out_wav"]))


# %% [9] BATCH EVALUATION over a CVSS (ES->EN) subset  ->  metrics CSV
# ---------------------------------------------------------------------------
# Expected layout (CVSS / your own):
#   data/es/clip1.wav ...          (source Spanish audio)
#   data/refs.tsv  ->  "clip1.wav\t<reference English translation text>"
def batch_eval(audio_dir, refs_tsv=None, csv_out=os.path.join(OUT_DIR, "metrics.csv"),
               limit=None):
    refs = {}
    if refs_tsv and os.path.exists(refs_tsv):
        for line in open(refs_tsv, encoding="utf-8"):
            k, _, v = line.rstrip("\n").partition("\t")
            refs[k] = v
    wavs = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if limit:
        wavs = wavs[:limit]
    rows = []
    for i, w in enumerate(wavs, 1):
        try:
            r = speech_to_speech(w)
            row = {
                "file": os.path.basename(w),
                "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                "src_text": r["src_text"], "tgt_text": r["tgt_text"],
                "SECS": round(secs(r["in_wav"], r["out_wav"]), 4),
                "UTMOS": round(utmos(r["out_wav"]), 3),
                "roundtrip_WER": round(roundtrip_wer(r), 4),
            }
            ref = refs.get(os.path.basename(w))
            if ref:
                row["BLEU"] = round(bleu(ref, r["tgt_text"]), 2)
            rows.append(row)
            print("[%d/%d] %s  SECS=%.3f UTMOS=%.2f" %
                  (i, len(wavs), row["file"], row["SECS"], row["UTMOS"]))
        except Exception as e:
            print("  ! skipped", os.path.basename(w), "->", e)
        # checkpoint every 20 clips so a Colab timeout never loses progress
        if rows and i % 20 == 0:
            _write_csv(rows, csv_out)
    _write_csv(rows, csv_out)
    if rows:
        import statistics as st
        for m in ["SECS", "UTMOS", "roundtrip_WER", "BLEU"]:
            vals = [x[m] for x in rows if m in x]
            if vals:
                print("mean %-14s = %.3f (n=%d)" % (m, st.mean(vals), len(vals)))
    return rows

def _write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)

# batch_eval("data/es", refs_tsv="data/refs.tsv", limit=50)


# %% [10] BASELINE (cloning OFF) -- to prove the value of speaker preservation
# ---------------------------------------------------------------------------
# Synthesize the SAME translations with a fixed reference voice for ALL clips.
# The SECS gap between this baseline and Cell 9 is your key result.
def synth_singlevoice(text, tgt_lang, fixed_ref_wav, out_path):
    return synthesize(text, tgt_lang, speaker_wav=fixed_ref_wav, out_path=out_path)
# Compare mean SECS(cloning) vs mean SECS(single-voice) -> report the delta.
