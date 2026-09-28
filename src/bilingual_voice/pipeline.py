"""Lazy-loaded local inference pipeline with row-level auditability."""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

from .metrics import normalized_wer
from .prosody import extract_prosody

RESULT_FIELDS = [
    "sample_id",
    "speaker_id",
    "same_speaker_pair",
    "direction",
    "source_audio",
    "target_reference_audio",
    "source_reference_text",
    "asr_text",
    "translation_reference",
    "translated_text",
    "tts_asr_text",
    "generated_audio",
    "tts_seed",
    "asr_wer",
    "tts_intelligibility_wer",
    "speaker_similarity",
    "human_target_speaker_similarity",
    "f0_mean_source",
    "f0_mean_generated",
    "f0_mean_target_reference",
    "f0_std_source",
    "f0_std_generated",
    "f0_std_target_reference",
    "energy_source",
    "energy_generated",
    "energy_target_reference",
    "duration_source",
    "duration_generated",
    "duration_target_reference",
    "speaking_rate_source",
    "speaking_rate_generated",
    "speaking_rate_target_reference",
    "status",
    "error_stage",
    "error_message",
]

# Result column stem -> extract_prosody key. Every prosody column follows the
# "{column}_{role}" convention that metrics.py and plots.py read back.
PROSODY_COLUMNS = (
    ("f0_mean", "f0_mean"),
    ("f0_std", "f0_std"),
    ("energy", "energy_mean"),
    ("duration", "duration"),
    ("speaking_rate", "speaking_rate"),
)


def sample_seed(pair_id: str, direction: str) -> int:
    """Stable per-sample TTS seed. Uses sha256, not hash(), which is salted per process."""
    digest = hashlib.sha256(f"{pair_id}|{direction}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _resolve_tts_device(requested: str) -> str:
    """"auto" prefers the Mac GPU when the machine has one."""
    import torch

    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _memoize_voice_cloning(tts, keep: int = 16) -> None:
    """Cache the cloned XTTS voice so the same reference audio is only analysed once.

    A dubbed video has a handful of speakers but many segments per speaker, and `tts_to_file`
    re-clones the voice from the reference file on every call: on a clip with 39 segments that is
    39 passes over the same few references, and it is pure waste. `keep` bounds the cache at more
    voices than a video realistically has, so alternating speakers never evict each other.

    `clone_voice` is the right seam rather than `get_conditioning_latents`: it is what
    `XTTS.synthesize` actually calls, and it wraps the reference-audio loading and resampling as
    well as the latent computation. Wrapping it (instead of reimplementing the inference path)
    means every sampling setting -- temperature, penalties, top-k/p, text splitting -- keeps its
    exact behaviour, because `synthesize` still derives them from the model config as before.

    The cache key includes each reference file's size and modification time, so editing the audio
    still recomputes.
    """
    model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
    if model is None:
        return
    # Coqui moved voice cloning to VoiceMixin.clone_voice; fall back for older versions.
    name = "clone_voice" if hasattr(model, "clone_voice") else "get_conditioning_latents"
    original = getattr(model, name, None)
    if original is None or getattr(original, "_memoized", False):
        return

    cache: dict[tuple, object] = {}

    def _fingerprint(value):
        items = value if isinstance(value, (list, tuple)) else [value]
        marks = []
        for item in items:
            try:
                stat = Path(item).stat()
                marks.append((str(item), stat.st_size, stat.st_mtime_ns))
            except (OSError, TypeError, ValueError):
                marks.append((repr(item), None, None))
        return tuple(marks)

    def cached(*args, **kwargs):
        try:
            reference = args[0] if args else kwargs.get("speaker_wav", kwargs.get("audio_path"))
            key = (
                _fingerprint(reference),
                repr(args[1:]),
                repr(sorted(kwargs.items(), key=lambda kv: kv[0])),
            )
        except Exception:  # noqa: BLE001 - an uncacheable call must still work
            return original(*args, **kwargs)
        if key not in cache:
            if len(cache) >= keep:
                cache.pop(next(iter(cache)))
            cache[key] = original(*args, **kwargs)
        return cache[key]

    cached._memoized = True
    cached._cache = cache
    setattr(model, name, cached)


def _clear_voice_cache(tts) -> None:
    """Forget cached cloned voices, e.g. after moving the model to another device."""
    model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
    for name in ("clone_voice", "get_conditioning_latents"):
        cache = getattr(getattr(model, name, None), "_cache", None)
        if cache is not None:
            cache.clear()


class LocalModels:
    def __init__(
        self,
        whisper_size: str = "small",
        mt_device: str = "cpu",
        model_root: Path = Path("models"),
        mt_model_dirs: dict[str, Path] | None = None,
        tts_device: str | None = None,
    ) -> None:
        self.whisper_size = whisper_size
        self.mt_device = mt_device
        # XTTS device. "auto" prefers the Mac GPU, which measured ~24% faster on this machine
        # (1.37 -> 1.04 seconds of compute per second of audio) at comparable speaker similarity
        # (ECAPA 0.701 vs 0.689 -- within the spread of a stochastic sampler). The gain is modest
        # rather than dramatic because generation is autoregressive: each audio token depends on
        # the last, which is close to a GPU's worst case. A probe below falls back to CPU when the
        # device cannot actually run the model, following the pattern finetune.py already uses.
        # The research pipeline defaults to CPU, the configuration every reported number was
        # produced with (README section 1); the web app asks for "auto" explicitly.
        self.tts_device = (tts_device or os.environ.get("BVT_TTS_DEVICE", "cpu")).strip().lower()
        self._active_tts_device = "cpu"
        self.model_root = model_root.resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.mt_model_dirs = {
            direction: path.expanduser().resolve()
            for direction, path in (mt_model_dirs or {}).items()
        }
        self._asr = None
        self._mt: dict[str, tuple[object, object]] = {}
        self._tts = None
        self._speaker = None

    @property
    def asr(self):
        if self._asr is None:
            import whisper

            self._asr = whisper.load_model(
                self.whisper_size,
                device="cpu",
                download_root=str(self.model_root / "cache" / "whisper"),
            )
        return self._asr

    def transcribe(self, audio: Path, language: str) -> str:
        result = self.asr.transcribe(
            str(audio), language=language, task="transcribe", verbose=False
        )
        return result["text"].strip()

    def translate(self, text: str, source: str, target: str) -> str:
        return self.translate_many([text], source, target)[0]

    def mt_model(self, source: str, target: str):
        """The (tokenizer, model) MarianMT pair for a direction, loaded once and cached."""
        from transformers import MarianMTModel, MarianTokenizer

        key = f"{source}-{target}"
        if key not in self._mt:
            pretrained_name = {
                "en-es": "Helsinki-NLP/opus-mt-en-es",
                "es-en": "Helsinki-NLP/opus-mt-es-en",
            }[key]
            model_name = str(self.mt_model_dirs.get(key, pretrained_name))
            cache_dir = self.model_root / "huggingface"
            load_kwargs = {} if key in self.mt_model_dirs else {"cache_dir": cache_dir}
            tokenizer = MarianTokenizer.from_pretrained(model_name, **load_kwargs)
            model = MarianMTModel.from_pretrained(model_name, **load_kwargs).to(self.mt_device)
            model.eval()
            self._mt[key] = (tokenizer, model)
        return self._mt[key]

    def translate_many(self, texts: list[str], source: str, target: str) -> list[str]:
        """Translate a batch in one forward pass.

        Dubbing translates every segment of a clip with the same model; doing them one at a time
        pays the per-call overhead once per segment for no reason. Empty strings are passed
        through untouched rather than sent to the model, which would produce spurious output.
        """
        tokenizer, model = self.mt_model(source, target)

        wanted = [i for i, text in enumerate(texts) if text and text.strip()]
        output = [""] * len(texts)
        if not wanted:
            return output

        encoded = tokenizer(
            [texts[i] for i in wanted],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        encoded = {name: tensor.to(self.mt_device) for name, tensor in encoded.items()}
        generated = model.generate(**encoded, max_new_tokens=512)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for position, index in enumerate(wanted):
            output[index] = decoded[position].strip()
        return output

    @property
    def tts(self):
        if self._tts is None:
            if os.environ.get("COQUI_TOS_AGREED") != "1":
                raise RuntimeError("Review the Coqui model license, then set COQUI_TOS_AGREED=1")

            os.environ.setdefault("TTS_HOME", str(self.model_root / "coqui"))
            from TTS.api import TTS

            device = _resolve_tts_device(self.tts_device)
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self._active_tts_device = device
            _memoize_voice_cloning(self._tts)
        return self._tts

    def synthesize(
        self,
        text: str,
        speaker_wav: Path,
        language: str,
        output: Path,
        seed: int,
        speed: float = 1.0,
        split_sentences: bool = True,
    ) -> None:
        if not text.strip():
            raise ValueError("XTTS input text is empty")
        import torch

        output.parent.mkdir(parents=True, exist_ok=True)
        # Resolve the lazy model before seeding so loading cannot consume the RNG stream.
        tts = self.tts

        # XTTS's own speed control stretches its latent sequence before the vocoder, so pitch and
        # voice are kept. Passed only when asked for, so default runs are byte-identical to the
        # evaluated pipeline.
        extra = {} if speed == 1.0 else {"speed": float(speed)}

        def _run() -> None:
            torch.manual_seed(seed)
            tts.tts_to_file(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language,
                file_path=str(output),
                split_sentences=split_sentences,
                **extra,
            )

        try:
            _run()
        except Exception as exc:
            # Not every PyTorch op has an MPS kernel, and a missing one raises partway through
            # generation. Retry once on CPU and stay there, rather than losing the job -- a
            # load-time probe cannot catch this because the failing op may be input-dependent.
            if self._active_tts_device == "cpu":
                raise
            print(
                f"  XTTS failed on {self._active_tts_device} "
                f"({type(exc).__name__}: {str(exc)[:120]}); falling back to cpu",
                flush=True,
            )
            self._tts = tts.to("cpu")
            self._active_tts_device = "cpu"
            # Cached voices hold tensors on the old device; reusing them on CPU fails every call.
            _clear_voice_cache(self._tts)
            _memoize_voice_cloning(self._tts)
            tts = self._tts
            _run()

    @property
    def speaker_encoder(self):
        if self._speaker is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.inference.classifiers import EncoderClassifier
            self._speaker = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_root / "spkrec-ecapa-voxceleb"),
                run_opts={"device": "cpu"},
            )
        return self._speaker

    def speaker_similarity(self, first: Path, second: Path) -> float:
        import torch
        import torchaudio
        from torch.nn import functional

        def embedding(path: Path):
            signal, sample_rate = torchaudio.load(path)
            signal = signal.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                signal = torchaudio.functional.resample(signal, sample_rate, 16000)
            return self.speaker_encoder.encode_batch(signal).squeeze()

        with torch.inference_mode():
            return float(functional.cosine_similarity(embedding(first), embedding(second), dim=0))


def _base_result(row: dict[str, str], direction: str) -> dict[str, str]:
    source_language, target_language = direction.split("-")
    source_name = "english" if source_language == "en" else "spanish"
    target_name = "spanish" if target_language == "es" else "english"
    return {
        "sample_id": row["pair_id"],
        "speaker_id": row[f"{source_name}_speaker_id"],
        "same_speaker_pair": row["same_speaker_pair"],
        "direction": direction,
        "source_audio": row[f"{source_name}_audio"],
        "target_reference_audio": row[f"{target_name}_audio"],
        "source_reference_text": row.get(f"{source_name}_text", ""),
        "translation_reference": row.get(f"{target_name}_text", ""),
        **{
            field: ""
            for field in RESULT_FIELDS
            if field
            not in {
                "sample_id",
                "speaker_id",
                "same_speaker_pair",
                "direction",
                "source_audio",
                "target_reference_audio",
                "source_reference_text",
                "translation_reference",
            }
        },
    }


def evaluate_one(
    row: dict[str, str], direction: str, models: LocalModels, output_root: Path
) -> dict[str, str]:
    result = _base_result(row, direction)
    source_language, target_language = direction.split("-")
    source_audio = Path(result["source_audio"])
    target_audio = Path(result["target_reference_audio"])
    generated = output_root / direction / f"{row['pair_id']}_{direction}.wav"
    stage = "asr"
    try:
        result["asr_text"] = models.transcribe(source_audio, source_language)
        asr_wer = normalized_wer(result["source_reference_text"], result["asr_text"])
        result["asr_wer"] = "" if asr_wer is None else str(asr_wer)

        stage = "translation"
        result["translated_text"] = models.translate(
            result["asr_text"], source_language, target_language
        )

        stage = "tts"
        seed = sample_seed(row["pair_id"], direction)
        result["tts_seed"] = str(seed)
        models.synthesize(result["translated_text"], source_audio, target_language, generated, seed)
        result["generated_audio"] = str(generated.resolve())

        stage = "tts_asr"
        tts_asr = models.transcribe(generated, target_language)
        result["tts_asr_text"] = tts_asr
        intelligibility = normalized_wer(result["translated_text"], tts_asr)
        result["tts_intelligibility_wer"] = "" if intelligibility is None else str(intelligibility)

        stage = "speaker"
        result["speaker_similarity"] = str(models.speaker_similarity(source_audio, generated))
        result["human_target_speaker_similarity"] = str(
            models.speaker_similarity(source_audio, target_audio)
        )

        stage = "prosody"
        measured = {
            "source": extract_prosody(source_audio, result["source_reference_text"]),
            "generated": extract_prosody(generated, result["translated_text"]),
            "target_reference": extract_prosody(target_audio, result["translation_reference"]),
        }
        for column, feature in PROSODY_COLUMNS:
            for role, features in measured.items():
                result[f"{column}_{role}"] = str(features[feature])
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - failures must become auditable CSV rows
        result["status"] = "failed"
        result["error_stage"] = stage
        result["error_message"] = f"{type(exc).__name__}: {exc}"[:1000]
    return result


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_results(rows: list[dict[str, str]], path: Path) -> None:
    """Replace the results CSV atomically so a crash mid-write cannot truncate it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_evaluation(
    rows: Iterable[dict[str, str]],
    directions: list[str],
    output_csv: Path,
    output_audio: Path,
    models: LocalModels,
    resume: bool = False,
) -> list[dict[str, str]]:
    existing = read_results(output_csv) if resume else []
    completed = {
        (row["sample_id"], row["direction"]) for row in existing if row.get("status") == "ok"
    }
    all_rows = existing
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for direction in directions:
            key = (row["pair_id"], direction)
            if key in completed:
                continue
            result = evaluate_one(row, direction, models, output_audio)
            all_rows = [
                old for old in all_rows if (old.get("sample_id"), old.get("direction")) != key
            ]
            all_rows.append(result)
            _write_results(all_rows, output_csv)
    return all_rows
