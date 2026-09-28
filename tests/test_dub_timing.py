"""Fitting synthesised lines into the video's timeline, and translating them (dubbing.py)."""

import numpy as np
import pytest
import soundfile

from webapp.backend import dubbing, speakers

SR = 24000


def _tone(seconds: float, hz: float = 220.0, amp: float = 0.3):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * hz * t)


def _silence(seconds: float):
    return np.zeros(int(seconds * SR))


# --- trimming and placement ----------------------------------------------------------------------


def test_tighten_cuts_the_padding_and_long_pauses_but_keeps_speech() -> None:
    # XTTS output: speech, a 1.4 s pause between sentences, speech, then coqui's 0.417 s zero pad.
    line = np.concatenate([_tone(1.0), _silence(1.4), _tone(0.8), _silence(0.417)])
    out = dubbing.tighten(line, SR)
    speech = 1.0 + dubbing.MAX_PAUSE_S + 0.8
    assert speech <= len(out) / SR <= speech + dubbing.TAIL_KEEP_S + 0.1
    assert np.sum(out**2) == pytest.approx(np.sum(line**2), rel=0.02)  # no speech lost


def test_tighten_keeps_short_pauses_and_a_little_air() -> None:
    line = np.concatenate([_silence(0.3), _tone(1.0), _silence(0.15), _tone(1.0), _silence(0.5)])
    out = dubbing.tighten(line, SR)
    body = 1.0 + 0.15 + 1.0
    assert body + 0.1 <= len(out) / SR <= body + dubbing.LEAD_KEEP_S + dubbing.TAIL_KEEP_S + 0.1


def test_tighten_leaves_silence_alone() -> None:
    assert len(dubbing.tighten(_silence(1.0), SR)) == int(1.0 * SR)


def test_place_never_writes_past_the_next_line_and_fades_what_it_drops() -> None:
    timeline = np.zeros(10 * SR)
    dropped = dubbing.place(timeline, _tone(3.0), offset=1 * SR, limit=3 * SR, sr=SR)
    assert dropped == pytest.approx(1.0, abs=1e-3)
    assert not np.any(timeline[3 * SR :])  # nothing summed into the next line's time
    assert abs(timeline[3 * SR - 1]) < 1e-3  # faded to zero at the cut, no click


def test_place_keeps_a_line_that_fits_untouched() -> None:
    timeline = np.zeros(5 * SR)
    line = _tone(1.0)
    assert dubbing.place(timeline, line, offset=SR, limit=3 * SR, sr=SR) == 0
    assert np.allclose(timeline[SR : 2 * SR], line)


def test_xtts_speed_is_only_used_when_needed_and_never_past_its_free_range() -> None:
    text = "x" * 60  # 60 characters: 5 s of Spanish speech at the measured 12 chars/s
    assert dubbing.xtts_speed(text, "es", window=6.0) == 1.0
    assert dubbing.xtts_speed(text, "es", window=4.4) == pytest.approx(5.0 / 4.4, abs=1e-3)
    assert dubbing.xtts_speed(text, "es", window=1.0) == dubbing.MAX_XTTS_SPEED
    assert dubbing.MAX_XTTS_SPEED <= 1.25  # above this XTTS starts dropping words


# --- translation -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Accent? I don't have an accent.", ["Accent?", "I don't have an accent."]),
        ("Hola, como estás? Bien y tú?", ["Hola, como estás?", "Bien y tú?"]),
        ("No. Yeah, I agree. I", ["No.", "Yeah, I agree. I"]),  # cut-off word stays attached
        ("It was 2.5 times more.", ["It was 2.5 times more."]),
        ("¿Qué? ¡No! Vale.", ["¿Qué?", "¡No!", "Vale."]),
        ("no punctuation at all", ["no punctuation at all"]),
    ],
)
def test_split_sentences(text, expected) -> None:
    assert dubbing.split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "whole"),
    [
        ("¿Acento? No tengo acento.", True),  # a one-word sentence: XTTS hallucinates on it alone
        ("Dios mío, me encanta tu acento. ¿De dónde eres?", False),
        ("Gracias, soy de España. A mí también me gusta tu acento.", False),
    ],
)
def test_lines_with_a_very_short_sentence_are_synthesised_whole(text, whole) -> None:
    assert dubbing.synthesise_whole(text) is whole


def test_translate_lines_batches_sentences_and_rejoins_them_per_line() -> None:
    class Models:
        def __init__(self):
            self.calls = []

        def translate_many(self, texts, source, target):
            self.calls.append(list(texts))
            return [f"<{t}>" for t in texts]

    models = Models()
    out = dubbing.translate_lines(models, ["A. B?", "C"], "en", "es")
    assert out == ["<A.> <B?>", "<C>"]
    assert models.calls == [["A.", "B?", "C"]]  # one batch for the whole job


# --- speakers in short clips -------------------------------------------------------------------------


def test_short_clips_need_less_speech_per_speaker_long_ones_are_unchanged() -> None:
    assert speakers.min_speaker_seconds(8.2) == pytest.approx(0.3 * 8.2)
    assert speakers.min_speaker_seconds(3.0) == speakers.MIN_SPEAKER_FLOOR_S
    assert speakers.min_speaker_seconds(13.4) == speakers.MIN_SPEAKER_S
    assert speakers.min_speaker_seconds(120.0) == speakers.MIN_SPEAKER_S


def test_a_three_line_short_gets_both_speakers() -> None:
    """The Short that came out in one voice: A asks, B answers (2.9 s of speech), A replies."""
    rng = np.random.default_rng(5)
    a, b = np.zeros(192), np.zeros(192)
    a[0], b[1] = 1.0, 1.0
    embeddings = [v + 0.6 * rng.standard_normal(192) / np.sqrt(192) for v in (a, b, a)]
    durations = [3.56, 2.90, 1.74]
    labels = speakers.cluster_speakers(
        embeddings, durations, min_speaker_s=speakers.min_speaker_seconds(sum(durations))
    )
    assert labels == [0, 1, 0]


# --- the whole loop ------------------------------------------------------------------------------------


class _XTTS:
    """Speaks at 8 chars/s at speed 1 with a 1.2 s pause mid-line and coqui's 0.417 s pad. The first
    take of line 0 is a runaway (four times slower), as measured in real jobs."""

    def __init__(self):
        self.calls: list[tuple[str, int, float]] = []
        self.speaker_encoder = None

    def translate_many(self, texts, source, target):
        return [t.upper() for t in texts]

    def synthesize(self, text, speaker_wav, language, output, seed, speed=1.0,
                   split_sentences=True):
        self.calls.append((text, seed, speed))
        rate = 8.0 * speed
        if seed == 498:  # line 0, first take
            rate /= 4
        seconds = len(text.replace(" ", "")) / rate
        half = _tone(seconds / 2, 200.0)
        audio = np.concatenate([half, _silence(1.2), half, _silence(0.417)])
        soundfile.write(str(output), audio, SR, subtype="PCM_16")

    def speaker_similarity(self, first, second):
        return 0.5


def _run(tmp_path, monkeypatch, segments, factory=None):
    monkeypatch.setenv("BVT_BACKGROUND", "none")
    monkeypatch.setenv("BVT_SPEAKERS", "1")
    upload = tmp_path / "upload.wav"
    t = np.arange(16000 * 10) / 16000
    soundfile.write(str(upload), 0.2 * np.sin(2 * np.pi * 200 * t), 16000)

    def fake_extract(video, out_wav):
        audio, sr = soundfile.read(str(video))
        soundfile.write(str(out_wav), audio, sr)
        return out_wav

    monkeypatch.setattr(dubbing, "extract_audio", fake_extract)
    monkeypatch.setattr(dubbing, "ffprobe_duration", lambda p: soundfile.info(str(p)).duration)
    monkeypatch.setattr(dubbing, "detect_language", lambda models, path: ("en", 0.99))
    monkeypatch.setattr(dubbing, "transcribe_segments", lambda models, path, lang: segments)
    monkeypatch.setattr(dubbing, "align_words", lambda *args, **kwargs: [])
    monkeypatch.setattr(dubbing, "has_video_stream", lambda path: False)
    monkeypatch.setattr(
        dubbing.speakers, "assign_speakers", lambda models, path, segs, forced: [0] * len(segs)
    )
    models = (factory or _XTTS)()
    return dubbing.dub(upload, "en-es", tmp_path / "job", models), models


SHORT = [  # the three lines of the Short, ~3.3 words/s in the original
    {"start": 0.0, "end": 3.5, "text": "oh my god I love your accent. Where are you from?"},
    {"start": 4.0, "end": 6.9, "text": "thank you I am from spain. I like your accent too."},
    {"start": 7.2, "end": 8.9, "text": "accent? I do not have an accent."},
]


def test_dub_never_overlaps_voices_and_caps_compression(tmp_path, monkeypatch) -> None:
    result, _ = _run(tmp_path, monkeypatch, SHORT)
    assert result.failed_segments == 0
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        # never over the next line, and the same speaker keeps a short pause between lines
        assert a.start + a.final_duration <= b.start - 0.1 + 1e-3
    for s in result.segments:
        assert s.trimmed_duration < s.raw_tts_duration - 1.0  # pauses and padding cut first
        assert s.tts_speed <= dubbing.MAX_XTTS_SPEED
        assert 0.67 - 1e-6 <= s.scale_applied <= 1.0  # WSOLA, compress only, never below 0.67


def test_a_runaway_take_is_sampled_again_and_the_fast_take_kept(tmp_path, monkeypatch) -> None:
    result, models = _run(tmp_path, monkeypatch, SHORT)
    seeds = [seed for _, seed, _ in models.calls]
    assert seeds == [498, 498 + 7919, 499, 500]  # only line 0's runaway take was sampled again
    assert result.segments[0].raw_tts_duration < 8.0  # the normal take (~6.5 s), not the ~21 s one
    assert not list((tmp_path / "job" / "segments").glob("*_take*.wav"))  # spare takes removed


def test_a_line_shorter_than_its_slot_is_not_stretched(tmp_path, monkeypatch) -> None:
    lines = [{"start": 0.0, "end": 5.0, "text": "short line"}]
    result, _ = _run(tmp_path, monkeypatch, lines)
    s = result.segments[0]
    assert s.scale_applied == 1.0 and not s.clamped
    assert s.final_duration == pytest.approx(s.trimmed_duration)


def test_default_synthesis_passes_no_speed_so_research_runs_are_unchanged() -> None:
    import inspect

    from bilingual_voice.pipeline import LocalModels

    source = inspect.getsource(LocalModels.synthesize)
    assert 'extra = {} if speed == 1.0 else {"speed": float(speed)}' in source


def test_compression_stops_at_2x_and_the_rest_is_faded_not_squeezed(tmp_path, monkeypatch) -> None:
    lines = [  # far too much text for a 1 s line followed immediately by the next one
        {"start": 0.0, "end": 1.0, "text": "a very long sentence that cannot possibly fit here"},
        {"start": 1.0, "end": 3.0, "text": "next"},
    ]
    result, _ = _run(tmp_path, monkeypatch, lines)
    s = result.segments[0]
    assert s.clamped and s.scale_applied == pytest.approx(0.67)  # WSOLA's measured floor
    assert s.tts_speed == dubbing.MAX_XTTS_SPEED
    assert s.truncated_s > 0 and s.start + s.final_duration <= lines[1]["start"] + 1e-3


class _MishearingXTTS(_XTTS):
    """Whisper stand-in included: it mishears the first take of the second line, then hears right."""

    def __init__(self):
        super().__init__()
        self.last_text = ""
        models = self

        class Ears:
            def transcribe(self, audio, **kwargs):
                _, seed, _ = models.calls[-1]
                if seed == 499:  # line 2, first take
                    return {"text": "algo completamente distinto"}
                return {"text": models.calls[-1][0]}

        self.asr = Ears()


def test_a_misheard_take_is_sampled_again_and_the_clear_one_kept(tmp_path, monkeypatch) -> None:
    lines = [
        {"start": 0.0, "end": 3.0, "text": "first line here."},
        {"start": 4.0, "end": 7.0, "text": "second line here."},
    ]
    result, models = _run(tmp_path, monkeypatch, lines, factory=_MishearingXTTS)
    second = result.segments[1]
    assert [seed for _, seed, _ in models.calls if seed in (499, 499 + 7919)] == [499, 499 + 7919]
    assert second.takes == 2 and second.take_cer == 0.0
    assert result.segments[0].takes <= 2  # line 1's first take is a runaway, resampled once


def test_plain_text_ignores_accents_and_punctuation() -> None:
    assert dubbing._plain("¿Acento? No tengo acento.") == "acento no tengo acento"
    assert dubbing._plain("Él está aquí.") == "el esta aqui"
