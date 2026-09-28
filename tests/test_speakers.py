"""Per-speaker voice references for dubbing (webapp/backend/speakers.py)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from webapp.backend import speakers


def _voice(direction: int, n: int, seed: int, dim: int = 192, noise: float = 0.75):
    """`n` embeddings of one synthetic voice: a fixed direction plus per-segment noise.

    A noise norm of 0.75 puts same-voice cosine near 0.64 and different-voice cosine near 0,
    roughly where ECAPA puts real same- and different-speaker segments.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(dim)
    base[direction] = 1.0
    return [base + noise * rng.standard_normal(dim) / np.sqrt(dim) for _ in range(n)]


def test_two_voices_get_two_labels_in_order_of_first_appearance() -> None:
    a, b = _voice(0, 6, 1), _voice(1, 6, 2)
    order = ["b", "a", "b", "a", "a", "b", "b", "a", "a", "b", "a", "b"]
    embeddings = [(b if who == "b" else a).pop() for who in order]
    labels = speakers.cluster_speakers(embeddings, [2.5] * len(order))
    assert labels == [0 if who == "b" else 1 for who in order]


def test_one_voice_is_not_split() -> None:
    embeddings = _voice(3, 20, 7)
    assert set(speakers.cluster_speakers(embeddings, [2.0] * 20)) == {0}


def test_short_segments_join_the_nearest_voice_instead_of_forming_one() -> None:
    a, b = _voice(0, 5, 1), _voice(1, 5, 2)
    embeddings = a + b + [a[0] * 0.9, b[0] * 0.9]
    durations = [3.0] * 10 + [0.6, 0.6]  # the last two are too short to form a group
    labels = speakers.cluster_speakers(embeddings, durations)
    assert labels[10] == labels[0] and labels[11] == labels[5] and labels[0] != labels[5]


def test_unembeddable_segment_takes_its_neighbour_in_time() -> None:
    a, b = _voice(0, 3, 1), _voice(1, 3, 2)
    embeddings = [*a, None, *b]
    starts = [0, 2, 4, 6.05, 6.35, 8.35, 10.35]
    durations = [2.0, 2.0, 2.0, 0.3, 2.0, 2.0, 2.0]  # 0.3 s fragment touching B's next turn
    labels = speakers.cluster_speakers(embeddings, durations, starts)
    assert labels[3] == labels[4] != labels[0]


def test_a_trailing_fragment_stays_with_the_turn_it_ends() -> None:
    """Distance is the gap between segments, not between their starts: a tail word that Whisper
    split off a long turn belongs to that turn even though the next turn STARTS closer to it."""
    a, b = _voice(0, 3, 1), _voice(1, 3, 2)
    embeddings = [*a, None, *b]
    starts = [0.0, 3.0, 10.5, 18.5, 21.5, 25.5, 29.0]
    durations = [2.5, 6.0, 8.0, 0.35, 3.5, 3.0, 3.0]
    labels = speakers.cluster_speakers(embeddings, durations, starts)
    assert labels[3] == labels[2] != labels[4]


def test_a_voice_with_almost_no_speech_is_folded_into_the_nearest() -> None:
    a, b = _voice(0, 8, 1), _voice(1, 1, 2)
    # b forms its own group (2.0 s >= MIN_EMBED_S) but has under MIN_SPEAKER_S of speech in total.
    labels = speakers.cluster_speakers(a + b, [2.0] * 8 + [2.0])
    assert set(labels) == {0}


def test_short_segments_count_towards_keeping_a_voice() -> None:
    a, b = _voice(0, 8, 1), _voice(1, 3, 2)
    # b: one 2.0 s segment forms the group; two 1.2 s segments join it -> 4.4 s, enough to keep.
    labels = speakers.cluster_speakers(a + b, [2.0] * 8 + [2.0, 1.2, 1.2])
    assert len(set(labels)) == 2 and len(set(labels[8:])) == 1


def test_the_speaker_count_can_be_forced() -> None:
    embeddings = _voice(0, 6, 1) + _voice(1, 6, 2)
    assert set(speakers.cluster_speakers(embeddings, [2.0] * 12, num_speakers=1)) == {0}
    assert len(set(speakers.cluster_speakers(embeddings, [2.0] * 12, num_speakers=2))) == 2


def test_nothing_to_embed_means_one_voice() -> None:
    assert speakers.cluster_speakers([None, None], [0.2, 0.3]) == [0, 0]
    assert speakers.cluster_speakers([], []) == []


@pytest.mark.parametrize(("value", "expected"), [("auto", None), ("", None), ("2", 2)])
def test_requested_speakers(value, expected, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BVT_SPEAKERS", value)
    assert speakers.requested_speakers() == expected


@pytest.mark.parametrize("value", ["0", "two", "-1"])
def test_bad_speaker_setting_fails_loudly(value, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BVT_SPEAKERS", value)
    with pytest.raises(RuntimeError):
        speakers.requested_speakers()


def _two_voice_recording(path: Path, sr: int = 16000):
    """Alternating 2 s turns: speaker A is a 200 Hz tone, speaker B an 800 Hz tone."""
    t = np.arange(int(2.0 * sr)) / sr
    turns, segments, clock = [], [], 0.0
    for k in range(6):
        freq = 200.0 if k % 2 == 0 else 800.0
        turns += [0.5 * np.sin(2 * np.pi * freq * t), np.zeros(int(0.5 * sr))]
        segments.append({"start": clock, "end": clock + 2.0, "text": f"turn {k}"})
        clock += 2.5
    soundfile.write(str(path), np.concatenate(turns), sr)
    return segments


def _dominant_hz(audio, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(audio))
    return float(np.fft.rfftfreq(len(audio), 1 / sr)[int(np.argmax(spectrum))])


def test_references_contain_only_their_own_speaker(tmp_path: Path) -> None:
    wav = tmp_path / "vocals.wav"
    segments = _two_voice_recording(wav)
    labels = [0, 1, 0, 1, 0, 1]
    refs = speakers.build_references(wav, segments, labels, tmp_path / "refs", max_seconds=4.5)
    assert set(refs) == {0, 1}
    for label, freq in [(0, 200.0), (1, 800.0)]:
        audio, sr = soundfile.read(str(refs[label]["path"]))
        assert sr == 16000
        assert abs(_dominant_hz(audio, sr) - freq) < 5
        assert refs[label]["seconds"] <= 4.5 + 1e-6
        assert len(audio) / sr <= 4.5 + 2 * speakers.REFERENCE_GAP_S


def test_assign_speakers_labels_real_audio_through_the_encoder(tmp_path: Path) -> None:
    import torch

    class ToneEncoder:
        """Stands in for ECAPA: the embedding is the energy near 200 Hz and near 800 Hz."""

        def encode_batch(self, wavs):
            audio = wavs.squeeze(0).numpy()
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
            low = spectrum[(freqs > 150) & (freqs < 250)].sum()
            high = spectrum[(freqs > 750) & (freqs < 850)].sum()
            return torch.tensor([[low, high, 1e-3]])

    class Models:
        speaker_encoder = ToneEncoder()

    wav = tmp_path / "vocals_16k.wav"
    segments = _two_voice_recording(wav)
    assert speakers.assign_speakers(Models(), wav, segments) == [0, 1, 0, 1, 0, 1]


# --- calibrated behaviour: these fail if the threshold, linkage or gates drift -------------------


def _realistic(n_voices: int, per_voice: int, same: float, cross: float, seed: int = 3):
    """Embeddings with a chosen mean same-voice and cross-voice cosine, like real ECAPA segments.

    Every vector = shared part + voice part + segment noise, so different voices are not
    orthogonal (real speakers share a lot of the embedding space).
    """
    rng = np.random.default_rng(seed)
    dim = 192
    shared = rng.standard_normal(dim)
    shared /= np.linalg.norm(shared)
    voices = []
    for _ in range(n_voices):
        v = rng.standard_normal(dim)
        voices.append(v / np.linalg.norm(v))
    # cos(same) = (a^2 + b^2) / (a^2 + b^2 + r^2); cos(cross) = a^2 / (a^2 + b^2 + r^2), with the
    # voice-direction and noise vectors near-orthogonal in 192 dimensions.
    a2 = cross / same
    b2 = 1.0 - a2
    r2 = (1.0 - same) / same
    out, who = [], []
    for k, v in enumerate(voices):
        for _ in range(per_voice):
            noise = rng.standard_normal(dim)
            noise /= np.linalg.norm(noise)
            out.append(np.sqrt(a2) * shared + np.sqrt(b2) * v + np.sqrt(r2) * noise)
            who.append(k)
    return out, who


def test_realistic_different_voices_split_at_the_calibrated_threshold() -> None:
    # Cross-voice similarity 0.12 is typical of two different people; a threshold far above the
    # calibrated 0.75 (e.g. 0.95) would merge them back into the one-voice bug. Within-voice 0.40
    # keeps the gap (0.28) below SPLIT_GAP, so only the threshold itself can separate them here.
    embeddings, who = _realistic(2, 8, same=0.40, cross=0.12)
    labels = speakers.cluster_speakers(embeddings, [2.5] * len(who))
    assert len(set(labels)) == 2
    assert all((labels[i] == labels[0]) == (who[i] == who[0]) for i in range(len(who)))


def test_realistic_single_voice_stays_whole_at_the_calibrated_threshold() -> None:
    # Same-voice similarity 0.35 is normal for short, noisy segments of ONE person; a much lower
    # threshold (e.g. 0.4) would split them into several voices.
    embeddings, _ = _realistic(1, 16, same=0.35, cross=0.35)
    assert set(speakers.cluster_speakers(embeddings, [2.0] * 16)) == {0}


def test_similar_voices_with_tight_clean_turns_are_still_separated() -> None:
    """Two people who sound alike (between-speaker cosine 0.32, like two women in DRAL) merge under
    the fixed threshold; long clean turns make each person's segments agree strongly (0.77), and
    that structure is what splits them."""
    embeddings, who = _realistic(2, 6, same=0.77, cross=0.32)
    labels = speakers.cluster_speakers(embeddings, [6.0] * len(who))
    assert all((labels[i] == labels[0]) == (who[i] == who[0]) for i in range(len(who)))


def test_one_voice_with_tight_turns_is_not_split_by_the_cohesion_test() -> None:
    embeddings, _ = _realistic(1, 12, same=0.77, cross=0.77)
    assert set(speakers.cluster_speakers(embeddings, [6.0] * 12)) == {0}


def test_linkage_is_average_not_single() -> None:
    """Points along an arc: each is close to its neighbours, the ends are opposite. Single linkage
    chains the whole arc into one voice; average linkage (the calibrated choice) does not."""
    angles = np.linspace(0.0, np.pi, 12)
    embeddings = [np.r_[np.cos(t), np.sin(t), np.zeros(190)] for t in angles]
    assert len(set(speakers.cluster_speakers(embeddings, [2.0] * 12))) >= 2


def test_forced_count_is_obeyed_even_when_auto_would_differ() -> None:
    two = _voice(0, 6, 1) + _voice(1, 6, 2)
    assert len(set(speakers.cluster_speakers(two, [2.0] * 12, num_speakers=3))) == 3
    one = _voice(0, 12, 5)
    assert len(set(speakers.cluster_speakers(one, [2.0] * 12, num_speakers=2))) == 2


def test_forced_count_gives_a_short_lines_speaker_their_own_voice() -> None:
    """B only answers in short lines. Forcing two voices must separate A from B, not split A."""
    a, b = _voice(0, 6, 1), _voice(1, 6, 2)
    embeddings = [x for pair in zip(a, b, strict=True) for x in pair]
    durations = [3.0, 1.0] * 6
    labels = speakers.cluster_speakers(embeddings, durations, num_speakers=2)
    assert set(labels[0::2]) == {0} and set(labels[1::2]) == {1}


def test_auto_mode_rescues_a_distinct_speaker_who_only_uses_short_lines() -> None:
    a, b = _voice(0, 6, 1), _voice(1, 6, 2)
    embeddings = [x for pair in zip(a, b, strict=True) for x in pair]
    labels = speakers.cluster_speakers(embeddings, [3.0, 1.0] * 6)  # B: 6 s of short lines
    assert set(labels[0::2]) == {0} and set(labels[1::2]) == {1}


def test_noisy_short_segments_do_not_form_groups_of_their_own() -> None:
    """Short clips of the SAME person agree with each other poorly (pairwise cosine ~0.12 here) but
    each still leans towards that person. Only long segments may form groups, so these must join
    A -- if short segments were clustered directly they would split off, even with the fold off."""
    rng = np.random.default_rng(11)
    base = np.zeros(192)
    base[0] = 1.0
    long_a = [base + 0.6 * rng.standard_normal(192) / np.sqrt(192) for _ in range(6)]
    short_a = [base + 2.7 * rng.standard_normal(192) / np.sqrt(192) for _ in range(6)]
    labels = speakers.cluster_speakers(
        long_a + short_a, [3.0] * 6 + [1.0] * 6, min_speaker_s=0.0
    )
    assert set(labels) == {0}


def _tone_recording(path: Path, turns, sr: int = 16000):
    """Write turns [(seconds, hz, amplitude)] back to back with 0.5 s gaps; return segments."""
    audio, segments, clock = [], [], 0.0
    for seconds, hz, amp in turns:
        t = np.arange(int(seconds * sr)) / sr
        audio += [amp * np.sin(2 * np.pi * hz * t), np.zeros(int(0.5 * sr))]
        segments.append({"start": clock, "end": clock + seconds, "text": "x"})
        clock += seconds + 0.5
    soundfile.write(str(path), np.concatenate(audio), sr)
    return segments


def test_references_use_the_longest_segments_first(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    segments = _tone_recording(wav, [(0.7, 300, 0.3), (3.0, 300, 0.3), (2.0, 300, 0.3)])
    refs = speakers.build_references(wav, segments, [0, 0, 0], tmp_path / "r", max_seconds=3.0)
    # Longest first: the 3.0 s turn (trimmed to 2.8 s) plus 0.2 s of the 2.0 s one; never the 0.7 s.
    assert refs[0]["segments"] == 2
    assert refs[0]["seconds"] == pytest.approx(3.0, abs=0.01)


def test_short_pieces_are_used_only_to_reach_the_target(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    turns = [(3.0, 300, 0.3)] * 3 + [(0.6, 300, 0.3)] * 4  # 9 s of long turns, then short ones
    segments = _tone_recording(wav, turns)
    refs = speakers.build_references(wav, segments, [0] * 7, tmp_path / "r")
    assert refs[0]["segments"] == 3  # target of 8 s reached by the long turns alone


def test_a_speaker_with_too_little_audio_gets_no_reference(tmp_path: Path) -> None:
    wav = tmp_path / "v.wav"
    segments = _tone_recording(wav, [(3.0, 300, 0.3), (0.45, 600, 0.3)])
    refs = speakers.build_references(wav, segments, [0, 1], tmp_path / "r")
    assert 0 in refs and 1 not in refs  # 0.45 s would be under XTTS's usable minimum


def test_references_share_one_loudness_despite_a_transient(tmp_path: Path) -> None:
    from webapp.backend.background import _active_rms

    wav = tmp_path / "v.wav"
    segments = _tone_recording(wav, [(3.0, 300, 0.2), (3.0, 600, 0.2)])
    audio, sr = soundfile.read(str(wav))
    bang = int((segments[1]["start"] + 1.0) * sr)
    audio[bang] = 0.99  # a door slam inside speaker 2's turn
    soundfile.write(str(wav), audio, sr)
    refs = speakers.build_references(wav, segments, [0, 1], tmp_path / "r")
    levels = []
    for label in (0, 1):
        ref, _ = soundfile.read(str(refs[label]["path"]))
        levels.append(20 * np.log10(_active_rms(ref[np.newaxis, :])))
    assert abs(levels[0] - levels[1]) < 1.0
    assert abs(levels[0] - 20 * np.log10(speakers.REFERENCE_RMS)) < 1.0


# --- the wiring in dub(): which reference does each segment actually get? ------------------------


class _ToneEncoder:
    def encode_batch(self, wavs):
        import torch

        audio = wavs.squeeze(0).numpy()
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / 16000)
        low = spectrum[(freqs > 150) & (freqs < 250)].sum()
        high = spectrum[(freqs > 750) & (freqs < 850)].sum()
        return torch.tensor([[low, high, 1e-3]])


class _FailingEncoder:
    def encode_batch(self, wavs):
        raise RuntimeError("encoder unavailable")


class _StubModels:
    def __init__(self, encoder):
        self.speaker_encoder = encoder
        self.references: list[Path] = []

    def translate_many(self, texts, source, target):
        return [f"traducido {i}" for i in range(len(texts))]

    def synthesize(self, text, speaker_wav, language, output, seed, speed=1.0,
                   split_sentences=True):
        self.references.append(Path(speaker_wav))
        t = np.arange(24000) / 24000
        soundfile.write(str(output), 0.3 * np.sin(2 * np.pi * 440 * t), 24000, subtype="PCM_16")

    def speaker_similarity(self, first, second):
        return 0.5


def _run_dub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoder):
    from webapp.backend import dubbing

    monkeypatch.setenv("BVT_BACKGROUND", "none")
    monkeypatch.delenv("BVT_SPEAKERS", raising=False)
    upload = tmp_path / "upload.wav"
    segments = _two_voice_recording(upload)

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
    models = _StubModels(encoder)
    result = dubbing.dub(upload, "en-es", tmp_path / "job", models)
    return result, models


def test_dub_voices_each_speaker_from_their_own_reference(tmp_path, monkeypatch) -> None:
    result, models = _run_dub(tmp_path, monkeypatch, _ToneEncoder())
    speaker_dir = tmp_path / "job" / "speakers"
    assert [s.speaker for s in result.segments] == [1, 2, 1, 2, 1, 2]
    # XTTS clones from the denoised copy of each speaker's reference
    assert models.references == [speaker_dir / f"speaker_{k}_xtts.wav" for k in [1, 2, 1, 2, 1, 2]]
    assert [row["speaker"] for row in result.speakers] == [1, 2]
    assert all(row["voiced"] == 3 for row in result.speakers)
    assert result.speaker_detection_error == ""


def test_failed_detection_is_reported_and_never_falls_back_to_the_soundtrack(
    tmp_path, monkeypatch
) -> None:
    result, models = _run_dub(tmp_path, monkeypatch, _FailingEncoder())
    assert "encoder unavailable" in result.speaker_detection_error
    shared = tmp_path / "job" / "speakers" / "shared" / "speaker_1_xtts.wav"
    assert models.references == [shared] * 6  # one clean shared voice, not source.wav
