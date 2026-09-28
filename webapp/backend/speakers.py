"""Who is speaking in each segment, and a voice reference for each of them.

XTTS clones whatever voice its reference audio contains. The dubbing pass used to hand it the whole
soundtrack for every segment, and XTTS reads only the first 30 s of a reference (`max_ref_len` and
`gpt_cond_len` in the model config), so every actor came out in one voice: a blend of whoever spoke
first, music included. Zero-shot cloning is per speaker, so the reference has to be too.

Each Whisper segment is labelled with a speaker by clustering ECAPA embeddings -- the same
encoder the pipeline already uses to measure speaker similarity -- and each speaker gets a reference
built only from their own segments, cut from the isolated vocal stem when separation ran.
"""

from __future__ import annotations

import os
from pathlib import Path

# Clustering settings, calibrated on 1,338 simulated videos built from DRAL (87 speakers per
# language, 1-5 speakers per video, with and without added reverb and music residue). At these
# settings a one-speaker video is split into two voices 0.6-6 % of the time (clean to harsh
# audio), and 2-speaker videos have 90-93 % of segments voiced by the right speaker. The trade
# leans towards merging a minor speaker rather than splitting one actor into two voices, because a
# voice that changes mid-scene is the more jarring failure.
# Cosine distance at which two groups stop being merged (average linkage = mean similarity 0.25).
DISTANCE_THRESHOLD = 0.75
LINKAGE = "average"
# Segments shorter than this embed too noisily to be trusted when forming speaker groups; they are
# attached to the closest group afterwards instead.
MIN_EMBED_S = 1.6
# Short segments that fit no voice are grouped among themselves at this looser distance: short
# clips of one speaker agree with each other less than long ones do (median same-speaker similarity
# 0.20 under 1 s, 0.40 at 1-2 s), so the main threshold would scatter them into tiny groups. On
# DRAL partners where one person only answers in short lines, 0.90 gives that person their own
# voice in 56-68 % of dialogues (0 % without this pass) and leaves the split rate of one-speaker
# videos unchanged; 0.95 recovers more but starts splitting one-speaker videos in noisy audio.
OUTLIER_DISTANCE = 0.90
# A group is split in two when its halves are each tight and clearly apart: mean within-half
# similarity at least SPLIT_GAP above the similarity between the halves, which must itself stay
# under SPLIT_MAX_CROSS. This separates similar-sounding people with long, clean turns that the
# fixed threshold merges -- two women in a DRAL conversation, between-speaker similarity 0.32 but
# within-speaker 0.77, went from one voice to 10/10 segments correct -- while noisy segments of one
# person, which agree with each other only weakly, stay whole: on the 1,338 simulated videos the
# one-speaker split rate is unchanged for any gap from 0.30 to 0.45. 0.35 is mid-range; 0.45
# already misses the DRAL pair.
SPLIT_GAP = 0.35
SPLIT_MAX_CROSS = 0.40
# Shortest audio worth embedding at all. Anything shorter takes the label of its nearest neighbour
# in time, which in conversation is usually the same turn.
MIN_ASSIGN_S = 0.4
# A group whose speech adds up to less than this is more often an outlier segment -- a laugh, a
# shout, music residue -- than a real extra speaker, and below ~4 s the XTTS conditioning is too
# unstable to give a distinguishable clone anyway: fold it into its nearest group. This rule is what
# keeps one-speaker videos in one voice (it cuts the split rate about fourfold).
MIN_SPEAKER_S = 4.0
# In a short clip nobody reaches 4 s -- a 9 s Short with three lines has at most ~3.5 s per person --
# so the requirement shrinks to a share of the clip's total speech: min(4, max(1.5, 0.3 T)). On
# 6,000 simulated 2-8 line clips this separates the speakers in 45 % of multi-speaker clips (34 %
# before) and in 85 % of Short-like dialogues whose second speaker has 2.5-4 s (2 % before). It
# changes nothing at 13.3 s of speech or more, so long-video behaviour is exactly as calibrated;
# one-speaker short clips split 3.3 % of the time instead of 2.2 % (moderate noise).
MIN_SPEAKER_SHARE = 0.3
MIN_SPEAKER_FLOOR_S = 1.5
# Reference audio per speaker. XTTS conditions on at most 30 s of a file, gaps included; measured
# conditioning barely changes past 15 s (d-vector cosine 0.97 to the 25 s version).
REFERENCE_MAX_S = 25.0
# Reference pieces shorter than this are only used while the reference is under
# REFERENCE_TARGET_S: short Whisper segments carry more boundary bleed from the neighbouring turn.
REFERENCE_MIN_PIECE_S = 1.0
# Measured XTTS conditioning reaches 0.93 of its 25 s value by 8 s of speech, so aim for at least
# that before falling back on short pieces.
REFERENCE_TARGET_S = 8.0
# Below this there is nothing to condition on (and XTTS rejects references under 0.33 s).
MIN_REFERENCE_S = 1.0
# Active-speech RMS every reference is brought to (-20 dBFS).
REFERENCE_RMS = 0.1
# XTTS copies the reference's noise floor into every line it speaks, and noisy references are where
# its runaway takes (lines drawn out 2-5x) cluster. A light spectral gate on the copy XTTS clones
# from took runaways from 9/48 to 0/48 takes, total length -27 % and Whisper WER 0.107 -> 0.025 on
# 12 real lines x 4 seeds, with cleaner harmonics (HNR +0.9 dB). The ungated reference is kept for
# measuring speaker similarity.
DENOISE_STD = 1.5
DENOISE_MAX_REDUCTION_DB = 12.0
REFERENCE_GAP_S = 0.15
# Whisper's boundaries often clip into the neighbouring turn; trim this much off both ends of a
# segment before using it as reference audio, when the segment is long enough to spare it.
EDGE_TRIM_S = 0.1


def min_speaker_seconds(total_speech_s: float) -> float:
    """Least speech a voice needs to be kept as its own speaker, given the clip's total speech."""
    return min(MIN_SPEAKER_S, max(MIN_SPEAKER_FLOOR_S, MIN_SPEAKER_SHARE * total_speech_s))


def requested_speakers() -> int | None:
    """`BVT_SPEAKERS`: "auto" (default) detects the speakers; a number forces that many."""
    value = os.environ.get("BVT_SPEAKERS", "auto").strip().lower()
    if value in {"", "auto"}:
        return None
    message = f"BVT_SPEAKERS must be 'auto' or a positive integer, got {value!r}"
    try:
        count = int(value)
    except ValueError as exc:
        raise RuntimeError(message) from exc
    if count < 1:
        raise RuntimeError(message)
    return count


def _unit(vectors):
    import numpy as np

    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1.0)


def cluster_speakers(
    embeddings,
    durations,
    starts=None,
    *,
    num_speakers: int | None = None,
    threshold: float = DISTANCE_THRESHOLD,
    min_embed_s: float = MIN_EMBED_S,
    min_speaker_s: float = MIN_SPEAKER_S,
) -> list[int]:
    """Label each segment with a speaker, numbered in order of first appearance from 0.

    `embeddings[i]` is None for a segment too short to embed. In automatic mode only segments of at
    least `min_embed_s` form the first groups; shorter ones join the closest group -- unless they fit
    no group at all, in which case they may form a voice of their own (someone who only ever
    answers in short lines). With `num_speakers` set, every embeddable segment is clustered into
    exactly that many groups. Segments with no embedding take the label of the segment nearest to
    them in time.
    """
    import numpy as np

    n = len(durations)
    if n == 0:
        return []
    if starts is None:
        starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    starts = [float(x) for x in starts]
    ends = [starts[i] + float(durations[i]) for i in range(n)]
    have = [i for i in range(n) if embeddings[i] is not None]
    if not have:
        return [0] * n
    unit = {i: _unit(embeddings[i]) for i in have}
    floor = 1.0 - threshold  # similarity below which a segment belongs to no existing group

    def link(indices, count=None, distance=threshold):
        if len(indices) < 2 or count == 1:
            return [0] * len(indices)
        from scipy.cluster.hierarchy import fcluster, linkage

        tree = linkage(np.stack([unit[i] for i in indices]), method=LINKAGE, metric="cosine")
        if count is not None:
            found = fcluster(tree, t=min(count, len(indices)), criterion="maxclust")
        else:
            found = fcluster(tree, t=distance, criterion="distance")
        return [int(x) - 1 for x in found]

    if num_speakers is not None:
        # The operator has said how many voices there are. Clustering only the long segments into
        # that many groups would split the main speaker whenever another speaks only in short lines.
        group_of = dict(zip(have, link(have, num_speakers), strict=True))
    else:
        core = [i for i in have if durations[i] >= min_embed_s]
        if len(core) < 2:
            core = list(have)
        group_of = dict(zip(core, link(core), strict=True))

    def cohesive_split():
        """Split any group whose long segments fall into two tight, well-separated halves."""
        while True:
            for group in sorted(set(group_of.values())):
                members = [i for i, g in group_of.items() if g == group and durations[i] >= min_embed_s]
                if len(members) < 4:
                    continue
                halves = link(members, 2)
                parts = [[m for m, h in zip(members, halves, strict=True) if h == k] for k in (0, 1)]
                if min(len(part) for part in parts) < 2 or any(
                    sum(durations[i] for i in part) < min_speaker_s for part in parts
                ):
                    continue
                vecs = [np.stack([unit[i] for i in part]) for part in parts]
                within = [
                    float((v @ v.T)[np.triu_indices(len(v), 1)].mean()) for v in vecs
                ]
                cross = float((vecs[0] @ vecs[1].T).mean())
                if cross < SPLIT_MAX_CROSS and min(within) - cross >= SPLIT_GAP:
                    fresh = max(group_of.values()) + 1
                    for i in parts[1]:
                        group_of[i] = fresh
                    break  # regroup and look again
            else:
                return

    def centroids():
        groups = sorted(set(group_of.values()))
        vectors = [np.mean([unit[i] for i, g in group_of.items() if g == group], axis=0)
                   for group in groups]
        return groups, _unit(vectors)

    def assign(groups, cents):
        out = dict(group_of)
        for i in have:
            if i not in out:
                out[i] = groups[int(np.argmax(cents @ unit[i]))]
        return out

    if num_speakers is None:
        cohesive_split()

    if num_speakers is None:
        # Short segments that resemble no group are clustered among themselves; a group of them
        # with enough speech, and a centroid unlike every existing voice, is a speaker who only
        # talks in short lines. Requiring both keeps one actor's noisier lines from splitting off.
        groups, cents = centroids()
        outliers = [i for i in have if i not in group_of and float(np.max(cents @ unit[i])) < floor]
        found = link(outliers, distance=OUTLIER_DISTANCE)
        fresh = max(groups) + 1
        for candidate in sorted(set(found)):
            members = [i for i, g in zip(outliers, found, strict=True) if g == candidate]
            centre = _unit(np.mean([unit[i] for i in members], axis=0))
            if (sum(durations[i] for i in members) >= min_speaker_s
                    and float(np.max(cents @ centre)) < floor):
                for i in members:
                    group_of[i] = fresh
                fresh += 1

    groups, cents = centroids()
    assigned = assign(groups, cents)
    # Fold groups with too little speech -- counted after the short segments have joined them --
    # into their nearest neighbour, smallest first, unless the caller fixed the number of speakers.
    while num_speakers is None and len(groups) > 1:
        totals = {g: sum(durations[i] for i, lab in assigned.items() if lab == g) for g in groups}
        smallest = min(groups, key=lambda g: totals[g])
        if totals[smallest] >= min_speaker_s:
            break
        index = groups.index(smallest)
        sims = cents @ cents[index]
        sims[index] = -np.inf
        target = groups[int(np.argmax(sims))]
        group_of = {i: (target if g == smallest else g) for i, g in group_of.items()}
        groups, cents = centroids()
        assigned = assign(groups, cents)

    labels = [assigned.get(i) for i in range(n)]
    labelled = [i for i in range(n) if labels[i] is not None]

    def gap(i: int, j: int) -> float:
        return max(0.0, starts[j] - ends[i], starts[i] - ends[j])

    for i in range(n):
        if labels[i] is None:
            labels[i] = labels[min(labelled, key=lambda j: gap(i, j))]

    # Renumber by first appearance so "Speaker 1" is whoever talks first.
    order: dict[int, int] = {}
    for label in labels:
        order.setdefault(label, len(order))
    return [order[label] for label in labels]


def _mono(path: Path):
    import numpy as np
    import soundfile

    audio, sample_rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    return np.ascontiguousarray(audio.mean(axis=1)), int(sample_rate)


def spectral_gate(audio, sr: int, n_std: float = DENOISE_STD,
                  max_reduction_db: float = DENOISE_MAX_REDUCTION_DB):
    """Light stationary noise gate: the noise profile is taken from the quietest non-silent frames
    and each time-frequency bin below mean + n_std x std of it is turned down by at most
    `max_reduction_db` (a soft mask, so speech is never gated hard)."""
    import librosa
    import numpy as np
    from scipy import ndimage

    audio = np.asarray(audio, dtype=float)
    n_fft, hop = 2048, 512
    if audio.size < n_fft or not np.any(audio):
        return audio
    spectrum = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(spectrum)
    level_db = 20 * np.log10(magnitude + 1e-10)
    energy = (magnitude**2).sum(axis=0)
    # References contain digital-silence gaps between pieces; they say nothing about the noise.
    live = energy > 1e-8 * energy.max()
    noise = live & (energy <= np.percentile(energy[live], 15))
    if noise.sum() < 5:
        noise = live
    gate = level_db[:, noise].mean(axis=1) + n_std * level_db[:, noise].std(axis=1)
    mask = ndimage.uniform_filter((level_db > gate[:, None]).astype(float), size=(5, 7))
    floor = 10 ** (-max_reduction_db / 20)
    return librosa.istft(spectrum * (floor + (1 - floor) * mask), hop_length=hop, length=len(audio))


def embed_segments(encoder, audio_16k, segments, min_seconds: float = MIN_ASSIGN_S) -> list:
    """ECAPA embedding of each segment's audio (16 kHz mono array), None when too short."""
    import numpy as np
    import torch

    embeddings = []
    with torch.inference_mode():
        for seg in segments:
            start = max(0, int(float(seg["start"]) * 16000))
            end = min(len(audio_16k), int(float(seg["end"]) * 16000))
            if end - start < int(min_seconds * 16000):
                embeddings.append(None)
                continue
            chunk = torch.tensor(np.asarray(audio_16k[start:end], dtype=np.float32)).unsqueeze(0)
            vector = encoder.encode_batch(chunk).squeeze()
            embeddings.append(np.asarray(vector.detach().cpu().numpy(), dtype=float).reshape(-1))
    return embeddings


def assign_speakers(models, audio_16k_path: Path, segments, num_speakers: int | None = None):
    """Speaker label for every segment of `audio_16k_path` (the file Whisper transcribed)."""
    audio, sample_rate = _mono(audio_16k_path)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    embeddings = embed_segments(models.speaker_encoder, audio, segments)
    durations = [float(s["end"]) - float(s["start"]) for s in segments]
    starts = [float(s["start"]) for s in segments]
    return cluster_speakers(
        embeddings,
        durations,
        starts,
        num_speakers=num_speakers,
        min_speaker_s=min_speaker_seconds(sum(durations)),
    )


def build_references(
    voice_path: Path,
    segments,
    labels,
    out_dir: Path,
    max_seconds: float = REFERENCE_MAX_S,
) -> dict[int, dict]:
    """Write one reference wav per speaker from that speaker's own segments.

    `voice_path` should be the cleanest recording of the voices available -- the separated vocal
    stem when separation ran. The longest segments are used first: they carry the least boundary
    contamination from the neighbouring turn, and short ones are only added while the reference is
    still under `REFERENCE_TARGET_S`. A speaker with under `MIN_REFERENCE_S` of usable audio gets no
    reference, so the caller falls back instead of handing XTTS a clip too short to condition on.
    Returns {label: {"path", "xtts_path", "seconds", "segments"}}.
    """
    import numpy as np
    import soundfile

    from .background import _active_rms

    audio, sample_rate = _mono(voice_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(REFERENCE_GAP_S * sample_rate))
    trim = int(EDGE_TRIM_S * sample_rate)
    budget = int(max_seconds * sample_rate)
    target = int(REFERENCE_TARGET_S * sample_rate)
    references: dict[int, dict] = {}
    for label in sorted(set(labels)):
        members = [i for i, lab in enumerate(labels) if lab == label]
        length = {i: float(segments[i]["end"]) - float(segments[i]["start"]) for i in members}
        members.sort(key=lambda i: length[i], reverse=True)
        pieces, total, used = [], 0, 0
        for i in members:
            if length[i] < REFERENCE_MIN_PIECE_S and total >= target:
                break  # longest-first order: everything after this is short too
            start = max(0, int(float(segments[i]["start"]) * sample_rate))
            end = min(len(audio), int(float(segments[i]["end"]) * sample_rate))
            if end - start > 2 * trim + sample_rate:  # trim only what can spare it
                start, end = start + trim, end - trim
            piece = audio[start:end][: budget - total]
            if piece.size == 0:
                continue
            pieces.extend([piece, gap])
            total += piece.size
            used += 1
            if total >= budget:
                break
        if total < int(MIN_REFERENCE_S * sample_rate):
            continue
        reference = np.concatenate(pieces[:-1])
        level = _active_rms(reference[np.newaxis, :])
        if level > 0:
            # XTTS's style latent is level-sensitive (its d-vector is not), so every speaker's
            # reference is brought to the same loudness. Loudness, not peak: one door slam in the
            # stem would otherwise set the whole reference ~14 dB too quiet.
            reference = np.clip(reference * (REFERENCE_RMS / level), -0.99, 0.99)
        path = out_dir / f"speaker_{label + 1}.wav"
        soundfile.write(str(path), reference.astype(np.float32), sample_rate, subtype="PCM_16")
        xtts_path = out_dir / f"speaker_{label + 1}_xtts.wav"
        gated = np.clip(spectral_gate(reference, sample_rate), -0.99, 0.99)
        soundfile.write(str(xtts_path), gated.astype(np.float32), sample_rate, subtype="PCM_16")
        references[label] = {
            "path": path,  # as heard: the yardstick for speaker similarity
            "xtts_path": xtts_path,  # denoised: what XTTS clones from
            "seconds": round(total / sample_rate, 2),
            "segments": used,
        }
    return references
