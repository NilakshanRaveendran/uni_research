"""Deterministic speaker-component-disjoint dataset splitting."""

from __future__ import annotations

import random


def _speaker_components(rows: list[dict[str, str]]) -> list[list[str]]:
    speakers = sorted(
        {
            speaker
            for row in rows
            for speaker in (
                row.get("english_speaker_id", ""),
                row.get("spanish_speaker_id", ""),
            )
            if speaker
        }
    )
    parent = {speaker: speaker for speaker in speakers}

    def find(speaker: str) -> str:
        while parent[speaker] != speaker:
            parent[speaker] = parent[parent[speaker]]
            speaker = parent[speaker]
        return speaker

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for row in rows:
        english = row.get("english_speaker_id", "")
        spanish = row.get("spanish_speaker_id", "")
        if not english or not spanish:
            raise ValueError(f"Missing speaker ID for pair {row.get('pair_id', '<unknown>')}")
        union(english, spanish)

    components: dict[str, list[str]] = {}
    for speaker in speakers:
        components.setdefault(find(speaker), []).append(speaker)
    return sorted(components.values(), key=lambda item: tuple(item))


def assign_speaker_splits(
    rows: list[dict[str, str]],
    seed: int = 498,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> list[dict[str, str]]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    total = sum(ratios)
    ratios = tuple(ratio / total for ratio in ratios)

    components = _speaker_components(rows)
    if len(components) < 3:
        raise ValueError("At least three disconnected speaker components are required")
    random.Random(seed).shuffle(components)
    component_for_speaker = {
        speaker: index for index, group in enumerate(components) for speaker in group
    }
    weights = {index: 0 for index in range(len(components))}
    for row in rows:
        weights[component_for_speaker[row["english_speaker_id"]]] += 1

    # Assign large components first to the least-filled target partition. Randomizing
    # before the stable weight sort makes equal-weight tie-breaking seed-dependent.
    weighted = list(enumerate(components))
    weighted.sort(key=lambda item: weights[item[0]], reverse=True)
    names = ("train", "dev", "test")
    targets = {name: len(rows) * ratio for name, ratio in zip(names, ratios, strict=True)}
    current = {name: 0 for name in names}
    component_split: dict[int, str] = {}
    for index, _group in weighted:
        split = min(names, key=lambda name: current[name] / targets[name])
        component_split[index] = split
        current[split] += weights[index]
    assignment = {
        speaker: component_split[index] for speaker, index in component_for_speaker.items()
    }

    output = []
    for original in rows:
        row = dict(original)
        row["split"] = assignment[row["english_speaker_id"]]
        output.append(row)
    return output


def assert_speaker_disjoint(rows: list[dict[str, str]]) -> None:
    memberships: dict[str, set[str]] = {}
    for row in rows:
        for field in ("english_speaker_id", "spanish_speaker_id"):
            memberships.setdefault(row[field], set()).add(row["split"])
    leaked = {speaker: splits for speaker, splits in memberships.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Speaker leakage detected: {leaked}")
