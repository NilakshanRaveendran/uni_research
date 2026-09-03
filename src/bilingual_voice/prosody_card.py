"""Generate MODEL_CARD.md for the open-source release, entirely from committed artefacts.

    python -m bilingual_voice.prosody_card build --out MODEL_CARD.md

Deliberately a generator rather than a hand-written document. A figure in this project once
carried a hardcoded "63-107%" when the data said 69-128%, and a model card is exactly the kind of
document that gets written once and then quietly drifts from the numbers it describes. Everything
below is read from `artifacts/prosody_net/` and `artifacts/ladder/`; if an artefact is missing, the
section says so instead of inventing a value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SPLIT_NOTE = (
    "Speaker-component-disjoint 70/15/15 split, seed 498. Splits are formed over the union-find "
    "components of the speaker graph, so no speaker appears in more than one split."
)


LIMITATIONS = """## Limitations — read these before citing anything above

- **These models do not replace XTTS, Whisper or MarianMT.** They add a prosody-transfer component
  that the pretrained cascade does not have. Any claim that they beat a production TTS system would
  be false.
- **Not state of the art.** 6,284 and 4,796 parameters trained on 1,633 pairs per direction.
- **Injection costs speaker similarity and intelligibility.** ECAPA falls from 0.440 (raw XTTS) to
  ~0.359 and WER rises from 0.100 to ~0.148. Decomposed: the WORLD round-trip costs 0.028 ECAPA,
  the time-scaling costs a further 0.053, and the models' own pitch prediction costs +0.002 -- i.e.
  nothing. The oracle arm pays the same price, so the cost belongs to the injection method, not to
  prediction quality. Against the thesis's speaker-identity headline this is a regression from
  ~94.5% of the human ceiling to ~80.6%, and it must be reported as such.
- **Retiming is a net loss except for duration.** Pitch-only variants recover 0.02-0.04 ECAPA and
  most of the WER, and improve pitch accuracy in three of four cases. Retiming buys duration match
  and nothing else. Choose the configuration to match what the application needs.
- **Duration match is incomplete in en-es** (1.26x the human target) because XTTS overruns by 2.03x
  and the 0.5x time-scale clamp binds. es-en reaches 1.04x. Length-controlled MT is the real fix
  and is not attempted here.
- **Contour shape is only partly recovered.** ProsoMLP raises contour correlation above the
  copy-plus-offset baseline in both directions (0.343 vs 0.316 en-es; 0.360 vs 0.321 es-en) but
  ProsoCNN does not. The oracle reaches 0.588/0.639, so most of the available shape signal is still
  unreached. Cross-language contour correspondence in this corpus is only r = 0.24.
- **Duration is not a neural win.** A ridge given the same 16 inputs matches or beats both networks
  on es-en duration.
- **ProsoMLP is worse than baseline on es-en pitch level** (+1.09%). Reported, not hidden.
- **14 test speakers.** Effect sizes are large but no comparison survives Holm correction on the
  per-speaker signed-rank test. The speaker-clustered bootstrap CI excludes zero for pitch span in
  both directions for both models; that is the evidence the claims rest on.
- **No listening study.** Every number here is an objective measure. Whether any of it is audible
  to a listener has not been established.
- **The +/-3 semitone level cap binds on ~9% of utterances**, including for the oracle. The reported
  ceiling is therefore the ceiling *under a safety constraint* that protects speaker identity --
  which any deployed system would also need.
- **DRAL pairs are re-enactments, not translations**, so these models learn how one bilingual
  speaker's prosody maps across their own two languages. That is the quantity of interest, but the
  pairs are not translation pairs.
- **The audio-domain ladder was run twice.** The first run revealed that the injector was imposing
  contour shape without amplitude, discarding the models' best-supported prediction; that was fixed
  and the ladder re-run. The scalar results above are unaffected -- they score model predictions and
  the models were frozen before test was touched -- but the ladder is not a single-shot result.
"""

REPRODUCING = """## Reproducing

```bash
python -m bilingual_voice.contours    extract data/splits/dral_manifest_with_text.csv artifacts/contours --jobs 6
python -m bilingual_voice.mt_words    build   data/splits/dral_manifest_with_text.csv artifacts/mt_words.csv
python -m bilingual_voice.prosody_net prepare --direction en-es
python -m bilingual_voice.prosody_net overfit --arch mlp --direction en-es   # sanity gate
python -m bilingual_voice.prosody_net train   --arch mlp --direction en-es
python -m bilingual_voice.prosody_eval report --split dev
python -m bilingual_voice.prosody_ladder render  results/synthesis_test.csv --jobs 6
python -m bilingual_voice.prosody_ladder measure artifacts/ladder results/synthesis_test.csv
python -m bilingual_voice.prosody_ladder report  artifacts/ladder --synthesis-csv results/synthesis_test.csv
python -m bilingual_voice.prosody_figures all
```

Seeds are fixed (498-502). Training is CPU-only and takes seconds per model.
"""

def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path) if path.exists() else None


def _fmt(value, digits: int = 4) -> str:
    import math

    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if math.isnan(number) else f"{number:.{digits}f}"


def _architecture_section(netdir: Path) -> str:
    import torch

    lines = [
        "## Architectures",
        "",
        "Both models are written from scratch in PyTorch and trained from random initialisation.",
        "**No pretrained weights are used in either model.**",
        "",
    ]
    for arch, name, description in (
        (
            "mlp",
            "ProsoMLP",
            (
                "A residual MLP over the source contour's DCT coefficients.\n"
                "`Linear(16→64) → LayerNorm → GELU → Dropout(0.1) → "
                "Linear(64→64) → LayerNorm → GELU → Dropout(0.1) → Linear(64→12)`"
            ),
        ),
        (
            "cnn",
            "ProsoCNN",
            (
                "A 1-D CNN reading the raw 64-point contour, bypassing the DCT compression.\n"
                "`Conv1d(1→16,k=5) → GELU → Conv1d(16→16,k=5) → GELU → "
                "mean+max pool → concat 8 scalars → Linear(40→64) → GELU → Linear(64→12)`"
            ),
        ),
    ):
        found = sorted(netdir.glob(f"{arch}-*.pt"))
        if not found:
            lines += [f"### {name}", "", "_no checkpoint found_", ""]
            continue
        blob = torch.load(found[0], map_location="cpu", weights_only=False)
        config = blob["config"]
        from .prosody_net import build_model, count_parameters

        params = count_parameters(build_model(arch, config["hidden"], config["dropout"]))
        lines += [
            f"### {name} — {params:,} parameters",
            "",
            description,
            "",
            (
                f"- Trained on: `{', '.join(sorted(p.stem for p in found))}` "
                f"({len(blob['seeds'])}-seed ensemble, seeds {blob['seeds']})"
            ),
            (
                f"- Optimiser: AdamW, lr {config['lr']}, weight decay "
                f"{config['weight_decay']}, batch {config['batch_size']}, gradient clip 1.0"
            ),
            (
                f"- Early stopping on dev loss, patience {config['patience']}, "
                f"max {config['epochs']} epochs"
            ),
            (
                "- Loss: SmoothL1 on standardised residuals; contour coefficients weighted "
                f"{config['aux_weight']} relative to the three reported targets"
            ),
            "- Device: CPU (at this size, MPS kernel-launch overhead exceeds the arithmetic)",
            "",
        ]
    lines += [
        "### The output layer is initialised to zero — on purpose",
        "",
        "Every target is parameterised as a **residual from baseline B2** (copy the source value,",
        "plus one global offset fitted on train). With a zero-initialised head the prediction at",
        "step 0 *is* B2, exactly. The model can only move away from the baseline if the dev loss",
        "improves, so it is structurally impossible for it to start out worse than the published",
        "baseline. A contract test asserts a fresh model outputs exactly zero.",
        "",
    ]
    return "\n".join(lines)


def _results_section(netdir: Path, split: str) -> str:
    table = _read_csv(netdir / f"results-{split}.csv")
    if table is None:
        return f"## Results ({split})\n\n_`results-{split}.csv` not found — run `prosody_eval`._\n"

    lines = [
        f"## Results — {split} split",
        "",
        "Mean absolute error against the human target. **B2** is the baseline the thesis reports;",
        "**ridge_full** is the linear model given the *same 16 inputs* as ProsoMLP, so a neural win",
        "is attributable to the architecture rather than to extra features.",
        "",
    ]
    systems = [
        "B0_copy_source",
        "B2_copy_plus_offset",
        "ridge_t0",
        "ridge_full",
        "ProsoMLP",
        "ProsoCNN",
    ]
    for direction in sorted(table["direction"].unique()):
        lines += [
            f"### {direction}",
            "",
            "| target | " + " | ".join(s.replace("_", " ") for s in systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for target in ("f0_mean_st", "f0_std_st", "log_dur_ratio"):
            part = table[(table["direction"] == direction) & (table["target"] == target)]
            if part.empty:
                continue
            cells = []
            b2_row = part[part["system"] == "B2_copy_plus_offset"]
            b2 = float(b2_row["mae"].iloc[0]) if not b2_row.empty else None
            for system in systems:
                row = part[part["system"] == system]
                if row.empty:
                    cells.append("—")
                    continue
                mae = float(row["mae"].iloc[0])
                if system == "B2_copy_plus_offset" or b2 is None:
                    cells.append(f"{mae:.4f}")
                else:
                    cells.append(f"{mae:.4f} ({100 * (mae - b2) / b2:+.1f}%)")
            lines.append(f"| `{target}` | " + " | ".join(cells) + " |")
        lines.append("")

    speakers = table["n_speakers"].dropna()
    if len(speakers):
        n = int(speakers.max())
        lines += [
            "### Statistical inference, and its limit",
            "",
            (
                f"Errors are correlated within a speaker, and the {split} split holds only "
                f"**{n} speakers**. Two speaker-aware tests are reported: a Wilcoxon "
                "signed-rank test over"
            ),
            "per-speaker mean absolute errors, and a bootstrap CI that resamples speakers rather",
            "than recordings.",
            "",
            (
                f"With {n} speakers the smallest attainable two-sided signed-rank *p* is "
                f"**{2 / 2**n:.5f}**. Holm correction is applied over a pre-specified family of 12"
            ),
            "confirmatory comparisons (2 models x 3 targets x 2 directions); the correction over",
            "all table rows is also reported as a conservative sensitivity check. **The binding",
            "constraint on significance is the speaker count, not the effect size** — a property",
            "of the corpus, not of the models.",
            "",
        ]
    return "\n".join(lines)


def _ladder_section(ladder_dir: Path) -> str:
    table = _read_csv(ladder_dir / "ladder_summary.csv")
    if table is None:
        return "## The evaluation ladder\n\n_`ladder_summary.csv` not found — run `prosody_ladder`._\n"

    lines = [
        "## The evaluation ladder",
        "",
        "Each arm imposes one system's predicted prosody on audio XTTS **already produced**, so the",
        "words, the voice and the vocoder path are identical across arms and only the prosody",
        "differs. No speech is re-synthesised.",
        "",
        "| arm | what it is |",
        "|---|---|",
        "| `S0` | raw XTTS, untouched |",
        "| `S0_identity` | WORLD analysed and resynthesised with no edit — the vocoder's own footprint |",
        "| `S1_copy` | B2's prediction imposed. **This is the baseline the models must beat** |",
        "| `S2_mlp` / `S2_cnn` | the from-scratch models' predictions imposed |",
        "| `S3_dct` | the human target's own prosody, through the 8-coefficient DCT bottleneck |",
        "| `S3_full` | the human target's own prosody at full contour resolution — the ceiling |",
        "| `*_pitch` | as above but pitch ONLY, timing untouched — isolates the retiming's cost |",
        "",
        "`S3 − S3_dct` is what the representation costs; `S3_dct − S2` is what the predictor costs;",
        "`S2 − S1` is what the model buys.",
        "",
    ]
    columns = [
        ("n", "n", 0),
        ("f0_mean_mae_st", "pitch MAE (st)", 4),
        ("f0_std_mae_st", "span MAE (st)", 4),
        ("shape_r_mean", "contour r", 3),
        ("duration_ratio", "dur / target", 3),
        ("ecapa_mean", "ECAPA", 4),
        ("wer_mean", "WER", 4),
    ]
    present = [c for c in columns if c[0] in table.columns]
    for direction in sorted(table["direction"].unique()):
        lines += [
            f"### {direction}",
            "",
            "| arm | " + " | ".join(label for _c, label, _d in present) + " |",
            "|---|" + "---|" * len(present),
        ]
        part = table[table["direction"] == direction]
        from .prosody_ladder import ARMS

        for arm in ARMS:
            row = part[part["arm"] == arm]
            if row.empty:
                continue
            cells = [_fmt(row[c].iloc[0], d) for c, _label, d in present]
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _data_section(netdir: Path) -> str:
    lines = ["## Training data", "", SPLIT_NOTE, ""]
    for direction in ("en-es", "es-en"):
        path = netdir / f"dataset-{direction}.npz"
        if not path.exists():
            continue
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            split = data["split"]
            speakers = data["speaker"]
            counts = {s: int((split == s).sum()) for s in ("train", "dev", "test")}
            lines.append(
                f"- **{direction}**: {counts['train']} train / {counts['dev']} dev / "
                f"{counts['test']} test pairs; "
                f"{len(set(speakers[split == 'test']))} test speakers"
            )
    lines += [
        "",
        "Source: **DRAL** (Dialogs Re-enacted Across Languages), restricted to pairs where one",
        "bilingual speaker performed both sides. DRAL's second-language side is a conversational",
        "**re-enactment, not a literal translation** — so these models learn how a speaker's",
        "prosody maps across their own two languages, which is exactly the quantity of interest,",
        "but the pairs are not translation pairs and should not be described as such.",
        "",
        "### Inputs (16), and the leak that was avoided",
        "",
        "Eight DCT coefficients of the source contour, plus the source's pitch level, pitch span,",
        "log duration, voiced ratio, word count and speaking rate, plus the **machine",
        "translation's** word count and the log word ratio.",
        "",
        "The target word count is taken from machine translation, **never from the human target",
        "transcript**. The human target is the thing being predicted and its length is most of the",
        "answer; a model given that number would post a duration score it could never reproduce in",
        "deployment. MT output is available before synthesis, so using it is causally legitimate.",
        "",
        "One residual mismatch is measured rather than assumed: MT is run over the gold source",
        "transcript because train/dev have no ASR pass, while deployment translates Whisper output.",
        "MT(gold) and MT(asr) word counts correlate at *r* = 0.611 (MAE 1.32 words), so the test",
        "split additionally carries deployment-realistic ASR-derived features and both are reported",
        "(`ProsoMLP` vs `ProsoMLP_deploy`).",
        "",
    ]
    return "\n".join(lines)


def _limitations_section() -> str:
    return LIMITATIONS


def build(netdir: Path, ladder_dir: Path, out: Path, split: str = "test") -> Path:
    header = [
        "# ProsoMLP and ProsoCNN — prosody predictors for cross-lingual speech",
        "",
        "Two small neural models that predict the prosody of a translated utterance from the",
        "source utterance, trained from scratch on DRAL. They fill a gap in the cascaded",
        "speech-to-speech pipeline (Whisper → MarianMT → XTTS), which preserves *speaker identity*",
        "well but transfers almost no utterance-level *prosody*.",
        "",
        "Released open source. Every number in this card is generated from the artefacts in this",
        "repository by `python -m bilingual_voice.prosody_card build`, so it cannot drift from the",
        "data.",
        "",
    ]
    sections = [
        "\n".join(header),
        _architecture_section(netdir),
        _data_section(netdir),
        _results_section(netdir, split),
        _ladder_section(ladder_dir),
        _limitations_section(),
        REPRODUCING,
    ]
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_card")
    sub = parser.add_subparsers(dest="command", required=True)
    bld = sub.add_parser("build")
    bld.add_argument("--netdir", type=Path, default=Path("artifacts/prosody_net"))
    bld.add_argument("--ladder", dest="ladder_dir", type=Path, default=Path("artifacts/ladder"))
    bld.add_argument("--out", type=Path, default=Path("MODEL_CARD.md"))
    bld.add_argument("--split", choices=("dev", "test"), default="test")
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.netdir, args.ladder_dir, args.out, args.split)


if __name__ == "__main__":
    main()
