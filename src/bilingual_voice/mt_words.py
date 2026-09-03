"""Translate every manifest source text, so the prosody models can use the target word count.

    python -m bilingual_voice.mt_words build data/splits/dral_manifest_with_text.csv artifacts/mt_words.csv
    python -m bilingual_voice.mt_words leakcheck artifacts/mt_words.csv results/synthesis_test.csv

Why this exists, and why it is not optional.

A prosody model that predicts *duration* has to know how much text will be spoken. The obvious
place to get the target word count is the human target transcript -- and that is a LEAK: the human
target is the thing being predicted, and its length is most of the answer. A model fed that number
would post a duration score it could never reproduce in deployment.

The honest source is the machine translation, because in the real pipeline
(audio -> Whisper -> MarianMT -> XTTS) the MT output is what XTTS is about to speak. It exists
before synthesis, so using it is causally legitimate. This module produces it for every split with
the SAME fine-tuned checkpoints and the SAME greedy decoding the pipeline uses
(`pipeline.LocalModels.translate`: `generate(max_new_tokens=512)`, no beam search), so the feature
distribution at training time matches the feature distribution at inference time.

One residual mismatch, stated rather than hidden: MT is run here over the GOLD source transcript,
while the deployed pipeline translates the ASR output (corpus WER 0.150 en, 0.221 es). Source-side
text is an input, not a target, so this is a domain mismatch and not a leak -- and `leakcheck`
measures how big it actually is on the test split, where both texts exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .dral import read_manifest

DIRECTIONS = {"en-es": ("english_text", "es"), "es-en": ("spanish_text", "en")}
MODELS = {"en-es": "Helsinki-NLP/opus-mt-en-es", "es-en": "Helsinki-NLP/opus-mt-es-en"}
FIELDS = ["pair_id", "direction", "split", "source_text", "mt_text", "source_words", "mt_words"]
MAX_NEW_TOKENS = 512  # matches pipeline.LocalModels.translate exactly


def _load(direction: str, model_root: Path, finetuned: Path | None):
    from transformers import MarianMTModel, MarianTokenizer

    checkpoint = finetuned / f"best-{direction}" if finetuned else None
    if checkpoint and (checkpoint / "model.safetensors").is_file():
        return (
            MarianTokenizer.from_pretrained(str(checkpoint)),
            MarianMTModel.from_pretrained(str(checkpoint)),
            f"finetuned:{checkpoint}",
        )
    cache = model_root / "huggingface"
    name = MODELS[direction]
    return (
        MarianTokenizer.from_pretrained(name, cache_dir=cache),
        MarianMTModel.from_pretrained(name, cache_dir=cache),
        f"pretrained:{name}",
    )


def build(
    manifest: Path,
    out_csv: Path,
    model_root: Path = Path("models"),
    finetune_root: Path | None = Path("artifacts/finetune"),
    batch_size: int = 32,
    device_name: str = "auto",
) -> dict:
    """Translate every source text in both directions. Resumable at direction granularity."""
    import torch

    rows = read_manifest(manifest)
    device = torch.device(
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if device_name == "auto"
        else device_name
    )

    done: set[tuple[str, str]] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8-sig") as handle:
            done = {(r["pair_id"], r["direction"]) for r in csv.DictReader(handle)}
        print(f"resuming: {len(done)} rows already present")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    provenance: dict[str, str] = {}
    written = 0

    with out_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for direction, (source_key, _target_lang) in DIRECTIONS.items():
            todo = [
                row
                for row in rows
                if (row["pair_id"], direction) not in done and (row.get(source_key) or "").strip()
            ]
            if not todo:
                print(f"[{direction}] nothing to do")
                continue
            tokenizer, model, source_label = _load(direction, model_root, finetune_root)
            provenance[direction] = source_label
            model = model.to(device).eval()
            print(f"[{direction}] {len(todo)} texts, {source_label}, device={device}", flush=True)

            for start in range(0, len(todo), batch_size):
                chunk = todo[start : start + batch_size]
                texts = [(row[source_key] or "").strip() for row in chunk]
                encoded = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS)
                translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for row, source_text, mt_text in zip(chunk, texts, translations, strict=True):
                    mt_text = mt_text.strip()
                    writer.writerow(
                        {
                            "pair_id": row["pair_id"],
                            "direction": direction,
                            "split": row.get("split", ""),
                            "source_text": source_text,
                            "mt_text": mt_text,
                            "source_words": len(source_text.split()),
                            "mt_words": len(mt_text.split()),
                        }
                    )
                    written += 1
                if (start // batch_size) % 20 == 0:
                    handle.flush()
                    print(f"  {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
            del model

    summary = {"rows_written": written, "output": str(out_csv), "models": provenance}
    (out_csv.with_suffix(".provenance.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {written} rows to {out_csv}")
    return summary


def synthesis_pair_id(row: dict) -> str:
    """`results/synthesis_test.csv` calls the pair identifier `sample_id`; the manifest, the
    contour files and the cached datasets all call it `pair_id`. Reading the wrong one joins zero
    rows and looks like missing data rather than a key mismatch."""
    return (row.get("sample_id") or row.get("pair_id") or "").strip()


def leakcheck(mt_csv: Path, synthesis_csv: Path) -> dict:
    """How far is MT(gold source) from MT(ASR source), and from the human target length?

    Three quantities on the test split:
      * mt_words vs pipeline_words  -- the train/inference domain mismatch this module accepts
      * mt_words vs human_words     -- how much a model would gain from the forbidden feature
    """
    import numpy as np

    mt = {
        (r["pair_id"], r["direction"]): r
        for r in csv.DictReader(mt_csv.open(newline="", encoding="utf-8-sig"))
    }
    rows = list(csv.DictReader(synthesis_csv.open(newline="", encoding="utf-8-sig")))

    ours, pipeline, human = [], [], []
    for row in rows:
        key = (synthesis_pair_id(row), row.get("direction", ""))
        record = mt.get(key)
        if not record:
            continue
        pipeline_text = (row.get("translated_text") or "").strip()
        human_text = (row.get("translation_reference") or "").strip()
        if not pipeline_text or not human_text:
            continue
        ours.append(int(record["mt_words"]))
        pipeline.append(len(pipeline_text.split()))
        human.append(len(human_text.split()))

    if not ours:
        raise ValueError("no overlapping rows between the two files")
    ours, pipeline, human = np.array(ours), np.array(pipeline), np.array(human)
    out = {
        "n": len(ours),
        "mt_vs_pipeline_r": float(np.corrcoef(ours, pipeline)[0, 1]),
        "mt_vs_pipeline_mae_words": float(np.mean(np.abs(ours - pipeline))),
        "mt_vs_human_r": float(np.corrcoef(ours, human)[0, 1]),
        "mt_vs_human_mae_words": float(np.mean(np.abs(ours - human))),
        "pipeline_vs_human_r": float(np.corrcoef(pipeline, human)[0, 1]),
    }
    print(
        f"test rows compared: {out['n']}\n"
        f"  MT(gold) vs MT(asr)   r={out['mt_vs_pipeline_r']:.4f}  "
        f"MAE={out['mt_vs_pipeline_mae_words']:.2f} words   <- accepted domain mismatch\n"
        f"  MT(gold) vs human     r={out['mt_vs_human_r']:.4f}  "
        f"MAE={out['mt_vs_human_mae_words']:.2f} words\n"
        f"  MT(asr)  vs human     r={out['pipeline_vs_human_r']:.4f}"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.mt_words")
    sub = parser.add_subparsers(dest="command", required=True)

    bld = sub.add_parser("build")
    bld.add_argument("manifest", type=Path)
    bld.add_argument("output", type=Path)
    bld.add_argument("--batch-size", type=int, default=32)
    bld.add_argument("--device", default="auto")
    bld.add_argument("--pretrained", action="store_true", help="ignore fine-tuned checkpoints")

    chk = sub.add_parser("leakcheck")
    chk.add_argument("mt_csv", type=Path)
    chk.add_argument("synthesis_csv", type=Path)

    args = parser.parse_args(argv)
    if args.command == "build":
        build(
            args.manifest,
            args.output,
            finetune_root=None if args.pretrained else Path("artifacts/finetune"),
            batch_size=args.batch_size,
            device_name=args.device,
        )
    else:
        print(json.dumps(leakcheck(args.mt_csv, args.synthesis_csv), indent=2))


if __name__ == "__main__":
    main()
