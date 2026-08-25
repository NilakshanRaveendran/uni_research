"""MarianMT fine-tuning on DRAL, with dev-set checkpoint selection.

    python -m bilingual_voice.finetune train   <manifest.csv> <outdir> --direction en-es
    python -m bilingual_voice.finetune compare <manifest.csv> <outdir> --direction en-es

`train` fine-tunes the pretrained Helsinki-NLP model on the DRAL *train* split, evaluating on
*dev* after every epoch and keeping the checkpoint with the lowest dev loss. `compare` scores the
pretrained and fine-tuned models on the *test* split under identical decoding settings.

Important caveat for the write-up: the DRAL Spanish side is a conversational **re-enactment**, not
a literal translation. Fine-tuning on it therefore adapts the model to DRAL's paraphrase style and
disfluency conventions. A BLEU gain against DRAL references measures domain/style adaptation to the
reference distribution, NOT improved translation adequacy. Report it as such.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .dral import read_manifest

MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
}
MAX_LENGTH = 128
SEED = 498


def build_pairs(rows: list[dict[str, str]], split: str, direction: str) -> list[tuple[str, str]]:
    """(source_text, target_text) pairs for one split and direction. Skips empty either side."""
    source_key, target_key = (
        ("english_text", "spanish_text") if direction == "en-es" else ("spanish_text", "english_text")
    )
    pairs = []
    for row in rows:
        if row.get("split") != split:
            continue
        source = (row.get(source_key) or "").strip()
        target = (row.get(target_key) or "").strip()
        if source and target:
            pairs.append((source, target))
    return pairs


def _pick_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(pairs, size):
    for start in range(0, len(pairs), size):
        yield pairs[start : start + size]


def _encode(tokenizer, batch, device):
    sources = [source for source, _ in batch]
    targets = [target for _, target in batch]
    encoded = tokenizer(
        sources, text_target=targets, return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH,
    )
    labels = encoded["labels"]
    labels[labels == tokenizer.pad_token_id] = -100  # ignore padding in the loss
    encoded["labels"] = labels
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def _dev_loss(model, tokenizer, pairs, device, batch_size) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in _batches(pairs, batch_size):
            loss = model(**_encode(tokenizer, batch, device)).loss
            total += float(loss) * len(batch)
            count += len(batch)
    return total / count if count else float("nan")


def train(
    manifest: Path,
    outdir: Path,
    direction: str = "en-es",
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    device_name: str = "auto",
    model_root: Path = Path("models"),
) -> dict:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    torch.manual_seed(SEED)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    train_pairs = build_pairs(rows, "train", direction)
    dev_pairs = build_pairs(rows, "dev", direction)
    if not train_pairs or not dev_pairs:
        raise ValueError(f"No usable pairs: train={len(train_pairs)} dev={len(dev_pairs)}")
    print(f"[{direction}] train={len(train_pairs)} dev={len(dev_pairs)}", flush=True)

    cache = model_root / "huggingface"
    name = MODELS[direction]
    tokenizer = MarianTokenizer.from_pretrained(name, cache_dir=cache)
    model = MarianMTModel.from_pretrained(name, cache_dir=cache)

    device = _pick_device(device_name)
    try:
        model = model.to(device)
        # Fail fast if this device cannot run a forward+backward at all, rather than dying
        # mid-epoch after minutes of wasted work.
        probe = _encode(tokenizer, train_pairs[: min(2, len(train_pairs))], device)
        model(**probe).loss.backward()
        model.zero_grad(set_to_none=True)
    except Exception as exc:  # noqa: BLE001 - unsupported ops on MPS must not end the run
        print(f"  device {device} failed ({type(exc).__name__}: {exc}); falling back to cpu", flush=True)
        device = torch.device("cpu")
        model = model.to(device)
    print(f"  device={device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    best = {"dev_loss": float("inf"), "epoch": None}
    best_dir = outdir / f"best-{direction}"
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_pairs), generator=torch.Generator().manual_seed(SEED + epoch))
        shuffled = [train_pairs[i] for i in order.tolist()]
        running, seen = 0.0, 0
        for step, batch in enumerate(_batches(shuffled, batch_size), 1):
            loss = model(**_encode(tokenizer, batch, device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss) * len(batch)
            seen += len(batch)
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step} train_loss={running / seen:.4f}", flush=True)
        train_loss = running / seen
        dev_loss = _dev_loss(model, tokenizer, dev_pairs, device, batch_size)
        elapsed = time.time() - started
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss, "elapsed_s": elapsed}
        )
        print(
            f"[{direction}] epoch {epoch}: train={train_loss:.4f} dev={dev_loss:.4f} "
            f"({elapsed / 60:.1f} min)",
            flush=True,
        )
        # Checkpoint selection on DEV, never on test.
        if dev_loss < best["dev_loss"]:
            best = {"dev_loss": dev_loss, "epoch": epoch}
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  new best dev loss -> saved {best_dir}", flush=True)
        (outdir / f"history-{direction}.json").write_text(
            json.dumps({"direction": direction, "history": history, "best": best}, indent=2) + "\n"
        )

    _plot_curves(history, outdir, direction)
    return {"direction": direction, "history": history, "best": best}


def _plot_curves(history: list[dict], outdir: Path, direction: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    axis.plot(epochs, [h["dev_loss"] for h in history], "s-", label="dev")
    best = min(history, key=lambda h: h["dev_loss"])
    axis.axvline(best["epoch"], color="crimson", ls="--", lw=1,
                 label=f"selected epoch {best['epoch']}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(f"MarianMT fine-tuning: {direction}")
    axis.set_xticks(epochs)
    axis.legend()
    axis.grid(alpha=0.25)
    path = outdir / f"F6_finetune_curves_{direction}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _translate_all(model, tokenizer, sources, device, batch_size=16) -> list[str]:
    import torch

    model.eval()
    output: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            chunk = sources[start : start + batch_size]
            encoded = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=MAX_LENGTH, num_beams=4)
            output.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return output


def compare(
    manifest: Path,
    outdir: Path,
    direction: str = "en-es",
    device_name: str = "auto",
    model_root: Path = Path("models"),
) -> dict:
    """Score pretrained vs fine-tuned on the TEST split under identical decoding settings."""
    import sacrebleu
    from transformers import MarianMTModel, MarianTokenizer

    rows = read_manifest(manifest)
    test_pairs = build_pairs(rows, "test", direction)
    sources = [source for source, _ in test_pairs]
    references = [target for _, target in test_pairs]
    print(f"[{direction}] test pairs = {len(test_pairs)}", flush=True)

    device = _pick_device(device_name)
    cache = model_root / "huggingface"
    results = {}
    for label, source_path in (
        ("pretrained", MODELS[direction]),
        ("finetuned", str(outdir / f"best-{direction}")),
    ):
        if label == "finetuned" and not (outdir / f"best-{direction}").exists():
            print("  no fine-tuned checkpoint found; skipping", flush=True)
            continue
        kwargs = {"cache_dir": cache} if label == "pretrained" else {}
        tokenizer = MarianTokenizer.from_pretrained(source_path, **kwargs)
        model = MarianMTModel.from_pretrained(source_path, **kwargs).to(device)
        started = time.time()
        hypotheses = _translate_all(model, tokenizer, sources, device)
        results[label] = {
            "bleu": float(sacrebleu.corpus_bleu(hypotheses, [references]).score),
            "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
            "n": len(hypotheses),
            "decode_s": round(time.time() - started, 1),
        }
        print(f"  {label}: BLEU={results[label]['bleu']:.2f} chrF={results[label]['chrf']:.2f}",
              flush=True)
        (outdir / f"hyps-{label}-{direction}.txt").write_text(
            "\n".join(hypotheses) + "\n", encoding="utf-8"
        )

    if "pretrained" in results and "finetuned" in results:
        results["delta"] = {
            "bleu": round(results["finetuned"]["bleu"] - results["pretrained"]["bleu"], 3),
            "chrf": round(results["finetuned"]["chrf"] - results["pretrained"]["chrf"], 3),
        }
    results["caveat"] = (
        "DRAL Spanish is a conversational re-enactment, not a literal translation. A BLEU gain "
        "here reflects adaptation to the reference style and disfluency conventions, not improved "
        "translation adequacy."
    )
    (outdir / f"comparison-{direction}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.finetune")
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("manifest", type=Path)
    tr.add_argument("outdir", type=Path)
    tr.add_argument("--direction", choices=tuple(MODELS), default="en-es")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=8)
    tr.add_argument("--learning-rate", type=float, default=2e-5)
    tr.add_argument("--device", default="auto")

    cp = sub.add_parser("compare")
    cp.add_argument("manifest", type=Path)
    cp.add_argument("outdir", type=Path)
    cp.add_argument("--direction", choices=tuple(MODELS), default="en-es")
    cp.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.command == "train":
        summary = train(
            args.manifest, args.outdir, args.direction, args.epochs,
            args.batch_size, args.learning_rate, args.device,
        )
        print(json.dumps(summary["best"], indent=2))
    else:
        print(json.dumps(compare(args.manifest, args.outdir, args.direction, args.device), indent=2))


if __name__ == "__main__":
    main()
