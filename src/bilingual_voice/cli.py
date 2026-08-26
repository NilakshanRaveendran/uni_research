"""Command-line entry points for dataset preparation and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dral import build_manifest, inspect_dral, merge_transcripts, read_manifest, write_manifest
from .metrics import summarize_rows
from .pipeline import LocalModels, read_results, run_evaluation
from .plots import create_plots
from .splits import assert_speaker_disjoint, assign_speaker_splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bvt")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-dral")
    inspect.add_argument("root", type=Path)

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("root", type=Path)
    manifest.add_argument("output", type=Path)

    split = sub.add_parser("split-manifest")
    split.add_argument("manifest", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--seed", type=int, default=498)

    merge = sub.add_parser("merge-transcripts")
    merge.add_argument("manifest", type=Path)
    merge.add_argument("output", type=Path)
    merge.add_argument("--transcripts", nargs="+", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--split", choices=("train", "dev", "test"), default="test")
    run.add_argument("--limit", type=int)
    run.add_argument(
        "--directions",
        nargs="+",
        choices=("en-es", "es-en"),
        default=["en-es", "es-en"],
    )
    run.add_argument("--output", type=Path, default=Path("results/final_metrics.csv"))
    run.add_argument("--audio-output", type=Path, default=Path("outputs/zero_shot"))
    run.add_argument("--whisper-size", default="small")
    run.add_argument("--mt-device", choices=("cpu", "mps"), default="cpu")
    run.add_argument("--model-root", type=Path, default=Path("models"))
    run.add_argument(
        "--finetuned-mt-root",
        type=Path,
        help="Directory containing best-en-es/ and best-es-en/ checkpoints",
    )
    run.add_argument("--resume", action="store_true")

    summary = sub.add_parser("summarize")
    summary.add_argument("metrics", type=Path)
    summary.add_argument("output", type=Path)

    plots = sub.add_parser("plot-results")
    plots.add_argument("metrics", type=Path)
    plots.add_argument("output_dir", type=Path, default=Path("plots"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "inspect-dral":
        report = inspect_dral(args.root)
        print(
            json.dumps(
                {
                    "root": str(report.root),
                    "metadata_files": [str(path) for path in report.metadata_files],
                    "wav_count": report.wav_count,
                    "english_wav_count": report.english_wav_count,
                    "spanish_wav_count": report.spanish_wav_count,
                },
                indent=2,
            )
        )
    elif args.command == "build-manifest":
        rows = build_manifest(args.root)
        write_manifest(rows, args.output)
        print(f"Wrote {len(rows)} pairs to {args.output}")
    elif args.command == "split-manifest":
        rows = assign_speaker_splits(read_manifest(args.manifest), seed=args.seed)
        assert_speaker_disjoint(rows)
        write_manifest(rows, args.output)
        counts = {
            name: sum(row["split"] == name for row in rows) for name in ("train", "dev", "test")
        }
        print(json.dumps(counts, indent=2))
    elif args.command == "merge-transcripts":
        rows = merge_transcripts(read_manifest(args.manifest), args.transcripts)
        write_manifest(rows, args.output)
        print(f"Wrote {len(rows)} rows to {args.output}")
    elif args.command == "run":
        rows = [row for row in read_manifest(args.manifest) if row.get("split") == args.split]
        if args.limit is not None:
            rows = rows[: args.limit]
        mt_model_dirs = None
        if args.finetuned_mt_root:
            mt_model_dirs = {
                direction: args.finetuned_mt_root / f"best-{direction}"
                for direction in args.directions
            }
            missing = [str(path) for path in mt_model_dirs.values() if not path.is_dir()]
            if missing:
                raise FileNotFoundError(f"Fine-tuned MarianMT checkpoint missing: {missing}")
        models = LocalModels(
            args.whisper_size,
            args.mt_device,
            args.model_root,
            mt_model_dirs=mt_model_dirs,
        )
        results = run_evaluation(
            rows, args.directions, args.output, args.audio_output, models, args.resume
        )
        print(json.dumps(summarize_rows(results), indent=2))
    elif args.command == "summarize":
        summary = summarize_rows(read_results(args.metrics))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    elif args.command == "plot-results":
        written = create_plots(read_results(args.metrics), args.output_dir)
        print(json.dumps([str(path) for path in written], indent=2))


if __name__ == "__main__":
    main()
