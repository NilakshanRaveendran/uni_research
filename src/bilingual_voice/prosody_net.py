"""Two prosody predictors written from scratch, trained from random initialisation on DRAL.

    python -m bilingual_voice.prosody_net prepare  --direction en-es
    python -m bilingual_voice.prosody_net overfit  --arch mlp --direction en-es
    python -m bilingual_voice.prosody_net train    --arch mlp --direction en-es
    python -m bilingual_voice.prosody_eval report --split dev      # scoring lives in prosody_eval

No pretrained weights are involved anywhere in this file. Both models are a few thousand
parameters and train in minutes on CPU.

WHAT THEY PREDICT
Given only the SOURCE utterance (plus the machine translation's word count, which exists before
synthesis), predict the target-language utterance's pitch level, pitch span, duration ratio, and
the shape of its pitch contour. The first three are exactly the three quantities the thesis ridge
baseline predicts, so the comparison is like-for-like; the contour shape is the part the ridge
cannot represent at all, and is what the injector needs.

WHY THE OUTPUT LAYER IS INITIALISED TO ZERO
Every target is parameterised as a RESIDUAL from baseline B2 (copy the source value, plus one
global offset fitted on train). With a zero-initialised output layer the network's prediction at
step 0 IS B2, exactly. It can only move away from B2 if the dev loss improves. This makes it
structurally impossible for the model to start out worse than the published baseline -- the same
reasoning that fixed the ridge earlier in this project, where fitting the absolute target instead
of the residual made it 34% worse than B2.

HOW THE COMPARISON IS KEPT HONEST
* Baselines are recomputed on the EXACT rows the models see. Contour extraction drops a few rows
  that `analysis.py` kept, so quoting the published ridge numbers against a different row set
  would not be a valid comparison.
* `ridge_full` gives the ridge the SAME 16 inputs as the MLP, including the DCT coefficients and
  the MT word count. If a neural model wins, it wins on architecture, not on extra features.
* The target word count comes from machine translation, never from the human target transcript --
  see the leak discussion in `mt_words.py`.
* Train fits, dev selects, test is touched once.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import analysis
from .contours import N_COEF, N_POINTS, dct_encode, resample_contour

SEEDS = (498, 499, 500, 501, 502)

# The three REPORTED targets, identical to the thesis ridge's, so the table is like-for-like.
SCALAR_TARGETS = ("f0_mean_st", "f0_std_st", "log_dur_ratio")
# A fourth target that exists only to DRIVE the injector: the median of the frame-gated voiced
# pitch. `f0_mean_st` is a mean over every frame pyin accepted inside the 65-1000 Hz search band,
# so a handful of octave-doubled frames can drag it far off -- one XTTS output measured
# f0_std_st = 20.4 semitones, which is not speech. Using that as the injector's reference computes
# a nonsensical shift and then silently hits the level cap. The median over the 60-400 Hz band is
# robust, and predicting it means the shift uses ONE estimator on both sides.
AUX_TARGETS = ("level_med_st",)
LEVEL_MED = len(SCALAR_TARGETS)  # index of level_med_st in the output vector
DCT_OFFSET = len(SCALAR_TARGETS) + len(AUX_TARGETS)
N_OUT = DCT_OFFSET + N_COEF  # 3 reported + 1 injection level + 8 contour coefficients = 12

# The ridge's richest feature set, plus the two length features a duration predictor needs.
INPUT_SCALARS = [*analysis.FEATURES_T1, "mt_words", "log_word_ratio"]
N_SCALARS = len(INPUT_SCALARS)  # 8
N_IN_MLP = N_SCALARS + N_COEF  # 16

DEFAULTS = {
    "lr": 3e-3,
    "weight_decay": 1e-2,
    "batch_size": 64,
    "epochs": 400,
    "patience": 40,
    "dropout": 0.1,
    "hidden": 64,
    "aux_weight": 0.3,  # weight on the 8 contour coefficients relative to the 3 headline targets
}


# --------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------


def _mt_lookup(mt_csv: Path) -> dict[tuple[str, str], int]:
    import csv

    with mt_csv.open(newline="", encoding="utf-8-sig") as handle:
        return {
            (row["pair_id"], row["direction"]): int(row["mt_words"])
            for row in csv.DictReader(handle)
            if row.get("mt_words", "").strip()
        }


def _deploy_lookup(synthesis_csv: Path, direction: str) -> dict[str, tuple[float, float]]:
    """Deployment-realistic word counts for the test split: ASR source, MT of that ASR output.

    The models train on gold-transcript features because train/dev have no ASR pass. In the real
    pipeline both text features come from Whisper instead, and MT(gold) vs MT(asr) correlate at
    only r=0.611 (MAE 1.32 words). That gap is a measurable train/inference mismatch, so the test
    split carries a second copy of the four text-derived features and `prosody_eval` reports the
    models under both -- rather than asserting the mismatch does not matter.
    """
    import csv as _csv

    from .mt_words import synthesis_pair_id

    out: dict[str, tuple[float, float]] = {}
    with synthesis_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in _csv.DictReader(handle):
            if row.get("direction") != direction:
                continue
            asr = (row.get("asr_text") or "").strip()
            mt = (row.get("translated_text") or "").strip()
            if not asr or not mt:
                continue
            out[synthesis_pair_id(row)] = (float(len(asr.split())), float(len(mt.split())))
    return out


def prepare(
    direction: str,
    contours_dir: Path,
    features_csv: Path,
    mt_csv: Path,
    outdir: Path,
    synthesis_csv: Path | None = None,
) -> dict:
    """Join the feature table, the contours and the MT word counts into one cached .npz."""
    import numpy as np

    df, gate_stats = analysis.load_features(features_csv)
    frame = analysis.directional_frame(df, direction)
    mt = _mt_lookup(mt_csv)
    source_side, target_side = analysis.DIRECTIONS[direction]

    needed = INPUT_SCALARS[:-2] + [f"target_{t}" for t in SCALAR_TARGETS] + ["copy_f0_mean_st"]
    frame = frame.dropna(subset=[c for c in needed if c in frame.columns])

    rows, dropped = [], {"no_contour_file": 0, "no_mt_words": 0, "contour_unusable": 0}
    for record in frame.to_dict("records"):
        pair_id = record["pair_id"]
        words = mt.get((pair_id, direction))
        if words is None:
            dropped["no_mt_words"] += 1
            continue
        path = contours_dir / f"{pair_id}.npz"
        if not path.exists():
            dropped["no_contour_file"] += 1
            continue
        with np.load(path) as data:
            source = resample_contour(data[f"{source_side}_f0_hz"], data[f"{source_side}_voiced"])
            target = resample_contour(data[f"{target_side}_f0_hz"], data[f"{target_side}_voiced"])
        if source is None or target is None:
            dropped["contour_unusable"] += 1
            continue
        source_shape, source_level, _sr = source
        target_shape, target_level, _tr = target
        if not (np.isfinite(source_shape).all() and np.isfinite(target_shape).all()):
            dropped["contour_unusable"] += 1
            continue
        record["_s_level_med"] = source_level
        record["_t_level_med"] = target_level
        record["mt_words"] = float(words)
        record["log_word_ratio"] = math.log((words + 1.0) / (record["s_n_words"] + 1.0))
        record["_s_dct"] = dct_encode(source_shape)
        record["_t_dct"] = dct_encode(target_shape)
        record["_s_shape"] = source_shape
        record["_t_shape"] = target_shape
        rows.append(record)

    if not rows:
        raise ValueError(f"no usable rows for {direction}")

    scalars = np.array([[float(r[name]) for name in INPUT_SCALARS] for r in rows])

    deploy = _deploy_lookup(synthesis_csv, direction) if synthesis_csv else {}
    words_i = INPUT_SCALARS.index("s_n_words")
    rate_i = INPUT_SCALARS.index("s_speaking_rate")
    mt_i = INPUT_SCALARS.index("mt_words")
    ratio_i = INPUT_SCALARS.index("log_word_ratio")
    deploy_scalars = np.full_like(scalars, np.nan)
    deploy_hits = 0
    for j, record in enumerate(rows):
        alt = deploy.get(record["pair_id"])
        if alt is None:
            continue
        asr_words, mt_asr_words = alt
        duration = float(record["s_duration"])
        deploy_scalars[j] = scalars[j]
        deploy_scalars[j, words_i] = asr_words
        deploy_scalars[j, rate_i] = asr_words / duration if duration > 0 else np.nan
        deploy_scalars[j, mt_i] = mt_asr_words
        deploy_scalars[j, ratio_i] = math.log((mt_asr_words + 1.0) / (asr_words + 1.0))
        deploy_hits += 1
    source_dct = np.array([r["_s_dct"] for r in rows])
    source_shape = np.array([r["_s_shape"] for r in rows])
    target_shape = np.array([r["_t_shape"] for r in rows])
    target_dct = np.array([r["_t_dct"] for r in rows])

    raw = np.column_stack(
        [np.array([float(r[f"target_{t}"]) for r in rows]) for t in SCALAR_TARGETS]
        + [np.array([float(r["_t_level_med"]) for r in rows])]
        + [target_dct]
    )
    copy = np.column_stack(
        [np.array([float(r[f"copy_{t}"]) for r in rows]) for t in SCALAR_TARGETS]
        + [np.array([float(r["_s_level_med"]) for r in rows])]
        + [source_dct]
    )

    payload = {
        "scalars": scalars,
        "deploy_scalars": deploy_scalars,
        "source_dct": source_dct,
        "source_shape": source_shape,
        "target_shape": target_shape,
        "raw_targets": raw,
        "copy": copy,
        "split": np.array([r["split"] for r in rows]),
        "speaker": np.array([str(r["speaker"]) for r in rows]),
        "pair_id": np.array([r["pair_id"] for r in rows]),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"dataset-{direction}.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    counts = {s: int((payload["split"] == s).sum()) for s in ("train", "dev", "test")}
    summary = {
        "direction": direction,
        "rows": len(rows),
        "by_split": counts,
        "speakers_test": len(set(payload["speaker"][payload["split"] == "test"])),
        "dropped": dropped,
        "deploy_features_available": deploy_hits,
        "quality_gate": gate_stats.get("gate"),
        "output": str(path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def load_dataset(direction: str, outdir: Path):
    import numpy as np

    path = outdir / f"dataset-{direction}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run `prepare --direction {direction}` first")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def standardise(data: dict):
    """Standardise inputs on train; express targets as standardised residuals from B2.

    Returns a dict carrying everything needed to map model outputs back to real units.
    """
    import numpy as np

    train = data["split"] == "train"
    scalars, source_dct = data["scalars"], data["source_dct"]

    x_mean = scalars[train].mean(axis=0)
    x_std = scalars[train].std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    dct_scale = float(source_dct[train].std()) or 1.0

    # B2: copy the source value, plus one global offset fitted on train. Identical construction to
    # analysis.evaluate, so the B2 column here and the published B2 mean the same thing.
    offset = (data["raw_targets"][train] - data["copy"][train]).mean(axis=0)
    b2 = data["copy"] + offset
    residual = data["raw_targets"] - b2
    y_std = residual[train].std(axis=0)
    y_std[y_std < 1e-12] = 1.0

    assert np.abs(residual[train].mean(axis=0)).max() < 1e-8, "B2 residual must be centred on train"

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "x_scalars": (scalars - x_mean) / x_std,
        "x_dct": source_dct / dct_scale,
        "x_shape": data["source_shape"] / dct_scale,
        "y": residual / y_std,
        "b2": b2,
        "y_std": y_std,
        "offset": offset,
        "dct_scale": dct_scale,
    }


# --------------------------------------------------------------------------------------
# The two architectures
# --------------------------------------------------------------------------------------


def build_model(arch: str, hidden: int = 64, dropout: float = 0.1):
    """Return an nn.Module whose output layer is zero-initialised, so it starts as B2."""
    import torch
    from torch import nn

    class ProsoMLP(nn.Module):
        """Model 1: a residual MLP over the source contour's DCT coefficients."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(N_IN_MLP, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(hidden, N_OUT)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, scalars, dct, shape):
            return self.head(self.body(torch.cat([scalars, dct], dim=1)))

    class ProsoCNN(nn.Module):
        """Model 2: a 1-D CNN reading the raw contour, bypassing the DCT compression.

        The point of this arm is to answer a question the MLP cannot: does truncating the contour
        to 8 DCT coefficients throw away information a predictor could have used? A tie is as
        informative as a win -- it says the compression is sufficient.
        """

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(16, 16, kernel_size=5, padding=2),
                nn.GELU(),
            )
            self.body = nn.Sequential(
                nn.Linear(32 + N_SCALARS, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Linear(hidden, N_OUT)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, scalars, dct, shape):
            features = self.conv(shape.unsqueeze(1))
            pooled = torch.cat([features.mean(dim=2), features.amax(dim=2)], dim=1)
            return self.head(self.body(torch.cat([pooled, scalars], dim=1)))

    if arch == "mlp":
        return ProsoMLP()
    if arch == "cnn":
        return ProsoCNN()
    raise ValueError(f"unknown arch {arch!r}; expected 'mlp' or 'cnn'")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _tensors(std: dict, mask, device):
    import torch

    def take(array):
        return torch.tensor(array[mask], dtype=torch.float32, device=device)

    return take(std["x_scalars"]), take(std["x_dct"]), take(std["x_shape"]), take(std["y"])


def _loss_weights(aux_weight: float, device):
    import torch

    weights = [1.0] * (len(SCALAR_TARGETS) + len(AUX_TARGETS)) + [aux_weight] * N_COEF
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _weighted_smooth_l1(prediction, truth, weights):
    from torch.nn import functional

    per_element = functional.smooth_l1_loss(prediction, truth, reduction="none", beta=1.0)
    return (per_element * weights).mean()


def train_one(
    data: dict,
    std: dict,
    arch: str,
    seed: int,
    config: dict,
    verbose: bool = False,
) -> dict:
    """Train one model on train, select the checkpoint on dev. CPU: these models are tiny."""
    import numpy as np
    import torch

    device = torch.device("cpu")  # MPS kernel-launch overhead exceeds the arithmetic at this size
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_mask = data["split"] == "train"
    dev_mask = data["split"] == "dev"
    xs_tr, xd_tr, xc_tr, y_tr = _tensors(std, train_mask, device)
    xs_dv, xd_dv, xc_dv, y_dv = _tensors(std, dev_mask, device)

    model = build_model(arch, config["hidden"], config["dropout"]).to(device)
    weights = _loss_weights(config["aux_weight"], device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    n = len(y_tr)
    generator = torch.Generator().manual_seed(seed)
    best = {"dev_loss": float("inf"), "epoch": 0, "state": None}
    history = []
    since_improved = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, config["batch_size"]):
            index = order[start : start + config["batch_size"]]
            loss = _weighted_smooth_l1(
                model(xs_tr[index], xd_tr[index], xc_tr[index]), y_tr[index], weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            running += float(loss.detach()) * len(index)
        train_loss = running / n

        model.eval()
        with torch.inference_mode():
            dev_loss = float(_weighted_smooth_l1(model(xs_dv, xd_dv, xc_dv), y_dv, weights))
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})

        if dev_loss < best["dev_loss"] - 1e-7:
            best = {
                "dev_loss": dev_loss,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= config["patience"]:
                break
        if verbose and epoch % 25 == 0:
            print(f"    epoch {epoch:3d} train={train_loss:.5f} dev={dev_loss:.5f}", flush=True)

    model.load_state_dict(best["state"])
    return {
        "model": model,
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_dev_loss": best["dev_loss"],
        "epochs_run": len(history),
        "history": history,
        "n_parameters": count_parameters(model),
    }


def predict(model, std: dict, mask):
    """Model output mapped back to real units: B2 plus the de-standardised residual."""
    import torch

    device = torch.device("cpu")
    xs, xd, xc, _y = _tensors(std, mask, device)
    model.eval()
    with torch.inference_mode():
        out = model(xs, xd, xc).cpu().numpy()
    return std["b2"][mask] + out * std["y_std"]


def overfit_gate(direction: str, arch: str, outdir: Path, n_examples: int = 8) -> dict:
    """Can the model drive 8 fixed examples to near-zero loss with dropout off?

    If it cannot, the bug is in the data plumbing, not the hyperparameters. Finding that out now
    costs one minute; finding it out in week 4 costs the presentation.
    """
    import numpy as np
    import torch

    data = load_dataset(direction, outdir)
    std = standardise(data)
    train_index = np.flatnonzero(data["split"] == "train")[:n_examples]
    mask = np.zeros(len(data["split"]), dtype=bool)
    mask[train_index] = True

    torch.manual_seed(498)
    model = build_model(arch, DEFAULTS["hidden"], dropout=0.0)
    xs, xd, xc, y = _tensors(std, mask, torch.device("cpu"))
    weights = _loss_weights(1.0, torch.device("cpu"))
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)

    start_loss = float(_weighted_smooth_l1(model(xs, xd, xc), y, weights).detach())
    for _ in range(1500):
        loss = _weighted_smooth_l1(model(xs, xd, xc), y, weights)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    final = float(loss.detach())

    passed = final < start_loss * 0.02
    print(
        f"overfit gate {arch}/{direction}: {n_examples} examples, "
        f"{count_parameters(model)} params\n"
        f"  loss {start_loss:.6f} -> {final:.8f}  ({'PASS' if passed else 'FAIL'})"
    )
    return {"arch": arch, "start_loss": start_loss, "final_loss": final, "passed": passed}


def train(direction: str, arch: str, outdir: Path, seeds=SEEDS, **overrides) -> dict:
    """Train a seed ensemble. Checkpoints and history land in `outdir`."""
    import numpy as np
    import torch

    config = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    data = load_dataset(direction, outdir)
    std = standardise(data)
    counts = {s: int((data["split"] == s).sum()) for s in ("train", "dev", "test")}
    print(
        f"[{arch}/{direction}] train={counts['train']} dev={counts['dev']} test={counts['test']}",
        flush=True,
    )

    runs = []
    for seed in seeds:
        result = train_one(data, std, arch, seed, config, verbose=(seed == seeds[0]))
        runs.append(result)
        print(
            f"  seed {seed}: dev_loss={result['best_dev_loss']:.5f} "
            f"epoch={result['best_epoch']}/{result['epochs_run']} "
            f"params={result['n_parameters']}",
            flush=True,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / f"{arch}-{direction}.pt"
    torch.save(
        {
            "arch": arch,
            "direction": direction,
            "config": config,
            "state_dicts": [r["model"].state_dict() for r in runs],
            "seeds": list(seeds),
            "input_scalars": INPUT_SCALARS,
            "scalar_targets": list(SCALAR_TARGETS),
            "aux_targets": list(AUX_TARGETS),
            "n_coef": N_COEF,
            "n_points": N_POINTS,
            # Everything needed to map raw features to predictions without the dataset file.
            "norm": {
                "y_std": std["y_std"],
                "offset": std["offset"],
                "dct_scale": std["dct_scale"],
            },
        },
        checkpoint,
    )

    dev_mask = data["split"] == "dev"
    ensemble = np.mean([predict(r["model"], std, dev_mask) for r in runs], axis=0)
    truth = data["raw_targets"][dev_mask]
    b2_dev = std["b2"][dev_mask]
    dev_table = {}
    for i, name in enumerate(SCALAR_TARGETS):
        model_mae = float(np.mean(np.abs(ensemble[:, i] - truth[:, i])))
        b2_mae = float(np.mean(np.abs(b2_dev[:, i] - truth[:, i])))
        dev_table[name] = {
            "model_mae": model_mae,
            "b2_mae": b2_mae,
            "vs_b2_pct": round(100.0 * (model_mae - b2_mae) / b2_mae, 2) if b2_mae else None,
        }

    summary = {
        "arch": arch,
        "direction": direction,
        "config": config,
        "n_parameters": runs[0]["n_parameters"],
        "rows": counts,
        "seeds": [r["seed"] for r in runs],
        "best_dev_loss": [r["best_dev_loss"] for r in runs],
        "best_epoch": [r["best_epoch"] for r in runs],
        "dev_ensemble_vs_b2": dev_table,
        "checkpoint": str(checkpoint),
    }
    (outdir / f"history-{arch}-{direction}.json").write_text(
        json.dumps({"summary": summary, "runs": [r["history"] for r in runs]}, indent=2) + "\n"
    )
    print("\ndev (ensemble) vs B2:")
    for name, row in dev_table.items():
        print(
            f"  {name:14s} model={row['model_mae']:.4f} b2={row['b2_mae']:.4f} "
            f"{row['vs_b2_pct']:+.2f}%"
        )
    return summary


def load_ensemble(arch: str, direction: str, outdir: Path):
    """Rebuild a trained ensemble from its checkpoint."""
    import torch

    blob = torch.load(outdir / f"{arch}-{direction}.pt", map_location="cpu", weights_only=False)
    models = []
    for state in blob["state_dicts"]:
        model = build_model(blob["arch"], blob["config"]["hidden"], blob["config"]["dropout"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, blob


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_net")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--direction", choices=tuple(analysis.DIRECTIONS), default="en-es")
        p.add_argument("--outdir", type=Path, default=Path("artifacts/prosody_net"))

    pre = sub.add_parser("prepare")
    common(pre)
    pre.add_argument("--contours", type=Path, default=Path("artifacts/contours"))
    pre.add_argument("--features", type=Path, default=Path("artifacts/features_wideband.csv"))
    pre.add_argument("--mt", type=Path, default=Path("artifacts/mt_words.csv"))
    pre.add_argument(
        "--synthesis-csv",
        dest="synthesis_csv",
        type=Path,
        default=Path("results/synthesis_test.csv"),
        help="source of deployment-realistic (ASR-derived) test features",
    )

    ovf = sub.add_parser("overfit")
    common(ovf)
    ovf.add_argument("--arch", choices=("mlp", "cnn"), default="mlp")

    trn = sub.add_parser("train")
    common(trn)
    trn.add_argument("--arch", choices=("mlp", "cnn"), default="mlp")
    trn.add_argument("--lr", type=float)
    trn.add_argument("--epochs", type=int)
    trn.add_argument("--patience", type=int)
    trn.add_argument("--aux-weight", dest="aux_weight", type=float)
    trn.add_argument("--dropout", type=float)
    trn.add_argument("--hidden", type=int)
    trn.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))

    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.direction,
            args.contours,
            args.features,
            args.mt,
            args.outdir,
            args.synthesis_csv if args.synthesis_csv and args.synthesis_csv.exists() else None,
        )
    elif args.command == "overfit":
        result = overfit_gate(args.direction, args.arch, args.outdir)
        raise SystemExit(0 if result["passed"] else 1)
    else:
        train(
            args.direction,
            args.arch,
            args.outdir,
            seeds=tuple(args.seeds),
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            aux_weight=args.aux_weight,
            dropout=args.dropout,
            hidden=args.hidden,
        )


if __name__ == "__main__":
    main()
