"""Score the from-scratch models against every baseline, on identical rows.

    python -m bilingual_voice.prosody_eval report --split dev
    python -m bilingual_voice.prosody_eval report --split test        # once, at the end

Systems compared, all fitted on train and selected on dev:

    B1_train_mean         predict the training mean -- shows how much copying is worth
    B0_copy_source        copy the source utterance's value
    B2_copy_plus_offset    copy, plus one global offset fitted on train   <- THE baseline
    ridge_t0              the thesis ridge: 4 source scalars
    ridge_t1              + word count and speaking rate
    ridge_full            the SAME 16 inputs as ProsoMLP, including the DCT coefficients
    ProsoMLP / ProsoCNN   the from-scratch networks

`ridge_full` exists to close the obvious hole in the comparison. Giving a neural model more input
features than the linear baseline and then claiming an architectural win would be dishonest; this
arm gives the ridge everything the network sees, so any remaining gap is attributable to the model
rather than to the features.

Statistics follow the thesis: per-speaker Wilcoxon and a speaker-clustered bootstrap
(`analysis._cluster_stats`), because the test split holds a few hundred pairs but only ~14
speakers. Holm correction is applied across the whole family of comparisons against B2, and the
family size is printed so the correction can be checked.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import analysis
from .contours import dct_decode
from .prosody_net import (
    DCT_OFFSET,
    INPUT_SCALARS,
    SCALAR_TARGETS,
    load_dataset,
    load_ensemble,
    predict,
    standardise,
)

# The PRE-SPECIFIED confirmatory family: the claims actually being made. Baseline arms are
# descriptive context, not hypotheses, so folding them into the correction would inflate the family
# from 12 to 42 and make significance unreachable -- a -17% effect came out at Holm p=1.0 that way.
# Both corrections are reported: `holm_primary` over these systems, and `holm_all` over every
# comparison as a conservative sensitivity check. This split was fixed on DEV, before test was
# touched.
PRIMARY_SYSTEMS = ("ProsoMLP", "ProsoCNN")

RIDGE_SETS = {
    "ridge_t0": len(analysis.FEATURES_T0),  # 4
    "ridge_t1": len(analysis.FEATURES_T1),  # 6
    "ridge_full": len(INPUT_SCALARS),  # 8, plus the DCT block appended below
}


def _ridge_inputs(data, std, name: str):
    """Feature matrix for one ridge arm. `ridge_full` matches the MLP's input exactly."""
    import numpy as np

    columns = std["x_scalars"][:, : RIDGE_SETS[name]]
    if name == "ridge_full":
        return np.column_stack([columns, std["x_dct"]])
    return columns


def _deploy_view(data, std, mask):
    """A copy of `std` whose scalar features are the ASR-derived ones where available.

    Standardisation reuses the TRAIN mean and standard deviation -- recomputing them on the
    swapped matrix would leak evaluation statistics into the scaling. Rows without deployment
    features keep their gold features so the arrays stay aligned and the comparison stays paired;
    the coverage fraction is returned and printed rather than hidden.
    """
    import numpy as np

    if "deploy_scalars" not in data:
        return None, 0.0
    deploy = np.array(data["deploy_scalars"], dtype=float)
    usable = np.isfinite(deploy).all(axis=1)
    if not usable[mask].any():
        return None, 0.0
    merged = np.where(usable[:, None], deploy, data["scalars"])
    view = dict(std)
    view["x_scalars"] = (merged - std["x_mean"]) / std["x_std"]
    return view, float(usable[mask].mean())


def _holm(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni step-down adjustment, preserving input order and None entries."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adjusted: list[float | None] = list(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    running = 0.0
    for rank, (index, p) in enumerate(indexed):
        value = min(1.0, (total - rank) * p)
        running = max(running, value)  # enforce monotonicity
        adjusted[index] = running
    return adjusted


def evaluate(
    outdir: Path,
    split: str = "dev",
    directions=tuple(analysis.DIRECTIONS),
    arches=("mlp", "cnn"),
) -> dict:
    import numpy as np
    import pandas as pd
    from scipy.stats import wilcoxon

    rows: list[dict] = []
    shape_rows: list[dict] = []

    for direction in directions:
        data = load_dataset(direction, outdir)
        std = standardise(data)
        train = data["split"] == "train"
        dev = data["split"] == "dev"
        evalm = data["split"] == split
        if evalm.sum() < 10:
            print(f"skip {direction}: only {int(evalm.sum())} rows in {split}")
            continue

        # ---- a second feature matrix using deployment-realistic (ASR-derived) text features,
        # so each model is reported both as trained and as it would actually run.
        deploy_std, deploy_coverage = _deploy_view(data, std, evalm)

        # ---- neural ensembles, predicted once and reused for every target
        model_predictions: dict[str, np.ndarray] = {}
        for arch in arches:
            try:
                models, _blob = load_ensemble(arch, direction, outdir)
            except FileNotFoundError:
                print(f"  no checkpoint for {arch}/{direction}; skipping that arm")
                continue
            label = {"mlp": "ProsoMLP", "cnn": "ProsoCNN"}[arch]
            model_predictions[label] = np.mean(
                [predict(model, std, evalm) for model in models], axis=0
            )
            if deploy_std is not None:
                model_predictions[f"{label}_deploy"] = np.mean(
                    [predict(model, deploy_std, evalm) for model in models], axis=0
                )
        if deploy_std is not None:
            print(
                f"  [{direction}] deployment-feature coverage on {split}: "
                f"{deploy_coverage:.1%} of rows"
            )

        raw, copy = data["raw_targets"], data["copy"]

        for i, target in enumerate(SCALAR_TARGETS):
            y_tr, c_tr = raw[train, i], copy[train, i]
            y_dv, c_dv = raw[dev, i], copy[dev, i]
            y_ev, c_ev = raw[evalm, i], copy[evalm, i]

            offset = float(np.mean(y_tr - c_tr))
            predictions = {
                "B1_train_mean": np.full_like(y_ev, float(np.mean(y_tr))),
                "B0_copy_source": c_ev,
                "B2_copy_plus_offset": c_ev + offset,
            }

            # Ridge on the residual from copy-source, lambda by the one-SE rule on dev, exactly
            # as analysis.evaluate does it -- so these numbers are comparable with the thesis.
            for name in RIDGE_SETS:
                features = _ridge_inputs(data, std, name)
                lam = analysis._select_lambda(
                    features[train], y_tr - c_tr, features[dev], y_dv, c_dv
                )
                model = analysis.ridge_fit(features[train], y_tr - c_tr, lam)
                predictions[name] = c_ev + analysis.ridge_predict(model, features[evalm])

            for label, matrix in model_predictions.items():
                predictions[label] = matrix[:, i]

            b2_error = np.abs(predictions["B2_copy_plus_offset"] - y_ev)
            b2_mae = float(np.mean(b2_error))
            speakers = data["speaker"][evalm]

            for system, prediction in predictions.items():
                score = analysis._scores(prediction, y_ev)
                score.update(
                    direction=direction,
                    target=target,
                    system=system,
                    unit=analysis.TARGETS[target][0],
                    split=split,
                )
                if system != "B2_copy_plus_offset":
                    error = np.abs(prediction - y_ev)
                    identical = np.allclose(error, b2_error, rtol=1e-6, atol=1e-9)
                    score["mae_vs_b2_pct"] = (
                        round(100.0 * (score["mae"] - b2_mae) / b2_mae, 2) if b2_mae else None
                    )
                    score["equivalent_to_b2"] = bool(identical)
                    score["wilcoxon_p_vs_b2"] = (
                        1.0 if identical else float(wilcoxon(error, b2_error).pvalue)
                    )
                    score.update(analysis._cluster_stats(error, b2_error, speakers))
                    if identical:
                        # A signed-rank test over near-identical error vectors reports a tiny
                        # p-value for a zero-size effect. Do not let that read as a win.
                        score["wilcoxon_p_vs_b2_by_speaker"] = 1.0
                rows.append(score)

        # ---- contour shape: the part no ridge arm can represent
        true_shape = data["target_shape"][evalm]
        source_shape = data["source_shape"][evalm]
        # DCT_OFFSET, not len(SCALAR_TARGETS): the contour block starts AFTER the injection
        # level column, and slicing from 3 would treat that level as a DCT coefficient.
        n_coef = raw.shape[1] - DCT_OFFSET

        def _shape_r(predicted, truth):
            values = []
            for p, t in zip(predicted, truth, strict=True):
                if np.std(p) > 1e-9 and np.std(t) > 1e-9:
                    values.append(float(np.corrcoef(p, t)[0, 1]))
            return values

        arms = {"B0_copy_source": source_shape}
        for label, matrix in model_predictions.items():
            arms[label] = np.array(
                [dct_decode(matrix[j, DCT_OFFSET:]) for j in range(matrix.shape[0])]
            )
        # B2 for the shape block is copy-source plus the train offset on each coefficient.
        shape_offset = (raw[train, DCT_OFFSET:] - copy[train, DCT_OFFSET:]).mean(axis=0)
        arms["B2_copy_plus_offset"] = np.array(
            [
                dct_decode(copy[j, DCT_OFFSET:] + shape_offset)
                for j in np.flatnonzero(evalm)
            ]
        )
        for system, predicted in arms.items():
            values = _shape_r(predicted, true_shape)
            shape_rows.append(
                {
                    "direction": direction,
                    "split": split,
                    "system": system,
                    "n": len(values),
                    "n_coef": int(n_coef),
                    "shape_r_mean": float(np.mean(values)) if values else None,
                    "shape_r_median": float(np.median(values)) if values else None,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no results produced")

    # Two corrections, both reported. `primary` covers the pre-specified confirmatory family;
    # `all` covers every comparison in the table and is deliberately over-conservative.
    column = "wilcoxon_p_vs_b2_by_speaker"
    table["is_primary"] = table["system"].isin(PRIMARY_SYSTEMS)
    raw = [None if pd.isna(v) else float(v) for v in table[column]]
    table["holm_all"] = _holm(raw)
    primary_raw = [
        raw[i] if table["is_primary"].iloc[i] else None for i in range(len(table))
    ]
    table["holm_primary"] = _holm(primary_raw)
    family = int(table["is_primary"].sum() and pd.Series(primary_raw).notna().sum())
    family_all = int(table[column].notna().sum())

    order = [
        "direction",
        "target",
        "system",
        "split",
        "is_primary",
        "n",
        "mae",
        "medae",
        "pearson_r",
        "mae_vs_b2_pct",
        "n_speakers",
        "wilcoxon_p_vs_b2_by_speaker",
        "holm_primary",
        "holm_all",
        "mae_diff_cluster_ci95_low",
        "mae_diff_cluster_ci95_high",
        "cluster_ci_excludes_zero",
        "equivalent_to_b2",
    ]
    table = table[[c for c in order if c in table.columns]]
    return {
        "table": table,
        "shape": pd.DataFrame(shape_rows),
        "family_size": family,
        "family_size_all": family_all,
    }


def report(outdir: Path, split: str = "dev") -> None:
    import pandas as pd

    result = evaluate(outdir, split)
    table, shape = result["table"], result["shape"]

    outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outdir / f"results-{split}.csv", index=False)
    shape.to_csv(outdir / f"shape-{split}.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(f"\n=== {split.upper()}: mean absolute error vs baseline B2 ===")
        for direction in table["direction"].unique():
            for target in SCALAR_TARGETS:
                part = table[(table["direction"] == direction) & (table["target"] == target)]
                if part.empty:
                    continue
                print(f"\n{direction}  {target}")
                for _, row in part.iterrows():
                    delta = row.get("mae_vs_b2_pct")
                    flag = ""
                    if row["system"] != "B2_copy_plus_offset":
                        raw_p = row.get("wilcoxon_p_vs_b2_by_speaker")
                        holm = (
                            row.get("holm_primary")
                            if row.get("is_primary")
                            else row.get("holm_all")
                        )
                        which = "primary" if row.get("is_primary") else "all"
                        if row.get("equivalent_to_b2"):
                            flag = "= B2"
                        elif raw_p is not None and not pd.isna(raw_p):
                            mark = "*" if (holm is not None and not pd.isna(holm)
                                           and holm < 0.05) else " "
                            ci = "CI excl 0" if row.get("cluster_ci_excludes_zero") else "CI incl 0"
                            flag = (
                                f"p={raw_p:.4f} holm[{which}]="
                                f"{'—' if holm is None or pd.isna(holm) else f'{holm:.4f}'}{mark}"
                                f" {ci}"
                            )
                    print(
                        f"   {row['system']:22s} mae={row['mae']:.4f} "
                        f"{'' if delta is None or pd.isna(delta) else f'{delta:+7.2f}%'}  {flag}"
                    )

        print(f"\n=== {split.upper()}: contour shape correlation with the human target ===")
        for _, row in shape.iterrows():
            print(
                f"   {row['direction']}  {row['system']:22s} "
                f"mean r={row['shape_r_mean']:.4f} median r={row['shape_r_median']:.4f} "
                f"(n={row['n']})"
            )

    print(
        f"\nHolm families: primary={result['family_size']} (pre-specified: "
        f"{', '.join(PRIMARY_SYSTEMS)} x 3 targets x 2 directions); "
        f"all={result['family_size_all']} (sensitivity check)\n"
        f"With {int(table['n_speakers'].dropna().max())} speakers the smallest attainable "
        f"two-sided Wilcoxon p is {2 / 2 ** int(table['n_speakers'].dropna().max()):.5f}; "
        f"the speaker-clustered bootstrap CI is the better-powered instrument at this n."
    )
    print(f"wrote {outdir / f'results-{split}.csv'} and {outdir / f'shape-{split}.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--outdir", type=Path, default=Path("artifacts/prosody_net"))
    rep.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.outdir, args.split)


if __name__ == "__main__":
    main()
