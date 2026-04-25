#!/usr/bin/env python3
"""
evaluate.py — Paired comparison of three model conditions on the manuscript
evaluation set:

    AXIS-MURA-v1B   (LoRA fine-tune of MedGemma 1.5 4B)
    MedGemma 1.5 4B (zero-shot, medical-pretrained)
    Gemma 3 4B      (zero-shot, non-medical baseline)

Each input is a predictions CSV with columns: image, body_part, ground_truth,
prediction (booleans). One row per radiograph. Outputs:

    metrics_overall.json       per-model overall metrics with bootstrap CIs
    metrics_per_anatomy.json   per-model, per-body-part breakdown
    mcnemar.json               pairwise McNemar's tests (Holm-Bonferroni)
    summary.txt                human-readable summary

Run:
    python evaluate.py \\
        --axis-predictions path/to/axis_v1B_predictions.csv \\
        --medgemma-predictions path/to/medgemma_predictions.csv \\
        --gemma3-predictions path/to/gemma3_predictions.csv \\
        --output-dir results/
"""

import argparse
import json
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_predictions_csv(path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load a predictions CSV and normalise its columns."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        logger.error(f"CSV not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, dtype=str)
    logger.info(f"Loaded {len(df)} rows from {path}")

    required = {"image", "body_part", "ground_truth", "prediction"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing columns {missing} in {path}. Found: {list(df.columns)}")
        sys.exit(1)

    bool_map = {"true": True, "false": False, "1": True, "0": False}
    df["ground_truth"] = df["ground_truth"].str.strip().str.lower().map(bool_map)
    df["prediction"] = df["prediction"].str.strip().str.lower().map(bool_map)

    if df["ground_truth"].isna().any():
        logger.error(f"Unparseable ground_truth values in {path}")
        sys.exit(1)
    n_pred_na = df["prediction"].isna().sum()
    if n_pred_na > 0:
        logger.warning(f"Dropping {n_pred_na} rows with missing predictions in {path}")
        df = df.dropna(subset=["prediction"]).copy()

    df["image_key"] = df["image"].apply(_normalise_image_path)
    return df[["image_key", "image", "body_part", "ground_truth", "prediction"]]


def _normalise_image_path(p: str) -> str:
    """Extract MURA-relative portion: e.g. mura/valid/XR_HAND/..."""
    idx = p.find("mura/")
    if idx >= 0:
        return p[idx:]
    return p

# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_models(dfs: dict[str, pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    """Inner-join model DataFrames on image_key. Drop images not common to all."""
    keys_per_model = {name: set(df["image_key"]) for name, df in dfs.items()}
    common = set.intersection(*keys_per_model.values())
    for name, keys in keys_per_model.items():
        diff = keys - common
        if diff:
            logger.warning(f"{name}: {len(diff)} images not in common set (dropped)")

    if not common:
        logger.error("No common images across models")
        sys.exit(1)
    logger.info(f"Common images across all models: {len(common)}")

    first_name = next(iter(dfs))
    merged = dfs[first_name][dfs[first_name]["image_key"].isin(common)].copy()
    merged = merged.rename(columns={"prediction": f"pred_{first_name}"})
    merged = merged[["image_key", "image", "body_part", "ground_truth",
                     f"pred_{first_name}"]]

    for name, df in dfs.items():
        if name == first_name:
            continue
        sub = df[df["image_key"].isin(common)][["image_key", "prediction"]].copy()
        sub = sub.rename(columns={"prediction": f"pred_{name}"})
        merged = merged.merge(sub, on="image_key", how="inner")

    assert len(merged) == len(common), "Merge produced unexpected row count"
    return merged

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_mcc_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    mccs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
            continue
        mccs.append(matthews_corrcoef(yt, yp))
    if len(mccs) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(mccs, 2.5)), float(np.percentile(mccs, 97.5)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0.0)
    mcc = matthews_corrcoef(y_true, y_pred)
    mcc_lo, mcc_hi = bootstrap_mcc_ci(y_true, y_pred)

    return {
        "n": int(n),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy": round(float(acc), 4),
        "accuracy_ci": [round(x, 4) for x in wilson_ci(acc, n)],
        "sensitivity": round(float(sens), 4),
        "sensitivity_ci": [round(x, 4) for x in wilson_ci(sens, tp + fn)],
        "specificity": round(float(spec), 4),
        "specificity_ci": [round(x, 4) for x in wilson_ci(spec, tn + fp)],
        "ppv": round(float(ppv), 4),
        "ppv_ci": [round(x, 4) for x in wilson_ci(ppv, tp + fp)],
        "npv": round(float(npv), 4),
        "npv_ci": [round(x, 4) for x in wilson_ci(npv, tn + fn)],
        "f1": round(float(f1), 4),
        "mcc": round(float(mcc), 4),
        "mcc_ci": [round(mcc_lo, 4), round(mcc_hi, 4)],
    }

# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b_right_a_wrong = int(np.sum(~correct_a & correct_b))
    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    n_discord = b_right_a_wrong + a_right_b_wrong

    if n_discord <= 25:
        from scipy.stats import binomtest
        p_value = binomtest(b_right_a_wrong, n_discord, 0.5).pvalue
        method = "exact"
    else:
        chi2 = (abs(b_right_a_wrong - a_right_b_wrong) - 1) ** 2 / n_discord
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
        method = "chi2_cc"

    return {
        "b_right_a_wrong": b_right_a_wrong,
        "a_right_b_wrong": a_right_b_wrong,
        "n_discordant": n_discord,
        "statistic": round(float((abs(b_right_a_wrong - a_right_b_wrong) - 1) ** 2 / n_discord), 4) if n_discord > 0 else 0.0,
        "p_value": float(p_value),
        "method": method,
    }


def holm_bonferroni(results: list[dict], key: str = "p_value") -> list[dict]:
    n = len(results)
    indexed = sorted(enumerate(results), key=lambda x: x[1][key])
    for rank, (orig_idx, res) in enumerate(indexed):
        adj = min(1.0, res[key] * (n - rank))
        results[orig_idx]["p_adjusted"] = round(adj, 6)
        results[orig_idx]["significant"] = adj < 0.05
    return results

# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

def format_summary(model_names: list[str], overall_metrics: dict,
                   mcnemar_overall: list[dict],
                   anatomy_counts: dict, merged_len: int) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("AXIS Evaluation — Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total images: {merged_len}")
    lines.append("=" * 78)

    lines.append("\nPer-anatomy image counts:")
    for anat, cnt in sorted(anatomy_counts.items()):
        lines.append(f"  {anat}: {cnt}")

    lines.append("\n" + "-" * 78)
    lines.append("Overall Metrics:")
    lines.append(f"{'Model':<14} {'MCC':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7} "
                 f"{'PPV':>7} {'NPV':>7} {'F1':>7}")
    lines.append("-" * 78)
    for name in model_names:
        m = overall_metrics[name]
        lines.append(f"{name:<14} {m['mcc']:>7.4f} {m['accuracy']:>7.4f} "
                     f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f} "
                     f"{m['ppv']:>7.4f} {m['npv']:>7.4f} {m['f1']:>7.4f}")

    lines.append(f"\n{'Model':<14} {'MCC':>7} {'95% CI':>20}")
    lines.append("-" * 45)
    for name in model_names:
        m = overall_metrics[name]
        ci = m["mcc_ci"]
        lines.append(f"{name:<14} {m['mcc']:>7.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    lines.append("\n" + "-" * 78)
    lines.append("McNemar's Tests (overall, Holm-Bonferroni adjusted):")
    lines.append(f"{'Comparison':<30} {'n_discord':>10} {'p-value':>12} "
                 f"{'p-adj':>12} {'Sig':>5}")
    lines.append("-" * 78)
    for r in mcnemar_overall:
        sig = "*" if r.get("significant", False) else ""
        lines.append(f"{r['comparison']:<30} {r['n_discordant']:>10} "
                     f"{r['p_value']:>12.2e} {r['p_adjusted']:>12.2e} {sig:>5}")

    lines.append("\n" + "=" * 78)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison of AXIS, MedGemma, and Gemma 3"
    )
    parser.add_argument("--axis-predictions", required=True,
                        help="Path to AXIS-MURA-v1B predictions CSV")
    parser.add_argument("--medgemma-predictions", required=True,
                        help="Path to MedGemma 1.5 4B predictions CSV")
    parser.add_argument("--gemma3-predictions", required=True,
                        help="Path to Gemma 3 4B predictions CSV")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: ./results/eval_<DATE>)")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        out_dir = Path("results") / f"eval_{date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir / "evaluate.log")
    logger.info(f"Output directory: {out_dir}")

    model_csvs = OrderedDict([
        ("AXIS", args.axis_predictions),
        ("MedGemma", args.medgemma_predictions),
        ("Gemma3", args.gemma3_predictions),
    ])

    dfs = {}
    for name, path in model_csvs.items():
        logger.info(f"Loading {name}: {path}")
        dfs[name] = load_predictions_csv(path, logger)

    merged = align_models(dfs, logger)
    model_names = list(dfs.keys())
    logger.info(f"Aligned {len(merged)} images across {len(model_names)} models")

    y_true = merged["ground_truth"].values.astype(bool)
    anatomy = merged["body_part"].values
    anatomies = sorted(merged["body_part"].unique())
    anatomy_counts = {a: int((anatomy == a).sum()) for a in anatomies}
    logger.info(f"Anatomy counts: {anatomy_counts}")

    # Per-model overall metrics
    overall_metrics = {}
    for name in model_names:
        pred = merged[f"pred_{name}"].values.astype(bool)
        overall_metrics[name] = compute_metrics(y_true, pred)
        logger.info(f"{name}: MCC={overall_metrics[name]['mcc']:.4f} "
                    f"Acc={overall_metrics[name]['accuracy']:.4f}")

    # Per-model per-anatomy metrics
    per_anatomy_metrics = {}
    for name in model_names:
        per_anatomy_metrics[name] = {}
        pred = merged[f"pred_{name}"].values.astype(bool)
        for anat in anatomies:
            mask = anatomy == anat
            per_anatomy_metrics[name][anat] = compute_metrics(y_true[mask], pred[mask])

    # Pairwise McNemar's tests, overall + per-anatomy
    pairs = list(combinations(model_names, 2))
    mcnemar_overall = []
    mcnemar_per_anatomy: dict[str, list[dict]] = {a: [] for a in anatomies}

    for a_name, b_name in pairs:
        pred_a = merged[f"pred_{a_name}"].values.astype(bool)
        pred_b = merged[f"pred_{b_name}"].values.astype(bool)
        comp = f"{a_name} vs {b_name}"

        res = mcnemar_test(y_true, pred_a, pred_b)
        res.update({"comparison": comp, "model_a": a_name, "model_b": b_name})
        mcnemar_overall.append(res)

        for anat in anatomies:
            mask = anatomy == anat
            ra = mcnemar_test(y_true[mask], pred_a[mask], pred_b[mask])
            ra.update({"comparison": comp, "model_a": a_name, "model_b": b_name,
                       "anatomy": anat})
            mcnemar_per_anatomy[anat].append(ra)

    all_tests = list(mcnemar_overall)
    for anat in anatomies:
        all_tests.extend(mcnemar_per_anatomy[anat])
    holm_bonferroni(all_tests)

    # Write outputs
    def write_json(data, filename):
        path = out_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Wrote {path}")

    write_json(overall_metrics, "metrics_overall.json")
    write_json(per_anatomy_metrics, "metrics_per_anatomy.json")
    write_json({"overall": mcnemar_overall, "per_anatomy": mcnemar_per_anatomy},
               "mcnemar.json")

    summary = format_summary(model_names, overall_metrics, mcnemar_overall,
                             anatomy_counts, len(merged))
    (out_dir / "summary.txt").write_text(summary)
    logger.info(f"Wrote {out_dir / 'summary.txt'}")
    print("\n" + summary)
    logger.info("Done.")


if __name__ == "__main__":
    main()
