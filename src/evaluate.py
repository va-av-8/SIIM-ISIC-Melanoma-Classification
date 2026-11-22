"""Evaluate patient-disjoint OOF predictions without touching training data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)

from src.common import config

LOG = logging.getLogger(__name__)


def metric_values(y: np.ndarray, score: np.ndarray) -> dict:
    if len(np.unique(y)) != 2:
        raise ValueError("ROC-AUC/AP need both classes in every validation fold")
    fpr, tpr, _ = roc_curve(y, score)
    # TPR @ FPR <= 0.05 is a diagnostic point on validation ROC curve;
    # it is NOT an independently validated deployment threshold.
    tpr_95spec = float(np.max(tpr[fpr <= .05]))
    return {
        "n_images": len(y),
        "n_positive": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "tpr_at_fpr_le_0_05": tpr_95spec,
    }


def plot_curves(data: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for weight, group in data.groupby("class_weight"):
        fpr, tpr, _ = roc_curve(group["target"], group["score"])
        ax.plot(fpr, tpr, label=f"{weight} (AUC={roc_auc_score(group.target, group.score):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="OOF ROC (all 5 folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_oof.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for weight, group in data.groupby("class_weight"):
        precision, recall, _ = precision_recall_curve(group["target"], group["score"])
        ap = average_precision_score(group["target"], group["score"])
        ax.plot(recall, precision, label=f"{weight} (AP={ap:.3f})")
    ax.axhline(float(data.groupby("class_weight")["target"].mean().iloc[0]),
               linestyle="--", linewidth=1, label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="OOF Precision–Recall (all 5 folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr_oof.png", dpi=140)
    plt.close(fig)


def evaluate(config_path: str) -> dict:
    cfg = config(config_path)
    path = Path(cfg["data"]["output_dir"]) / "oof_predictions.csv"
    df = pd.read_csv(path)
    required = {"image_name", "patient_id", "target", "fold", "score", "class_weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing OOF columns: {required - set(df.columns)}")
    if df.duplicated(["image_name", "class_weight"]).any():
        raise ValueError("Each image must have one OOF prediction per class_weight")
    if df["score"].isna().any() or not df["score"].between(0, 1).all():
        raise ValueError("Invalid OOF probabilities")
    per_fold = []
    pooled = []
    error_examples = []
    confusion = []
    for weight, group in df.groupby("class_weight", sort=True):
        for fold, fold_data in group.groupby("fold"):
            record = metric_values(fold_data["target"].to_numpy(), fold_data["score"].to_numpy())
            per_fold.append({"class_weight": weight, "fold": int(fold), **record})
        summary = metric_values(group["target"].to_numpy(), group["score"].to_numpy())
        pooled.append({"class_weight": weight, **summary})
        confusion_matrix_values = confusion_matrix(group.target, (group.score >= .5).astype(int), labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix_values.ravel()
        confusion.append({"class_weight": weight, "threshold": 0.5,
                          "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
        false_positives = group[(group.target == 0) & (group.score >= .5)].nlargest(3, "score")
        false_negatives = group[(group.target == 1) & (group.score < .5)].nsmallest(3, "score")
        for error_type, sample in (("FP", false_positives), ("FN", false_negatives)):
            for row in sample.itertuples():
                error_examples.append({"class_weight": weight, "error": error_type,
                                       "image_name": row.image_name, "patient_id": row.patient_id,
                                       "target": row.target, "score": row.score, "fold": row.fold})
    per_fold = pd.DataFrame(per_fold)
    overview = per_fold.groupby("class_weight").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"),
        tpr_95spec_mean=("tpr_at_fpr_le_0_05", "mean"),
        mean_prevalence=("prevalence", "mean"),
    ).reset_index()

    report = Path(cfg["data"]["report_dir"])
    report.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(report / "fold_metrics.csv", index=False)
    overview.to_csv(report / "metrics_summary.csv", index=False)
    pd.DataFrame(pooled).to_csv(report / "pooled_oof_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(report / "confusion_threshold_0_5.csv", index=False)
    pd.DataFrame(error_examples, columns=["class_weight", "error", "image_name", "patient_id", "target", "score", "fold"]).to_csv(report / "error_examples.csv", index=False)
    plot_curves(df, report)
    LOG.info("Five-fold mean ± std:\n%s", overview.to_string(index=False))
    return {"fold_metrics": per_fold, "summary": overview}


def main():
    parser = argparse.ArgumentParser(description="OOF ROC-AUC, AP, error analysis and graphs")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    evaluate(args.config)


if __name__ == "__main__":
    main()
