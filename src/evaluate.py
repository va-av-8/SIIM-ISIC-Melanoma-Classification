"""Evaluate patient-disjoint OOF predictions without touching training data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
)
from matplotlib.ticker import PercentFormatter

from src.common import config, image_path

LOG = logging.getLogger(__name__)
EXPERIMENT_METRICS = ('roc_auc', 'average_precision', 'tpr_at_fpr_le_0_05')


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
    return evaluate_predictions(df, cfg)


def evaluate_predictions(df: pd.DataFrame, cfg: dict, model_label: str = 'B0') -> dict:
    """The same complete report set for standalone baseline and every pipeline variant."""
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
        tpr_95spec_std=("tpr_at_fpr_le_0_05", "std"),
        mean_prevalence=("prevalence", "mean"),
    ).reset_index()

    report = Path(cfg["data"]["report_dir"])
    report.mkdir(parents=True, exist_ok=True)
    from src.train import save_fold_balance
    save_fold_balance(df.drop_duplicates('image_name'), report)
    per_fold.to_csv(report / "fold_metrics.csv", index=False)
    overview.to_csv(report / "metrics_summary.csv", index=False)
    pd.DataFrame(pooled).to_csv(report / "pooled_oof_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(report / "confusion_threshold_0_5.csv", index=False)
    errors = pd.DataFrame(error_examples, columns=["class_weight", "error", "image_name", "patient_id", "target", "score", "fold"])
    errors.to_csv(report / "error_examples.csv", index=False)
    plot_curves(df, report)
    figure_dir = Path(cfg['data'].get('figure_dir', report.parent / 'figures'))
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Preserve both the original report paths and the README/publication figure paths.
    for name in ('roc_oof.png', 'pr_oof.png'):
        if (report / name).resolve() != (figure_dir / name).resolve():
            shutil.copyfile(report / name, figure_dir / name)
    plot_error_examples(errors.loc[errors.class_weight.eq('none')], cfg, model_label)
    LOG.info("Five-fold mean ± std:\n%s", overview.to_string(index=False))
    resolutions = evaluate_resolutions(cfg, oof=df, model_label=model_label)
    figures = {name: figure_dir / f'{name}.png'
               for name in ('roc_oof', 'pr_oof', 'resolution_analysis', 'hard_errors')}
    return {"fold_metrics": per_fold, "summary": overview,
            "resolution_metrics": resolutions, 'figures': figures}


def draw_error_image(ax, row, images_dir):
    """Original JPEG and the shared HW1 error caption; never impute missing images."""
    from PIL import Image
    path = image_path(images_dir, row.image_name)
    if not path.is_file():
        raise FileNotFoundError(f'Error example JPEG not found: {path}')
    with Image.open(path) as source:
        preview = source.convert('RGB')
        preview.thumbnail((512, 512))
        ax.imshow(preview)
    ax.axis('off')
    ax.set_title(f'{row.error}: {row.image_name}\ntarget={row.target}, score={row.score:.6f}, fold={row.fold}',
                 fontsize=10)


def plot_error_examples(errors: pd.DataFrame, cfg: dict, model_label: str) -> None:
    """Two highest-score FP and two lowest-score FN, using the report's threshold 0.5."""
    report = Path(cfg['data']['report_dir'])
    figure_dir = Path(cfg['data'].get('figure_dir', report.parent / 'figures'))
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    missing_images = 0
    for row_axes, error_type in zip(axes, ('FP', 'FN')):
        rows = errors.loc[errors.error.eq(error_type)].sort_values(
            'score', ascending=error_type == 'FN').head(2).reset_index(drop=True)
        for i, ax in enumerate(row_axes):
            ax.axis('off')
            if i >= len(rows):
                ax.text(.5, .5, f'No further {error_type} at threshold 0.5', ha='center')
                continue
            row = rows.iloc[i]
            path = image_path(cfg['data']['images_dir'], row.image_name) if cfg['data'].get('images_dir') else None
            if path is not None and path.is_file():
                draw_error_image(ax, row, cfg['data']['images_dir'])
            else:
                ax.text(.5, .5, 'JPEG unavailable', ha='center')
                missing_images += 1
            ax.set_title(f'{error_type}: {row.image_name}\ntarget={row.target}, score={row.score:.6f}, fold={row.fold}',
                         fontsize=10)
    fig.suptitle(f'{model_label}: OOF errors at threshold 0.5')
    fig.tight_layout()
    fig.savefig(figure_dir / 'hard_errors.png', dpi=160)
    plt.close(fig)
    if missing_images:
        LOG.warning('%s: %d error JPEGs unavailable; hard_errors.png labels the missing panels',
                    model_label, missing_images)


def merge_b0_resolutions(oof: pd.DataFrame, stats: pd.DataFrame,
                         excluded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Match B0 to original JPEG sizes, accounting only for audited EDA exclusions."""
    for name, frame, required in (
        ("OOF", oof, {"image_name", "target", "score", "class_weight"}),
        ("image_statistics", stats, {"image_name", "width", "height"}),
    ):
        if missing := required - set(frame.columns):
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    # train.py serializes class_weight=None as the literal string 'none'.
    b0 = oof.loc[oof.class_weight.eq("none"), ["image_name", "target", "score"]].copy()
    if b0.empty:
        raise ValueError('No B0 predictions: expected class_weight="none"')
    for name, frame in (("B0 OOF", b0), ("image_statistics", stats)):
        if frame.image_name.isna().any() or frame.image_name.duplicated().any():
            raise ValueError(f"Missing or duplicated image_name in {name}")
    if not b0.target.isin([0, 1]).all():
        raise ValueError("B0 target must be binary (0/1)")
    if not b0.score.between(0, 1).all():
        raise ValueError("Invalid B0 OOF probabilities")
    sizes = stats[["width", "height"]].to_numpy(dtype=float)
    if not (np.isfinite(sizes).all() and (sizes > 0).all()
            and (sizes == np.floor(sizes)).all()):
        raise ValueError("Image width and height must be finite positive integers")

    b0_names, stats_names = set(b0.image_name), set(stats.image_name)
    missing, extra = b0_names - stats_names, stats_names - b0_names
    if missing:
        raise ValueError(f"Image sets do not match: {len(missing)} B0 images lack sizes; "
                         f"examples: {sorted(missing)[:5]}")
    if extra:
        if excluded is None:
            raise ValueError(f"Image sets do not match: {len(extra)} extra EDA images; "
                             "provide saved excluded_images.csv to verify EDA cleaning")
        if ("image_name" not in excluded or excluded.image_name.isna().any()
                or excluded.image_name.duplicated().any()):
            raise ValueError("Missing or duplicated image_name in excluded_images")
        if set(excluded.image_name) != extra:
            raise ValueError("Image sets do not match: extra EDA images differ from excluded_images")
        LOG.info("Verified %d EDA exclusions; no B0 predictions excluded", len(extra))
    matched_stats = stats.loc[stats.image_name.isin(b0_names)].copy()
    if set(matched_stats.image_name) != b0_names:
        raise ValueError("Image sets do not match after EDA exclusions")
    merged = b0.merge(matched_stats, on="image_name", how="left",
                      validate="one_to_one", suffixes=("", "_eda"), indicator=True)
    if len(merged) != len(b0) or not merged["_merge"].eq("both").all():
        raise ValueError("Lost B0 images during resolution merge")
    if "target_eda" in merged and not merged.target.eq(merged.target_eda).all():
        raise ValueError("OOF and image_statistics have different target values")
    merged[["width", "height"]] = merged[["width", "height"]].astype("int64")
    LOG.info("Matched all %d B0 images to original resolutions", len(merged))
    return merged[["image_name", "target", "score", "width", "height"]]


def resolution_metrics(data: pd.DataFrame, top_n: int = 10,
                       min_positive: int = 10) -> pd.DataFrame:
    """Pooled OOF AP for every exact width/height pair; mark fixed plot selection."""
    for name, value in (("top_n", top_n), ("min_positive", min_positive)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    records = []
    for (width, height), group in data.groupby(["width", "height"], sort=True):
        n, positive = len(group), int(group.target.sum())
        prevalence = positive / n
        ap = float(average_precision_score(group.target, group.score)) if positive else np.nan
        records.append(dict(width=width, height=height, resolution=f"{width}×{height}",
                            n=n, n_positive=positive, prevalence=prevalence,
                            average_precision=ap,
                            ap_over_prevalence=ap / prevalence if positive else np.nan))
    metrics = pd.DataFrame(records).sort_values(
        ["n", "width", "height"], ascending=[False, True, True]).reset_index(drop=True)
    selected = metrics.loc[metrics.n_positive >= min_positive].head(top_n).index
    metrics["shown_in_plot"] = metrics.index.isin(selected)
    return metrics


def plot_resolution_analysis(metrics: pd.DataFrame, path: Path,
                             top_n: int, min_positive: int, model_label: str = 'B0') -> None:
    shown = metrics.loc[metrics.shown_in_plot]
    n, positive = int(shown.n.sum()), int(shown.n_positive.sum())
    total, total_positive = int(metrics.n.sum()), int(metrics.n_positive.sum())
    positive_coverage = f"{positive / total_positive:.1%}" if total_positive else "undefined"
    coverage = (f"Coverage: {n:,}/{total:,} images ({n / total:.1%}); "
                f"{positive}/{total_positive} melanomas ({positive_coverage})")
    LOG.info("Plot selection: top %d by n, n_positive >= %d; %d groups. %s",
             top_n, min_positive, len(shown), coverage)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(shown) * 1.15), 8), sharex=True)
    fig.suptitle(f"{model_label}: quality by original image resolution\n"
                 f"Top {top_n} groups by n with n_positive ≥ {min_positive}\n{coverage}",
                 fontsize=12)
    x = np.arange(len(shown))
    axes[0].bar(x - .2, shown.average_precision * 100, width=.4, label="Average Precision (AP)")
    axes[0].bar(x + .2, shown.prevalence * 100, width=.4, label="Prevalence")
    axes[0].set_ylabel("AP and prevalence (%)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].legend()
    axes[1].bar(x, shown.ap_over_prevalence, width=.65, label="AP / prevalence")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1, label="AP = prevalence")
    axes[1].set_ylabel("AP / prevalence (ratio)")
    axes[1].set_xlabel("Original width × height (pixels); group counts")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{r.resolution}\nn={r.n:,}\nn_positive={r.n_positive}" for r in shown.itertuples()],
        rotation=35, ha="right", fontsize=9)
    axes[1].legend()
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        if shown.empty:
            ax.text(.5, .5, "No groups meet the selection criteria",
                    transform=ax.transAxes, ha="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_resolutions(config_path: str, *, oof: pd.DataFrame | None = None,
                         report_dir=None, figure_dir=None, model_label: str = 'B0') -> pd.DataFrame:
    """Analyze saved sizes with standalone OOF or predictions from the experiment pipeline."""
    cfg = config_path if isinstance(config_path, dict) else config(config_path)
    data, options = cfg["data"], cfg.get("resolution_analysis", {})

    def read_saved(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required saved CSV not found: {path}. "
                                    "Restore the HW1 artifact or update the YAML paths; "
                                    "no EDA or training is run automatically.")
        return pd.read_csv(path)

    if oof is None:
        oof = read_saved(Path(data["output_dir"]) / "oof_predictions.csv")
    stats = read_saved(Path(data["eda_dir"]) / "image_statistics.csv")
    excluded = None
    if {"image_name", "class_weight"}.issubset(oof) and "image_name" in stats:
        b0_names = set(oof.loc[oof.class_weight.eq("none"), "image_name"])
        if b0_names and set(stats.image_name) - b0_names:
            excluded = read_saved(Path(data["eda_dir"]) / "excluded_images.csv")
    merged = merge_b0_resolutions(oof, stats, excluded)
    top_n, min_positive = options.get("top_n", 10), options.get("min_positive", 10)
    metrics = resolution_metrics(merged, top_n, min_positive)
    report = Path(report_dir if report_dir is not None else data["report_dir"])
    figure_dir = Path(figure_dir if figure_dir is not None else
                      data.get("figure_dir", report.parent / "figures"))
    report.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report / "resolution_metrics.csv", index=False, na_rep="NaN")
    plot_resolution_analysis(metrics, figure_dir / "resolution_analysis.png", top_n, min_positive, model_label)
    LOG.info("Saved all %d resolution groups to %s", len(metrics), report / "resolution_metrics.csv")
    return metrics


def validate_oof(group, folds, expected_weight='none'):
    """Check a single variant against every saved image, label, patient and fold."""
    required = {'image_name', 'patient_id', 'target', 'fold', 'score', 'class_weight'}
    if missing := required - set(group.columns):
        raise ValueError(f'Missing OOF columns: {sorted(missing)}')
    reference = folds.set_index('image_name').sort_index()
    if (group.image_name.isna().any() or group.image_name.duplicated().any()
            or set(group.image_name) != set(reference.index)):
        raise ValueError('OOF image keys differ from saved folds')
    aligned = group.set_index('image_name').loc[reference.index]
    for column in ('patient_id', 'target', 'fold'):
        if not aligned[column].eq(reference[column]).all():
            raise ValueError(f'OOF {column} differs from saved folds')
    if not group.class_weight.eq(expected_weight).all() or not group.score.between(0, 1).all():
        raise ValueError(f'Expected class_weight={expected_weight} and valid OOF probabilities')


def evaluate_experiments(predictions, folds, report_dir, figure_path):
    """Validate paired OOF coverage, then compare configurations on identical folds."""
    from src.train import variant_spec
    required = {'variant', 'image_name', 'patient_id', 'target', 'fold', 'score', 'class_weight'}
    if missing := required - set(predictions.columns):
        raise ValueError(f'Missing experiment OOF columns: {sorted(missing)}')
    if predictions.variant.isna().any() or 'B0' not in set(predictions.variant):
        raise ValueError('Experiment comparison requires B0 OOF predictions')
    order = list(dict.fromkeys(predictions.variant))
    per_fold = []
    for variant, group in predictions.groupby('variant', sort=False):
        spec = variant_spec(variant)
        expected_weight = spec['class_weight'] or 'none'
        validate_oof(group, folds, expected_weight)
        for fold, part in group.groupby('fold'):
            per_fold.append(dict(stage=spec['stage'], variant=variant, fold=int(fold),
                                 **metric_values(part.target.to_numpy(), part.score.to_numpy())))
    per_fold = pd.DataFrame(per_fold)
    summary = per_fold.groupby('variant', sort=False)[list(EXPERIMENT_METRICS)].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary = summary.reindex(order).reset_index()
    base = per_fold.loc[per_fold.variant.eq('B0'), ['fold', *EXPERIMENT_METRICS]]
    paired = per_fold.merge(base, on='fold', validate='many_to_one', suffixes=('', '_B0'))
    delta_columns = []
    for metric in EXPERIMENT_METRICS:
        column = f'{metric}_delta_vs_B0'
        paired[column] = paired[metric] - paired[f'{metric}_B0']
        delta_columns.append(column)
    deltas = paired[['variant', 'fold', *delta_columns]]
    delta_summary = deltas.groupby('variant', sort=False)[delta_columns].agg(['mean', 'std'])
    delta_summary.columns = ['_'.join(column) for column in delta_summary.columns]
    summary = summary.merge(delta_summary.reset_index(), on='variant', validate='one_to_one')
    summary.insert(0, 'stage', summary.variant.map(lambda name: variant_spec(name)['stage']))
    per_fold = per_fold.merge(deltas, on=['variant', 'fold'], validate='one_to_one')
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (('all_fold_metrics', per_fold), ('all_metrics_summary', summary)):
        table.to_csv(report_dir / f'{name}.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, EXPERIMENT_METRICS,
                                ('ROC-AUC', 'Average Precision', 'Recall @ FPR ≤ 5%')):
        ax.errorbar(np.arange(len(order)), summary[f'{metric}_mean'] * 100,
                    yerr=summary[f'{metric}_std'] * 100, fmt='o', capsize=4,
                    label='Mean ± fold std')
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([f"{name}\n{variant_spec(name)['stage']}" for name in order],
                           rotation=30, ha='right', fontsize=8)
        ax.set(title=title, ylabel='Score (%)')
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    fig.suptitle('Baseline → metadata hypothesis → ablations (same 5 folds)')
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return {'fold_metrics': per_fold, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description="OOF ROC-AUC, AP, error analysis and graphs")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resolution-analysis", action="store_true",
                        help="Only analyze B0 by resolution; by default all OOF reports are generated")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.resolution_analysis:
        evaluate_resolutions(args.config)
    else:
        evaluate(args.config)


if __name__ == "__main__":
    main()
