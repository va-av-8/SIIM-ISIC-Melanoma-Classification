"""M-full error/slice diagnostics and robustness, using the existing HW2 training code."""
import argparse
import hashlib
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import config, image_path, load_data
from src.evaluate import draw_error_image, metric_values, validate_oof
from src.pipeline import (experiment_protocol, load_result, save_run_result,
                          variant_config, wandb_sdk)
from src.train import (experiment_features, fit_oof, generate_folds,
                       load_or_compute_embeddings, make_folds)

LOG = logging.getLogger(__name__)
AGE_GROUPS = ['<40', '40–59', '60–69', '70+', 'Unknown']
MP_GROUPS = ['<1 MP', '1–3 MP', '3–10 MP', '10–20 MP', '≥20 MP']
SEEDS = (42, 17, 123)
FRACTIONS = (.2, .5, 1.)
METRICS = ('roc_auc', 'average_precision')


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required CSV not found: {path}. Restore saved artifacts '
                                'or run --stage all on Kaggle.')
    return pd.read_csv(path)


def check_keys(frame, name):
    if ('image_name' not in frame or frame.image_name.isna().any()
            or frame.image_name.duplicated().any()):
        raise ValueError(f'{name}: missing or duplicated image_name')


def age_groups(age):
    age = pd.to_numeric(age, errors='raise')
    if ((age.dropna() < 0) | ~np.isfinite(age.dropna())).any():
        raise ValueError('Age must be finite, nonnegative or missing')
    groups = pd.cut(age, [-np.inf, 40, 60, 70, np.inf], right=False,
                    labels=AGE_GROUPS[:-1]).astype(object)
    return groups.fillna('Unknown')


def mp_groups(width, height):
    sizes = pd.DataFrame({'width': width, 'height': height}).apply(pd.to_numeric, errors='raise')
    known = sizes.stack()
    if ((known <= 0) | ~np.isfinite(known) | (known != np.floor(known))).any():
        raise ValueError('JPEG dimensions must be positive integers or missing')
    mp = sizes.width * sizes.height / 1_000_000
    return pd.cut(mp, [0, 1, 3, 10, 20, np.inf], right=False,
                  labels=MP_GROUPS).astype(object).fillna('Unknown')


def group_metrics(frame, column, order):
    """Pooled OOF metrics; include empty groups and retain undefined values as NaN."""
    records = []
    for label in order:
        part = frame.loc[frame[column].eq(label)]
        n, positive = len(part), int(part.target.sum())
        prevalence = positive / n if n else np.nan
        ap = average_precision_score(part.target, part.score) if positive else np.nan
        auc = roc_auc_score(part.target, part.score) if part.target.nunique() == 2 else np.nan
        records.append(dict(group=label, n=n, n_positive=positive, prevalence=prevalence,
                            roc_auc=auc, average_precision=ap,
                            ap_over_prevalence=ap / prevalence if positive else np.nan))
    return pd.DataFrame(records)


def slice_frame(oof, data, stats, excluded=None):
    for name, frame in (('OOF', oof), ('clean data', data), ('image_statistics', stats)):
        check_keys(frame, name)
    if set(oof.image_name) != set(data.image_name):
        raise ValueError('OOF and cleaned image sets differ')
    if not {'width', 'height'}.issubset(stats):
        raise ValueError('image_statistics must contain width and height')
    extra = set(stats.image_name) - set(oof.image_name)
    if extra:
        if excluded is None:
            raise ValueError('Extra image_statistics keys require saved excluded_images.csv')
        check_keys(excluded, 'excluded_images')
        if not extra.issubset(set(excluded.image_name)) or set(excluded.image_name) & set(oof.image_name):
            raise ValueError('Extra image_statistics keys do not match EDA exclusions')
    merged = oof.merge(data[['image_name', 'age_approx', 'target']], on='image_name',
                       validate='one_to_one', suffixes=('', '_data'))
    if not merged.target.eq(merged.target_data).all():
        raise ValueError('OOF targets differ from cleaned data')
    columns = ['image_name', 'width', 'height'] + (['target'] if 'target' in stats else [])
    merged = merged.merge(stats[columns], on='image_name', how='left',
                          validate='one_to_one', suffixes=('', '_stats'), indicator=True)
    if 'target_stats' in merged:
        present = merged['_merge'].eq('both')
        if not merged.loc[present, 'target'].eq(merged.loc[present, 'target_stats']).all():
            raise ValueError('OOF targets differ from image_statistics')
    merged['age_group'] = age_groups(merged.age_approx)
    merged['mp_group'] = mp_groups(merged.width, merged.height)
    LOG.info('Unknown JPEG size: %d/%d images', merged.mp_group.eq('Unknown').sum(), len(merged))
    return merged


def plot_slices(table, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(table))
    ax.bar(x - .2, table.average_precision, width=.4, label='AP')
    ax.bar(x + .2, table.prevalence, width=.4, label='Prevalence')
    # Explicit annotations prevent undefined AP from looking like a zero-height bar.
    for i, row in enumerate(table.itertuples()):
        if pd.isna(row.average_precision):
            ax.text(i - .2, .02, 'AP: NaN', rotation=90, ha='center', fontsize=8)
    ax.set_xticks(x, [f'{r.group}\nn={r.n:,}' for r in table.itertuples()])
    ax.set(title=title, ylabel='AP / Prevalence (0–1)', xlabel='Group', ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_slices(oof, data, cfg, report, figures):
    eda = Path(cfg['data']['eda_dir'])
    stats = read_csv(eda / 'image_statistics.csv')
    check_keys(stats, 'image_statistics')
    excluded = read_csv(eda / 'excluded_images.csv') if set(stats.image_name) - set(oof.image_name) else None
    merged = slice_frame(oof, data, stats, excluded)
    orders = [('age', 'age_group', AGE_GROUPS), ('mp', 'mp_group', MP_GROUPS + ['Unknown'])]
    report.mkdir(parents=True, exist_ok=True)
    for name, column, order in orders:
        table = group_metrics(merged, column, order)
        # Always retain Unknown in CSV, including n=0; plot the five specified MP bins.
        table.to_csv(report / f'{name}_slices.csv', index=False, na_rep='NaN')
        shown = table if name == 'age' or merged.mp_group.eq('Unknown').any() else table.iloc[:5]
        plot_slices(shown, figures / f'{name}_slices.png',
                    f'M-full: pooled OOF by {"original age" if name == "age" else "original JPEG megapixels"}')


def select_errors(oof):
    """Up to two most confident FN/FP each, then nearest-threshold errors to total five."""
    errors = oof.loc[oof.target.ne(oof.score.ge(.5).astype(int))].copy()
    errors['error'] = np.where(errors.target.eq(1), 'FN', 'FP')
    errors['distance'] = (errors.score - .5).abs()
    chosen = []
    for kind in ('FN', 'FP'):
        part = errors.loc[errors.error.eq(kind)].sort_values(
            ['score', 'image_name'], ascending=[kind == 'FN', True]).head(2)
        chosen.extend(part.image_name)
    rest = errors.loc[~errors.image_name.isin(chosen)].sort_values(['distance', 'image_name'])
    chosen.extend(rest.head(5 - len(chosen)).image_name)
    return errors.set_index('image_name', drop=False).loc[chosen].reset_index(drop=True)


def plot_errors(oof, cfg, path):
    selected = select_errors(oof)
    if selected.empty:
        raise ValueError('No M-full OOF errors at threshold 0.5; no error PNG can be created')
    # Fail before writing any PNG if a selected original is unavailable; no placeholders.
    for name in selected.image_name:
        jpeg = image_path(cfg['data']['images_dir'], name)
        if not jpeg.is_file():
            raise FileNotFoundError(f'Error example JPEG not found: {jpeg}')
    fp = int(((oof.target == 0) & (oof.score >= .5)).sum())
    fn = int(((oof.target == 1) & (oof.score < .5)).sum())
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 5), squeeze=False)
    try:
        for ax, (_, row) in zip(axes[0], selected.iterrows()):
            draw_error_image(ax, row, cfg['data']['images_dir'])
        fig.suptitle(f'M-full: {len(selected)}/5 examples; available FP={fp}, FN={fn}\n'
                     'Threshold 0.5 is illustrative, not clinically validated')
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    LOG.info('Selected %d errors: %s; available FP=%d, FN=%d', len(selected),
             selected.error.value_counts().to_dict(), fp, fn)


def nested_train_masks(folds, fold, seed=42):
    """Shuffle sorted patients within two strata; prefixes yield nested whole-patient sets."""
    train = folds.loc[folds.fold.ne(fold)]
    strata = train.groupby('patient_id', sort=True).target.max()
    if set(strata.unique()) != {0, 1}:
        raise ValueError(f'Fold {fold}: both patient strata are required in train')
    rng = np.random.default_rng(seed)
    ordered = [rng.permutation(strata.index[strata.eq(label)].to_numpy()) for label in (0, 1)]
    masks = {}
    for fraction in FRACTIONS:
        patients = np.concatenate([values[:max(1, int(np.floor(len(values) * fraction)))]
                                   for values in ordered])
        masks[fraction] = folds.patient_id.isin(patients).to_numpy()
    return masks


def training_counts(data, mask):
    part = data.loc[mask]
    patients = part.groupby('patient_id').target.max()
    ages = age_groups(part.age_approx).value_counts()
    return dict(train_n_patients=len(patients), train_n_images=len(part),
                train_n_positive=int(part.target.sum()), train_prevalence=float(part.target.mean()),
                train_positive_patients=int(patients.eq(1).sum()),
                train_negative_patients=int(patients.eq(0).sum()),
                **{f'train_age_{key}': int(ages.get(label, 0)) for key, label in
                   zip(('lt40', '40_59', '60_69', '70plus', 'unknown'), AGE_GROUPS)})


def robustness_table(records, factor):
    folds = pd.DataFrame(records)
    folds.insert(0, 'row_type', 'fold')
    rows = []
    values = [col for col in folds if col not in ('row_type', 'fold', factor)]
    for value, part in folds.groupby(factor, sort=False):
        for kind in ('mean', 'std'):
            stats = part[values].mean() if kind == 'mean' else part[values].std(ddof=1)
            rows.append(dict(row_type=kind, **{factor: value}, fold=np.nan, **stats.to_dict()))
    return pd.concat([folds, pd.DataFrame(rows)], ignore_index=True)


def plot_robustness(table, factor, order, path):
    mean = table.loc[table.row_type.eq('mean')].set_index(factor).loc[list(order)]
    std = table.loc[table.row_type.eq('std')].set_index(factor).loc[list(order)]
    seed_test = factor == 'split_seed'
    labels = [str(v) if seed_test else f'{v:.0%}' for v in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, METRICS, ('ROC-AUC', 'Average Precision')):
        ax.errorbar(np.arange(len(order)), mean[metric], yerr=std[metric],
                    fmt='o' if seed_test else 'o-', capsize=4, label='Mean ± sample std (5 folds)')
        ax.set_xticks(np.arange(len(order)), labels)
        ax.set(title=title, xlabel='Split seed' if seed_test else 'Fraction of train patients',
               ylabel='Score')
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('M-full: split robustness' if seed_test else 'M-full: fixed-validation train-size robustness')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def log_robustness(cfg, protocol, tables, figures):
    sdk = wandb_sdk(cfg)
    if sdk is None:
        return
    options = cfg['wandb']
    output = Path(variant_config(cfg, 'M-full')['data']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        factor = 'split_seed' if name == 'seed_robustness' else 'train_fraction'
        run = sdk.init(project=options.get('project', 'siim-isic-melanoma'),
                       entity=options.get('entity'), mode=options.get('mode', 'online'),
                       group='m_full_robustness', name=f'M-full-{name}', job_type='robustness',
                       dir=str(output), config={**protocol, 'variant': 'M-full',
                       'factor': factor, 'split_seeds': list(SEEDS), 'train_fractions': list(FRACTIONS),
                       'subsample_seed': cfg['validation']['seed'],
                       'features': ['ResNet-18 embeddings (512)', 'age_approx', 'sex',
                                    'anatom_site_general_challenge']})
        success = False
        try:
            for row in table.loc[table.row_type.eq('fold')].to_dict('records'):
                run.log({key: value for key, value in row.items() if key != 'row_type'})
            for row in table.loc[table.row_type.ne('fold')].to_dict('records'):
                for metric in METRICS:
                    run.summary[f'{factor}/{row[factor]:g}/{metric}_{row["row_type"]}'] = row[metric]
            run.log({'metrics': sdk.Table(dataframe=table),
                     'comparison': sdk.Image(str(figures / f'{name}.png'))})
            success = True
        finally:
            run.finish(exit_code=0 if success else 1)


def run_robustness(data, original, features, cfg, report, figures, protocol):
    frame, transform = experiment_features(features, data, 'M-full')
    seed_records, size_records = [], []

    def evaluate(folds, factor, value, masks=None):
        LOG.info('M-full robustness: %s=%s', factor, value)
        oof = fit_oof(folds, frame, cfg, preprocessor=transform, train_masks=masks)
        records = []
        for fold, part in oof.groupby('fold'):
            metrics = metric_values(part.target.to_numpy(), part.score.to_numpy())
            row = {factor: value, 'fold': fold, **{key: metrics[key] for key in METRICS}}
            if masks is not None:
                row.update(training_counts(data, masks[fold]))
            records.append(row)
        return records

    for seed in SEEDS:
        folds = generate_folds(data, cfg, split_seed=seed)
        if seed == 42 and not folds.equals(original):
            raise ValueError('Seed=42 generated folds differ from saved HW2 folds')
        seed_records.extend(evaluate(folds, 'split_seed', seed))
    subsets = {fold: nested_train_masks(original, fold, cfg['validation']['seed'])
               for fold in sorted(original.fold.unique())}
    for fraction in FRACTIONS:
        masks = {fold: subset[fraction] for fold, subset in subsets.items()}
        size_records.extend(evaluate(original, 'train_fraction', fraction, masks))
    tables = {'seed_robustness': robustness_table(seed_records, 'split_seed'),
              'train_size_robustness': robustness_table(size_records, 'train_fraction')}
    report.mkdir(parents=True, exist_ok=True)
    for name, factor, order in (('seed_robustness', 'split_seed', SEEDS),
                                ('train_size_robustness', 'train_fraction', FRACTIONS)):
        tables[name].to_csv(report / f'{name}.csv', index=False, na_rep='NaN')
        plot_robustness(tables[name], factor, order, figures / f'{name}.png')
    log_robustness(cfg, protocol, tables, figures)
    return tables


def run(config_path, stage='all'):
    if stage not in ('all', 'errors', 'slices', 'robustness'):
        raise ValueError(f'Unknown diagnostics stage: {stage}')
    cfg = config(config_path)
    if cfg['validation']['seed'] != 42 or cfg['validation']['n_splits'] != 5:
        raise ValueError('HW3 requires the original five HW2 folds with seed=42')
    heavy = stage in ('all', 'robustness')
    if heavy:
        wandb_sdk(cfg)  # Fail early if enabled but not installed.
    clean = Path(cfg['data']['clean_csv'])
    if not clean.is_file() and stage == 'all':
        from src.eda import run as run_eda
        run_eda(config_path)
    if not clean.is_file():
        raise FileNotFoundError(f'Cleaned data not found: {clean}. Run --stage all on Kaggle.')
    data = load_data(clean)
    if not Path(cfg['data']['splits_csv']).is_file() and not heavy:
        raise FileNotFoundError(f'Saved HW2 folds not found: {cfg["data"]["splits_csv"]}')
    folds = make_folds(data, cfg)
    if not generate_folds(data, cfg).equals(folds):
        raise ValueError('Seed=42 generated folds differ from saved HW2 folds; refusing to overwrite')
    protocol = experiment_protocol(data, folds, cfg)
    run_cfg = variant_config(cfg, 'M-full')
    report, figures = Path(run_cfg['data']['report_dir']), Path(run_cfg['data']['figure_dir'])
    features = None
    if heavy:
        # Exactly one shared extraction/cache load, outside all seed/fraction loops.
        features = load_or_compute_embeddings(folds, cfg)
        feature_hash = hashlib.sha256(features.tobytes()).hexdigest()
    if stage != 'robustness':
        oof_path = Path(run_cfg['data']['output_dir']) / 'oof_predictions.csv'
        if not oof_path.is_file() and stage == 'all':
            frame, transform = experiment_features(features, data, 'M-full')
            oof = fit_oof(folds, frame, cfg, output=run_cfg['data']['output_dir'], preprocessor=transform)
            save_run_result(oof, run_cfg, ['M-full'], protocol, feature_hash)
        oof, saved = load_result(cfg, 'M-full', protocol, check_environment=heavy)
        validate_oof(oof, folds)
        if heavy and saved['embeddings_sha256'] != feature_hash:
            raise ValueError('M-full OOF used different embeddings; restore matching artifacts or '
                             'move the old M-full outputs aside before running --stage all')
        if stage in ('all', 'errors'):
            plot_errors(oof, cfg, figures / 'error_analysis.png')
        if stage in ('all', 'slices'):
            run_slices(oof, data, cfg, report, figures)
    if heavy:
        protocol['embeddings_sha256'] = feature_hash
        return run_robustness(data, folds, features, cfg, report, figures, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--stage', choices=['all', 'errors', 'slices', 'robustness'], default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run(args.config, args.stage)


if __name__ == '__main__':
    main()
