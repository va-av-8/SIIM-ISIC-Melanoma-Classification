"""One experiment: baseline, metadata hypothesis, ablations and optional W&B."""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import logging
from pathlib import Path

import pandas as pd

from src.common import config, load_data
from src.evaluate import evaluate_experiments, evaluate_predictions
from src.train import (METADATA, VARIANTS, embedding_signature, experiment_features,
                       fit_oof, load_or_compute_embeddings, make_folds, make_logreg,
                       save_fold_balance, variant_spec)

LOG = logging.getLogger(__name__)
HYPOTHESIS = ('Age, sex and anatomical site contain additional information about melanoma '
              'risk and may improve ROC-AUC/AP over frozen image-only embeddings.')
BASELINE_HYPOTHESIS = ('Frozen image embeddings predict melanoma; compare unweighted '
                       'LogisticRegression (B0) with class_weight=balanced (B1).')


def experiment_protocol(data, folds, cfg):
    # Labels, metadata and fold assignments are fingerprinted, never sent as rows to W&B.
    columns = ['image_name', 'patient_id', 'target', *METADATA]
    source = data.reindex(columns=columns).copy()
    source['fold'] = folds.fold.to_numpy()
    return {
        'schema_version': 2,
        'dataset_sha256': hashlib.sha256(source.to_csv(index=False).encode()).hexdigest(),
        'folds_sha256': hashlib.sha256(folds.to_csv(index=False).encode()).hexdigest(),
        'validation': {**cfg['validation'], 'splitter': 'StratifiedGroupKFold',
                       'group_column': 'patient_id'},
        'logreg': make_logreg(cfg).get_params(),
        'image_extractor': embedding_signature(cfg),
        'libraries': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn')},
        'metadata_preprocessing': 'train-only: median age (empty=0), StandardScaler; '
                                  'categorical MISSING, OneHotEncoder(handle_unknown=ignore)',
    }


def wandb_sdk(cfg):
    options = cfg.get('wandb', {})
    if not options.get('enabled', False):
        return None
    if options.get('mode', 'online') not in ('online', 'offline'):
        raise ValueError('wandb.mode must be online or offline; use enabled: false to disable')
    try:
        return importlib.import_module('wandb')
    except ImportError as exc:
        raise ImportError('W&B is enabled: install wandb or set wandb.enabled: false') from exc


class ExperimentLogger:
    """One configuration per run; lazy optional SDK, with fold and future epoch axes."""
    def __init__(self, cfg, variant, protocol, output_dir):
        self.run = None
        self.sdk = wandb_sdk(cfg)
        options = cfg.get('wandb', {})
        if self.sdk is None:
            return
        spec = variant_spec(variant)
        payload = {**protocol, 'variant': variant, 'stage': spec['stage'],
                   'hypothesis': BASELINE_HYPOTHESIS if spec['stage'] == 'baseline' else HYPOTHESIS,
                   'logreg': {**protocol.get('logreg', {}), 'class_weight': spec['class_weight']},
                   'features': ['ResNet-18 embeddings (512)', *VARIANTS[variant]]}
        self.run = self.sdk.init(
            project=options.get('project', 'siim-isic-melanoma'), entity=options.get('entity'),
            group=options.get('group', 'metadata_experiment'), name=variant,
            mode=options.get('mode', 'online'), dir=str(output_dir), config=payload,
            job_type=spec['stage'])
        self.run.define_metric('fold')
        self.run.define_metric('fold/*', step_metric='fold')

    def log(self, values):
        if self.run is not None:
            self.run.log(values)

    def log_epoch(self, fold, epoch, metrics):
        """Future fine-tuning hook: independent epoch axis for each fold, no fake LR epochs."""
        if self.run is not None:
            prefix = f'epoch/fold_{fold}'
            self.run.define_metric(f'{prefix}/*', step_metric=f'{prefix}/epoch')
            self.log({f'{prefix}/epoch': epoch,
                      **{f'{prefix}/{key}': value for key, value in metrics.items()}})

    def log_results(self, variant, results, figure_path):
        if self.run is None:
            return
        rows = results['fold_metrics'].query('variant == @variant')
        for row in rows.to_dict('records'):
            self.log({'fold': row['fold'], **{f'fold/{key}': value for key, value in row.items()
                                             if key not in ('fold', 'variant', 'stage')}})
        summary = results['summary'].set_index('variant').loc[variant].drop('stage').to_dict()
        self.run.summary.update(summary)
        figures = {'comparison': self.sdk.Image(str(figure_path))}
        for name, path in results['figures'][variant].items():
            # Baseline's error collage/resolution analysis describe its unweighted B0.
            if variant != 'B1' or name in ('roc_oof', 'pr_oof'):
                figures[name] = self.sdk.Image(str(path))
        self.log({**figures, 'fold_metrics': self.sdk.Table(dataframe=rows),
                  'experiments_summary': self.sdk.Table(dataframe=results['summary'])})

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def variant_config(cfg, variant):
    """Keep baseline paths intact; other variants use sibling directories."""
    derived = {**cfg, 'data': dict(cfg['data'])}
    data = derived['data']
    figure_root = Path(data.get('figure_dir', Path(data['report_dir']).parent / 'figures'))
    data['figure_dir'] = str(figure_root)
    if variant not in ('B0', 'B1', 'baseline'):
        data['output_dir'] = str(Path(data['output_dir']).parent / variant)
        data['report_dir'] = str(Path(data['report_dir']).parent / variant)
        data['figure_dir'] = str(figure_root / variant)
    return derived


def load_result(cfg, group, protocol, check_environment=True):
    output = Path(variant_config(cfg, group)['data']['output_dir'])
    path = output / 'oof_predictions.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing OOF predictions: {path}. Train {group} first.')
    snapshot = output / 'run_config.json'
    saved = {}
    if snapshot.is_file():
        saved = json.loads(snapshot.read_text())
        expected = json.loads(json.dumps(protocol))
        stored = json.loads(json.dumps(saved.get('protocol', {})))
        if not check_environment:
            # CSV-only diagnostics need no matching extraction/runtime installation.
            # Data, folds, model parameters and preprocessing are still checked.
            for item in (expected, stored):
                item.pop('libraries', None)
                item.get('image_extractor', {}).pop('libraries', None)
        if stored != expected:
            raise ValueError(f'{group}: saved run does not match current data/folds/config')
        if saved.get('oof_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'{group}: OOF file differs from its saved run configuration')
    elif group != 'baseline':
        raise FileNotFoundError(f'Missing run configuration: {snapshot}. Retrain {group}.')
    else:
        LOG.warning('Legacy baseline has no run_config.json; checking OOF against current folds, '
                    'but original training parameters cannot be verified')
    predictions = pd.read_csv(path)
    if 'class_weight' not in predictions or predictions.empty:
        raise ValueError(f'{group}: missing class_weight or empty OOF')
    if group == 'baseline':
        predictions['variant'] = predictions.class_weight.map({'none': 'B0', 'balanced': 'B1'})
    else:
        predictions['variant'] = group
    if predictions.variant.isna().any():
        raise ValueError(f'{group}: unexpected class_weight in OOF')
    if saved and saved.get('variants') != {v: variant_spec(v) for v in predictions.variant.unique()}:
        raise ValueError(f'{group}: saved variant definitions differ from current configuration')
    return predictions, saved


def save_run_result(predictions, cfg, members, protocol, feature_hash):
    """Persist OOF and its provenance in the existing HW2 format."""
    output = Path(cfg['data']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'oof_predictions.csv'
    predictions.to_csv(path, index=False)
    saved = dict(protocol=protocol, variants={v: variant_spec(v) for v in members},
                 embeddings_sha256=feature_hash,
                 oof_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / 'run_config.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')


def run(config_path, variants=None, stage='all', recompute_embeddings=False):
    cfg = config(config_path)
    options = cfg.get('experiments', {})
    selected = list(variants if variants is not None else options.get('variants', VARIANTS))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(VARIANTS):
        raise ValueError(f'Choose unique variants from {list(VARIANTS)}')
    # The original baseline is one paired experiment, stored together as before.
    if set(selected) & {'B0', 'B1'}:
        selected = ['B0', 'B1', *[v for v in selected if v not in ('B0', 'B1')]]
    if stage not in ('all', 'prepare', 'train', 'evaluate'):
        raise ValueError(f'Unknown pipeline stage: {stage}')
    if stage == 'evaluate' and recompute_embeddings:
        raise ValueError('Evaluation uses saved artifacts; preparation flags are not allowed')
    if stage in ('all', 'evaluate'):
        wandb_sdk(cfg)  # Check the optional dependency before any costly preparation/training.
    if cfg['validation']['n_splits'] != 5:
        raise ValueError('The experiment requires five shared folds')
    split_path = Path(cfg['data']['splits_csv'])
    if stage == 'evaluate' and not split_path.is_file():
        raise FileNotFoundError(f'Saved folds not found: {split_path}. Run --stage prepare.')
    clean_path = Path(cfg['data']['clean_csv'])
    if not clean_path.is_file():
        if stage not in ('all', 'prepare'):
            raise FileNotFoundError(f'Cleaned data not found: {clean_path}. Run --stage prepare.')
        from src.eda import run as run_eda
        run_eda(config_path)
    data = load_data(clean_path)
    folds = make_folds(data, cfg)
    for variant in selected:
        if missing := set(VARIANTS[variant]) - set(data.columns):
            raise ValueError(f'Missing metadata for {variant}: {sorted(missing)}')
    report_root = Path(cfg['data']['report_dir']).parent
    figure_root = Path(cfg['data'].get('figure_dir', report_root / 'figures'))
    figure = figure_root / 'experiment_comparison.png'
    protocol = experiment_protocol(data, folds, cfg)
    groups = list(dict.fromkeys('baseline' if v in ('B0', 'B1') else v for v in selected))
    if stage == 'all' and 'baseline' not in groups:
        load_result(cfg, 'baseline', protocol)
    if stage in ('all', 'prepare', 'train'):
        features = load_or_compute_embeddings(folds, cfg, recompute=recompute_embeddings)
        if stage == 'prepare':
            return {'folds': folds}
        feature_hash = hashlib.sha256(features.tobytes()).hexdigest()
        for group in groups:
            run_cfg = variant_config(cfg, group)
            output = Path(run_cfg['data']['output_dir'])
            members = ['B0', 'B1'] if group == 'baseline' else [group]
            records = []
            for variant in members:
                spec = variant_spec(variant)
                LOG.info('Training %s (%s), class_weight=%s', variant, spec['stage'], spec['class_weight'])
                frame, transform = experiment_features(features, data, variant)
                records.append(fit_oof(folds, frame, cfg, output,
                                       class_weight=spec['class_weight'], preprocessor=transform))
            save_run_result(pd.concat(records, ignore_index=True), run_cfg, members,
                            protocol, feature_hash)
            save_fold_balance(folds, run_cfg['data']['report_dir'])
    if stage == 'train':
        return {'variants': selected}
    loaded = {group: load_result(cfg, group, protocol)
              for group in ['baseline', *[g for g in groups if g != 'baseline']]}
    feature_hashes = {saved['embeddings_sha256'] for _, saved in loaded.values() if saved}
    if len(feature_hashes) > 1:
        raise ValueError('Runs used different embeddings; retrain them on the same shared cache')
    predictions = pd.concat([frame for frame, _ in loaded.values()], ignore_index=True)
    results = evaluate_experiments(predictions, folds, report_root, figure)
    results['reports'], results['figures'] = {}, {}
    for group, (frame, _) in loaded.items():
        report = evaluate_predictions(frame.drop(columns='variant'), variant_config(cfg, group),
                                      model_label='B0' if group == 'baseline' else group)
        results['reports'][group] = report
        for variant in frame.variant.unique():
            results['figures'][variant] = report['figures']
    for variant in predictions.variant.unique():
        logger = ExperimentLogger(cfg, variant,
                                  {**protocol, 'embeddings_sha256': next(iter(feature_hashes), None)},
                                  variant_config(cfg, variant)['data']['output_dir'])
        success = False
        try:
            logger.log_results(variant, results, figure)
            success = True
        finally:
            logger.finish(exit_code=0 if success else 1)
    LOG.info('Experiment summary:\n%s', results['summary'].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS))
    parser.add_argument('--stage', choices=['all', 'prepare', 'train', 'evaluate'], default='all')
    parser.add_argument('--recompute-embeddings', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run(args.config, args.variants, args.stage, args.recompute_embeddings)


if __name__ == '__main__':
    main()
