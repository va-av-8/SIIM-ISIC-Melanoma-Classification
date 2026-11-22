import json
from pathlib import Path
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import pipeline, train
from src.evaluate import evaluate, evaluate_experiments, EXPERIMENT_METRICS, plot_error_examples


REPORT_FILES = {
    'fold_balance.csv', 'fold_metrics.csv', 'metrics_summary.csv', 'pooled_oof_metrics.csv',
    'confusion_threshold_0_5.csv', 'error_examples.csv', 'resolution_metrics.csv',
    'roc_oof.png', 'pr_oof.png',
}
FIGURE_FILES = {'roc_oof.png', 'pr_oof.png', 'resolution_analysis.png', 'hard_errors.png'}


@pytest.fixture
def experiment_case(tmp_path, monkeypatch):
    data = pd.DataFrame({
        'image_name': [f'image_{i:02}' for i in range(20)],
        'patient_id': [f'patient_{i}' for i in range(20)], 'target': [0, 1, 0, 0] * 5,
        'age_approx': [20, 40, np.nan, 60] * 5, 'sex': ['male', 'female', np.nan, 'male'] * 5,
        'anatom_site_general_challenge': ['torso', np.nan, 'head/neck', 'torso'] * 5,
        'diagnosis': ['benign', 'melanoma'] * 10,
        'benign_malignant': ['benign', 'malignant'] * 10,
    })
    eda = tmp_path / 'reports' / 'eda'
    eda.mkdir(parents=True)
    data.to_csv(eda / 'clean_train.csv', index=False)
    data[['image_name', 'target']].assign(width=32, height=24).to_csv(
        eda / 'image_statistics.csv', index=False)
    folds = data[['image_name', 'patient_id', 'target']].assign(fold=np.repeat(np.arange(5), 4))
    (tmp_path / 'splits').mkdir()
    folds.to_csv(tmp_path / 'splits' / 'folds.csv', index=False)
    cfg = {
        'data': {'clean_csv': str(eda / 'clean_train.csv'), 'eda_dir': str(eda),
                 'splits_csv': str(tmp_path / 'splits' / 'folds.csv'),
                 'figure_dir': str(tmp_path / 'reports' / 'figures'),
                 'output_dir': str(tmp_path / 'outputs' / 'baseline'),
                 'report_dir': str(tmp_path / 'reports' / 'baseline')},
        'model': {}, 'validation': {'n_splits': 5, 'seed': 42},
        'logreg': {'C': 1., 'solver': 'liblinear', 'l1_ratio': 0., 'max_iter': 2000,
                   'tol': .0001, 'class_weights': [None, 'balanced']},
        'experiments': {'variants': list(train.VARIANTS)},
        'wandb': {'enabled': False}, 'resolution_analysis': {'top_n': 10, 'min_positive': 2},
    }
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    features = np.random.default_rng(5).normal(size=(20, 512)).astype(np.float32)
    calls = []

    def extract(frame, config):
        calls.append(frame.image_name.tolist())
        return features.copy()

    monkeypatch.setattr(train, 'embeddings', extract)
    return path, cfg, data, folds, features, calls


def assert_complete_outputs(root, groups=('baseline', 'M-full', 'A-age', 'A-sex', 'A-site')):
    for group in groups:
        output, report = root / 'outputs' / group, root / 'reports' / group
        figures = root / 'reports' / 'figures'
        if group != 'baseline':
            figures /= group
        assert (output / 'oof_predictions.csv').is_file()
        assert (output / 'run_config.json').is_file()
        assert len(list(output.glob('lr_*_fold*.joblib'))) == (10 if group == 'baseline' else 5)
        assert {p.name for p in report.iterdir()} == REPORT_FILES
        for name in FIGURE_FILES:
            assert (figures / name).read_bytes().startswith(b'\x89PNG')
        for name in ('roc_oof.png', 'pr_oof.png'):
            assert (figures / name).read_bytes() == (report / name).read_bytes()
        assert pd.read_csv(report / 'fold_balance.csv').n_images.sum() == 20
        assert pd.read_csv(report / 'resolution_metrics.csv').n.sum() == 20
    assert (root / 'reports' / 'all_fold_metrics.csv').is_file()
    assert (root / 'reports' / 'all_metrics_summary.csv').is_file()
    assert (root / 'reports' / 'figures' / 'experiment_comparison.png').is_file()
    assert not (root / 'outputs' / 'experiments').exists()
    assert not (root / 'reports' / 'experiments').exists()


def test_preprocessing_learns_only_train_values():
    features = np.zeros((4, 512), dtype=np.float32)
    features[3] = 1000
    data = pd.DataFrame({'age_approx': [20, 40, np.nan, 1000],
                         'sex': ['male', 'female', np.nan, 'validation_only'],
                         'anatom_site_general_challenge': ['torso', 'torso', np.nan, 'new_site'],
                         'diagnosis': [0, 1, 0, 1]})
    frame, transform = train.experiment_features(features, data, 'M-full')
    transform.fit(frame.iloc[:3])
    age = transform.named_transformers_['age']
    assert age.named_steps['simpleimputer'].statistics_[0] == 30
    assert age.named_steps['standardscaler'].mean_[0] == 30
    assert transform.named_transformers_['image'].mean_[0] == 0
    encoder = transform.named_transformers_['categorical'].named_steps['onehotencoder']
    assert 'validation_only' not in encoder.categories_[0]
    assert 'new_site' not in encoder.categories_[1]
    assert 'MISSING' in encoder.categories_[0]
    before = transform.transform(frame.iloc[:3])
    assert np.isfinite(transform.transform(frame.iloc[[3]])).all()
    np.testing.assert_array_equal(before, transform.transform(frame.iloc[:3]))
    assert 'diagnosis' not in frame
    frame.loc[:2, 'age_approx'] = np.nan
    transform.fit(frame.iloc[:3])
    assert np.isfinite(transform.transform(frame)).all()


@pytest.mark.parametrize('variant,columns', [
    ('B0', []), ('B1', []), ('M-full', train.METADATA), ('A-age', train.METADATA[1:]),
    ('A-sex', [train.METADATA[0], train.METADATA[2]]), ('A-site', train.METADATA[:2]),
])
def test_ablation_feature_whitelist(experiment_case, variant, columns):
    _, _, data, _, features, _ = experiment_case
    frame, _ = train.experiment_features(features, data, variant)
    assert train.VARIANTS[variant] == columns
    if variant in ('B0', 'B1'):
        np.testing.assert_array_equal(frame, features)
    else:
        assert list(frame.columns[512:]) == columns
        assert not {'diagnosis', 'benign_malignant', 'target', 'patient_id'} & set(frame.columns)


def test_complete_reports_cache_baseline_and_csv_only_evaluation(experiment_case, monkeypatch, tmp_path):
    path, cfg, _, folds, features, calls = experiment_case
    original_import = pipeline.importlib.import_module

    def forbid_wandb(name, *args, **kwargs):
        if name == 'wandb' or name.startswith('wandb.'):
            pytest.fail('Disabled W&B must not be imported')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(pipeline.importlib, 'import_module', forbid_wandb)
    original_folds = Path(cfg['data']['splits_csv']).read_bytes()
    result = pipeline.run(path)
    assert len(calls) == 1
    assert len(result['fold_metrics']) == 30 and len(result['summary']) == 6
    assert Path(cfg['data']['splits_csv']).read_bytes() == original_folds
    assert_complete_outputs(tmp_path)
    per_fold = result['fold_metrics'].set_index(['variant', 'fold'])
    for row in result['fold_metrics'].itertuples(index=False):
        for metric in EXPERIMENT_METRICS:
            assert getattr(row, f'{metric}_delta_vs_B0') == pytest.approx(
                per_fold.loc[(row.variant, row.fold), metric] - per_fold.loc[('B0', row.fold), metric])
    baseline_dir = tmp_path / 'outputs' / 'baseline'
    oof = pd.read_csv(baseline_dir / 'oof_predictions.csv')
    assert set(oof.class_weight) == {'none', 'balanced'}
    for weight in (None, 'balanced'):
        baseline = oof.loc[oof.class_weight.eq(weight or 'none')].set_index('image_name')
        for fold in range(5):
            val = folds.fold.to_numpy() == fold
            clf = make_pipeline(StandardScaler(), train.make_logreg(cfg, weight))
            clf.fit(features[~val], folds.target.to_numpy()[~val])
            np.testing.assert_allclose(baseline.loc[folds.loc[val, 'image_name'], 'score'],
                                       clf.predict_proba(features[val])[:, 1], atol=1e-14)
    model = joblib.load(tmp_path / 'outputs' / 'M-full' / 'lr_none_fold0.joblib')
    assert 'diagnosis' not in model.named_steps['columntransformer'].feature_names_in_
    # All reports, including error examples and figures, must be recreated without fitting.
    before = {p.relative_to(tmp_path): pd.read_csv(p) for p in (tmp_path / 'reports').rglob('*.csv')
              if 'eda' not in p.parts}
    for p in (tmp_path / 'reports').rglob('*'):
        if p.is_file() and 'eda' not in p.parts:
            p.unlink()
    monkeypatch.setattr(pipeline, 'load_or_compute_embeddings',
                        lambda *a, **k: pytest.fail('Evaluation attempted extraction'))
    monkeypatch.setattr(pipeline, 'fit_oof', lambda *a, **k: pytest.fail('Evaluation attempted training'))
    replay = pipeline.run(path, stage='evaluate')
    assert_complete_outputs(tmp_path)
    pd.testing.assert_frame_equal(result['summary'], replay['summary'])
    for name, expected in before.items():
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), expected)
    assert len(calls) == 1


def test_standalone_commands_and_partial_variant(experiment_case, tmp_path):
    path, cfg, _, _, _, calls = experiment_case
    train.run(path)  # Original baseline output, without the new run_config.json.
    evaluate(path)
    output = Path(cfg['data']['output_dir'])
    baseline_bytes = (output / 'oof_predictions.csv').read_bytes()
    expected = {name: pd.read_csv(Path(cfg['data']['report_dir']) / name)
                for name in REPORT_FILES if name.endswith('.csv')}
    result = pipeline.run(path, variants=['M-full'])
    assert set(result['summary'].variant) == {'B0', 'B1', 'M-full'}
    assert (output / 'oof_predictions.csv').read_bytes() == baseline_bytes
    assert len(list(output.glob('*.joblib'))) == 10
    for name, frame in expected.items():
        pd.testing.assert_frame_equal(pd.read_csv(Path(cfg['data']['report_dir']) / name), frame)
    assert not (tmp_path / 'outputs' / 'A-age').exists()
    assert len(calls) == 1


def test_protocol_mismatch_and_full_rerun(experiment_case, tmp_path):
    path, cfg, data, _, _, calls = experiment_case
    pipeline.run(path, variants=['B0'], stage='train')
    pipeline.run(path, variants=['M-full'], stage='train')
    data.loc[0, 'age_approx'] = 99
    data.to_csv(cfg['data']['clean_csv'], index=False)
    with pytest.raises(ValueError, match='does not match current'):
        pipeline.run(path, stage='evaluate', variants=['M-full'])
    pipeline.run(path)
    assert_complete_outputs(tmp_path)
    assert len(calls) == 1
    saved_path = tmp_path / 'outputs' / 'M-full' / 'run_config.json'
    saved = json.loads(saved_path.read_text())
    saved['embeddings_sha256'] = 'different-embeddings'
    saved_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='different embeddings'):
        pipeline.run(path, variants=['M-full'], stage='evaluate')


def test_folds_created_when_missing_and_validated_when_present(experiment_case):
    path, cfg, data, folds, _, calls = experiment_case
    pd.testing.assert_frame_equal(train.make_folds(data, cfg), folds)
    folds.loc[0, 'target'] = 1
    folds.to_csv(cfg['data']['splits_csv'], index=False)
    with pytest.raises(ValueError, match='different target'):
        pipeline.run(path)
    Path(cfg['data']['splits_csv']).unlink()
    with pytest.raises(FileNotFoundError, match='Saved folds'):
        pipeline.run(path, stage='evaluate')
    assert not calls
    created = pipeline.run(path, stage='prepare')['folds']
    pd.testing.assert_frame_equal(created, train.make_folds(data, cfg))
    assert created.groupby('patient_id').fold.nunique().max() == 1


def test_oof_pairing_validation(experiment_case, tmp_path):
    _, _, _, folds, _, _ = experiment_case
    oof = folds.assign(score=.5, variant='B0', class_weight='none')
    bad = oof.copy()
    bad.loc[0, 'fold'] = 1
    with pytest.raises(ValueError, match='OOF fold differs'):
        evaluate_experiments(bad, folds, tmp_path / 'report', tmp_path / 'plot.png')
    with pytest.raises(ValueError, match='image keys'):
        evaluate_experiments(pd.concat([oof, oof.iloc[[0]]]), folds, tmp_path, tmp_path / 'p.png')


def test_error_collage_uses_selected_jpegs(tmp_path, monkeypatch):
    images = tmp_path / 'jpeg'
    images.mkdir()
    rows = []
    for i, (error, score) in enumerate([('FP', .6), ('FP', .9), ('FP', .8),
                                       ('FN', .4), ('FN', .01), ('FN', .1)]):
        name = f'image_{i}'
        Image.new('RGB', (16, 16), color=(i * 30, 0, 0)).save(images / f'{name}.jpg')
        rows.append(dict(image_name=name, error=error, target=int(error == 'FN'), score=score, fold=0))
    opened, original = [], Image.open

    def capture(path, *args, **kwargs):
        opened.append(Path(path).stem)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, 'open', capture)
    cfg = {'data': {'images_dir': str(images), 'report_dir': str(tmp_path / 'report'),
                    'figure_dir': str(tmp_path / 'figures')}}
    plot_error_examples(pd.DataFrame(rows), cfg, 'M-full')
    assert opened == ['image_1', 'image_2', 'image_4', 'image_5']
    assert (tmp_path / 'figures' / 'hard_errors.png').read_bytes().startswith(b'\x89PNG')


def test_wandb_one_run_per_variant_and_epoch_hook(experiment_case, monkeypatch):
    path, cfg, _, _, _, _ = experiment_case
    runs = []

    class FakeRun:
        def __init__(self, **kwargs):
            self.options, self.summary, self.logs, self.axes = kwargs, {}, [], []
            self.exit_code = None
            runs.append(self)

        def define_metric(self, *args, **kwargs):
            self.axes.append((args, kwargs))

        def log(self, values):
            self.logs.append(values)

        def finish(self, exit_code=0):
            self.exit_code = exit_code

    fake_sdk = types.SimpleNamespace(init=FakeRun, Image=str, Table=lambda **kwargs: kwargs)
    original_import = pipeline.importlib.import_module
    monkeypatch.setattr(pipeline.importlib, 'import_module',
                        lambda name: fake_sdk if name == 'wandb' else original_import(name))
    cfg['wandb'] = {'enabled': True, 'project': 'test-project', 'entity': None,
                    'group': 'test-group', 'mode': 'offline'}
    path.write_text(yaml.safe_dump(cfg))
    pipeline.run(path)
    assert len(runs) == 6
    for run in runs:
        assert run.options['group'] == 'test-group' and run.options['mode'] == 'offline'
        variant = run.options['config']['variant']
        spec = train.variant_spec(variant)
        assert run.options['config']['logreg']['class_weight'] == spec['class_weight']
        assert run.options['job_type'] == spec['stage']
        assert run.options['config']['validation']['seed'] == 42
        assert len([log for log in run.logs if 'fold' in log]) == 5
        assert 'average_precision_mean' in run.summary and 'tpr_at_fpr_le_0_05_std' in run.summary
        for name in ('comparison', 'roc_oof', 'pr_oof'):
            assert Path(run.logs[-1][name]).is_file()
        for name in ('resolution_analysis', 'hard_errors'):
            assert (name in run.logs[-1]) == (variant != 'B1')
        assert run.exit_code == 0
    logger = pipeline.ExperimentLogger(cfg, 'B0', {}, cfg['data']['output_dir'])
    logger.log_epoch(1, 2, {'loss': .5})
    assert runs[-1].logs[-1] == {'epoch/fold_1/epoch': 2, 'epoch/fold_1/loss': .5}
    logger.finish()


def test_wandb_missing_dependency_fails_before_training(experiment_case, monkeypatch):
    path, cfg, _, _, _, calls = experiment_case
    cfg['wandb'] = {'enabled': True}
    path.write_text(yaml.safe_dump(cfg))
    original_import = pipeline.importlib.import_module

    def missing_sdk(name):
        if name == 'wandb':
            raise ImportError('wandb not installed')
        return original_import(name)

    monkeypatch.setattr(pipeline.importlib, 'import_module', missing_sdk)
    with pytest.raises(ImportError, match='install wandb'):
        pipeline.run(path)
    assert not calls


def test_full_cli_from_raw_data(experiment_case, monkeypatch, tmp_path):
    path, cfg, data, _, _, calls = experiment_case
    Path(cfg['data']['clean_csv']).unlink()
    Path(cfg['data']['splits_csv']).unlink()
    (Path(cfg['data']['eda_dir']) / 'image_statistics.csv').unlink()
    data.to_csv(tmp_path / 'raw.csv', index=False)
    images = tmp_path / 'jpeg'
    images.mkdir()
    rng = np.random.default_rng(9)
    for name in data.image_name:
        Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(images / f'{name}.jpg')
    cfg['data'].update(labels_csv=str(tmp_path / 'raw.csv'), images_dir=str(images))
    cfg['eda'] = {'dark_pixel_threshold': 32}
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, 'argv', ['src.pipeline', '--config', str(path)])
    pipeline.main()
    assert len(calls) == 1
    assert_complete_outputs(tmp_path)
    assert (tmp_path / 'splits' / 'folds.csv').is_file()
    assert (tmp_path / 'outputs' / 'baseline' / 'embeddings.npz').is_file()
    eda = tmp_path / 'reports' / 'eda'
    for name in ('clean_train.csv', 'image_statistics.csv', 'excluded_images.csv',
                 'exact_duplicates.csv', 'class_counts.csv', 'patients.csv', 'missing.csv',
                 'missing_by_class.csv', 'missing_images.csv', 'unreferenced_images.csv',
                 'by_age.csv', 'by_sex.csv', 'by_anatom_site_general_challenge.csv',
                 'by_diagnosis.csv', 'by_benign_malignant.csv', 'summary.json'):
        assert (eda / name).is_file()
    for name in ('brightness', 'contrast', 'dark_fraction', 'images_per_patient',
                 'by_age', 'by_sex', 'by_anatom_site_general_challenge'):
        assert (eda / 'figures' / f'{name}.png').is_file()
