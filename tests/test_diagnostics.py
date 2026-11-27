"""HW3 tests use only small synthetic data; never invoke ResNet or real experiments."""
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from src import diagnostics as d, pipeline, train
from src.evaluate import validate_oof


@pytest.fixture
def case(tmp_path):
    rows = []
    for patient in range(60):
        for image in range(2):
            rows.append(dict(image_name=f'i{patient:03}_{image}', patient_id=f'p{patient:03}',
                             target=int(patient % 3 == 0 and image == 0),
                             age_approx=[np.nan, 39, 40, 60, 70][patient % 5],
                             sex='male' if patient % 2 else 'female',
                             anatom_site_general_challenge='torso'))
    data = pd.DataFrame(rows)
    cfg = {'data': {'clean_csv': str(tmp_path / 'reports/eda/clean_train.csv'),
                    'eda_dir': str(tmp_path / 'reports/eda'),
                    'images_dir': str(tmp_path / 'jpeg'),
                    'splits_csv': str(tmp_path / 'splits/folds.csv'),
                    'output_dir': str(tmp_path / 'outputs/baseline'),
                    'report_dir': str(tmp_path / 'reports/baseline'),
                    'figure_dir': str(tmp_path / 'reports/figures')},
           'validation': {'seed': 42, 'n_splits': 5}, 'model': {},
           'logreg': {'C': 1., 'solver': 'liblinear', 'l1_ratio': 0., 'max_iter': 2000,
                      'tol': .0001, 'class_weights': [None, 'balanced']},
           'wandb': {'enabled': False}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    clean = Path(cfg['data']['clean_csv'])
    clean.parent.mkdir(parents=True)
    data.to_csv(clean, index=False)
    data[['image_name', 'target']].assign(width=1000, height=1000).to_csv(
        clean.parent / 'image_statistics.csv', index=False)
    features = np.random.default_rng(8).normal(size=(len(data), 512)).astype('float32')
    folds = train.make_folds(data, cfg)
    return path, cfg, data, folds, features


def save_oof(case):
    _, cfg, _, folds, features = case
    # Deliberately erroneous scores for deterministic error diagnostics in temporary tests.
    oof = folds.assign(score=np.where(folds.target.eq(1), .1, .6), class_weight='none')
    data = pd.read_csv(cfg['data']['clean_csv'])
    protocol = pipeline.experiment_protocol(data, folds, cfg)
    pipeline.save_run_result(oof, pipeline.variant_config(cfg, 'M-full'), ['M-full'], protocol,
                             hashlib.sha256(features.tobytes()).hexdigest())
    return oof


def test_seed_reproducibility_and_patient_disjointness(case):
    _, cfg, data, original, _ = case
    pd.testing.assert_frame_equal(original, train.generate_folds(data, cfg, 42))
    assignments = []
    for seed in d.SEEDS:
        first = train.generate_folds(data, cfg, seed)
        pd.testing.assert_frame_equal(first, train.generate_folds(data, cfg, seed))
        assert first.groupby('patient_id').fold.nunique().max() == 1
        for fold in range(5):
            assert not set(first.loc[first.fold.eq(fold), 'patient_id']) & set(
                first.loc[first.fold.ne(fold), 'patient_id'])
        assignments.append(first.fold.to_numpy())
    assert any(not np.array_equal(assignments[0], other) for other in assignments[1:])
    assert cfg['validation']['seed'] == 42


def test_nested_stratified_whole_patient_subsamples(case):
    _, _, data, folds, _ = case
    for fold in range(5):
        subsets = d.nested_train_masks(folds, fold)
        repeat = d.nested_train_masks(folds.sample(frac=1, random_state=9), fold)
        ids = {f: set(folds.loc[mask, 'image_name']) for f, mask in subsets.items()}
        assert ids[.2] < ids[.5] < ids[1.]
        for fraction, mask in subsets.items():
            np.testing.assert_array_equal(mask, d.nested_train_masks(folds, fold)[fraction])
            shuffled = folds.sample(frac=1, random_state=9)
            assert set(shuffled.loc[repeat[fraction], 'image_name']) == ids[fraction]
            assert not (mask & folds.fold.eq(fold).to_numpy()).any()
            patients = set(folds.loc[mask, 'patient_id'])
            assert set(data.loc[data.patient_id.isin(patients), 'image_name']) == ids[fraction]
            assert set(folds.loc[mask].groupby('patient_id').target.max()) == {0, 1}
            counts = d.training_counts(data, mask)
            assert sum(v for k, v in counts.items() if k.startswith('train_age_')) == mask.sum()
            assert counts['train_n_patients'] == len(patients)
        np.testing.assert_array_equal(subsets[1.], folds.fold.ne(fold))


def test_fit_uses_subset_only_and_same_validation(case, monkeypatch):
    _, cfg, data, folds, features = case
    frame, transform = train.experiment_features(features, data, 'M-full')
    captures = []
    real_pipeline = train.make_pipeline

    def capture_pipeline(*steps):
        model = real_pipeline(*steps)
        actual_fit = model.fit

        def fit(x, y):
            result = actual_fit(x, y)
            assert len(x) == len(y)
            pre = model.named_steps['columntransformer']
            age = pre.named_transformers_['age'].named_steps['simpleimputer']
            expected = x.age_approx.median() if x.age_approx.notna().any() else 0
            assert age.statistics_[0] == expected
            np.testing.assert_allclose(pre.named_transformers_['image'].mean_,
                                       x.iloc[:, :512].astype('float64').mean())
            encoder = pre.named_transformers_['categorical'].named_steps['onehotencoder']
            for column, categories in zip(train.METADATA[1:], encoder.categories_):
                assert set(categories) == set(x[column].fillna('MISSING'))
            assert model.named_steps['logisticregression'].random_state == 42
            captures.append(set(x.index))
            return result
        model.fit = fit
        return model

    monkeypatch.setattr(train, 'make_pipeline', capture_pipeline)
    subsets = {fold: d.nested_train_masks(folds, fold) for fold in range(5)}
    expected_validation = None
    for fraction in d.FRACTIONS:
        masks = {fold: subsets[fold][fraction] for fold in range(5)}
        oof = train.fit_oof(folds, frame, cfg, preprocessor=transform, train_masks=masks)
        validate_oof(oof, folds)
        validation = oof[['image_name', 'patient_id', 'target', 'fold']]
        if expected_validation is None:
            expected_validation = validation
        pd.testing.assert_frame_equal(validation, expected_validation)
        assert captures[-5:] == [set(np.flatnonzero(masks[fold])) for fold in range(5)]
    with pytest.raises(ValueError, match='exclude validation'):
        train.fit_oof(folds, frame, cfg, preprocessor=transform,
                      train_masks={fold: np.ones(len(folds), dtype=bool) for fold in range(5)})


def test_original_image_only_fit_unchanged(case):
    _, cfg, _, folds, features = case
    for weight in (None, 'balanced'):
        oof = train.fit_oof(folds, features, cfg, class_weight=weight).set_index('image_name')
        for fold in range(5):
            val = folds.fold.eq(fold).to_numpy()
            original = make_pipeline(StandardScaler(), train.make_logreg(cfg, weight))
            original.fit(features[~val], folds.target.to_numpy()[~val])
            np.testing.assert_allclose(oof.loc[folds.loc[val, 'image_name'], 'score'],
                                       original.predict_proba(features[val])[:, 1], atol=1e-14)


def test_age_and_mp_boundaries_and_missing():
    assert d.age_groups(pd.Series([0, 39.99, 40, 59.99, 60, 69.99, 70, 99, np.nan])).tolist() == [
        '<40', '<40', '40–59', '40–59', '60–69', '60–69', '70+', '70+', 'Unknown']
    width = pd.Series([999, 1000, 2999, 3000, 9999, 10000, 19999, 20000, np.nan, 1000])
    height = pd.Series([1000] * 9 + [np.nan])
    assert d.mp_groups(width, height).tolist() == [
        '<1 MP', '1–3 MP', '1–3 MP', '3–10 MP', '3–10 MP', '10–20 MP',
        '10–20 MP', '≥20 MP', 'Unknown', 'Unknown']
    with pytest.raises(ValueError, match='positive integers'):
        d.mp_groups(pd.Series([0]), pd.Series([1000]))
    with pytest.raises(ValueError, match='nonnegative'):
        d.age_groups(pd.Series([-1]))


def test_pooled_metrics_single_classes_and_empty_groups():
    frame = pd.DataFrame({'group': ['mixed'] * 4 + ['negative'] * 2 + ['positive'] * 2,
                          'target': [1, 0, 1, 0, 0, 0, 1, 1],
                          'score': [.8, .9, .2, .1, .3, .1, .2, .7]})
    table = d.group_metrics(frame, 'group', ['mixed', 'negative', 'positive', 'empty']).set_index('group')
    mixed = table.loc['mixed']
    assert mixed.n == 4 and mixed.n_positive == 2 and mixed.prevalence == .5
    assert mixed.average_precision == pytest.approx(average_precision_score([1, 0, 1, 0], [.8, .9, .2, .1]))
    assert mixed.roc_auc == pytest.approx(roc_auc_score([1, 0, 1, 0], [.8, .9, .2, .1]))
    assert mixed.ap_over_prevalence == mixed.average_precision / .5
    assert table.loc['negative', ['roc_auc', 'average_precision', 'ap_over_prevalence']].isna().all()
    assert np.isnan(table.loc['positive', 'roc_auc'])
    assert table.loc['positive', 'average_precision'] == 1
    assert table.loc['empty', 'n'] == 0 and np.isnan(table.loc['empty', 'prevalence'])


def test_slice_merge_retains_missing_sizes_and_validates_keys(case):
    _, _, data, folds, _ = case
    oof = folds.assign(score=.3, class_weight='none')
    stats = data[['image_name', 'target']].assign(width=1000, height=1000)
    merged = d.slice_frame(oof, data, stats.iloc[1:])
    assert len(merged) == len(oof) and merged.mp_group.eq('Unknown').sum() == 1
    for bad in (pd.concat([stats, stats.iloc[:1]]), stats.assign(target=1)):
        with pytest.raises(ValueError):
            d.slice_frame(oof, data, bad)
    extra = pd.concat([stats, stats.iloc[:1].assign(image_name='excluded')])
    with pytest.raises(ValueError, match='exclusions'):
        d.slice_frame(oof, data, extra, pd.DataFrame({'image_name': ['wrong']}))
    assert len(d.slice_frame(oof, data, extra, pd.DataFrame({'image_name': ['excluded']}))) == len(oof)


def test_errors_deterministic_extremes_near_threshold_and_rare_type():
    oof = pd.DataFrame({'image_name': list('abcdefgh'), 'target': [1, 1, 0, 0, 1, 0, 1, 0],
                        'score': [.01, .02, .99, .98, .499, .501, .8, .1], 'fold': 0})
    selected = d.select_errors(oof)
    assert selected.image_name.tolist() == list('abcde')  # Ties by image_name.
    pd.testing.assert_frame_equal(selected, d.select_errors(oof.sample(frac=1, random_state=7)))
    rare = d.select_errors(oof.loc[oof.image_name.ne('d')])
    assert set(rare.error) == {'FP', 'FN'} and len(rare) == 5
    fn = d.select_errors(oof.loc[oof.target.eq(1)])
    assert set(fn.error) == {'FN'} and len(fn) == 3


def test_light_stages_do_not_extract_or_fit(case, monkeypatch, tmp_path):
    path, cfg, _, _, _ = case
    oof = save_oof(case)
    for function in ('load_or_compute_embeddings', 'fit_oof', 'wandb_sdk'):
        monkeypatch.setattr(d, function, lambda *a, **kw: pytest.fail('Heavy work in a light stage'))
    # Local CSV analysis must not require matching installed torch/torchvision versions.
    manifest = tmp_path / 'outputs/M-full/run_config.json'
    saved = json.loads(manifest.read_text())
    saved['protocol']['image_extractor']['libraries']['torch'] = 'other-runtime'
    manifest.write_text(json.dumps(saved))
    d.run(path, 'slices')
    report = tmp_path / 'reports/M-full'
    assert {p.name for p in report.iterdir()} == {'age_slices.csv', 'mp_slices.csv'}
    assert pd.read_csv(report / 'mp_slices.csv').n.sum() == len(oof)
    error_path = tmp_path / 'reports/figures/M-full/error_analysis.png'
    with pytest.raises(FileNotFoundError, match='JPEG not found'):
        d.run(path, 'errors')
    assert not error_path.exists()
    images = Path(cfg['data']['images_dir'])
    images.mkdir()
    for name in d.select_errors(oof).image_name:
        Image.new('RGB', (12, 16)).save(images / f'{name}.jpg')
    d.run(path, 'errors')
    assert error_path.read_bytes().startswith(b'\x89PNG')
    assert not (report / 'error_examples.csv').exists()


def test_oof_and_original_fold_mismatch_rejected_before_extraction(case, monkeypatch, tmp_path):
    path, cfg, data, folds, _ = case
    oof = save_oof(case)
    with pytest.raises(ValueError, match='OOF fold'):
        validate_oof(oof.assign(fold=(oof.fold + 1) % 5), folds)
    monkeypatch.setattr(d, 'load_or_compute_embeddings', lambda *a: pytest.fail('Unexpected extraction'))
    bad = folds.copy()
    bad['fold'] = (bad.fold + 1) % 5
    bad.to_csv(cfg['data']['splits_csv'], index=False)
    with pytest.raises(ValueError, match='Seed=42'):
        d.run(path, 'all')
    folds.to_csv(cfg['data']['splits_csv'], index=False)
    data.loc[0, 'age_approx'] = 88
    data.to_csv(cfg['data']['clean_csv'], index=False)
    with pytest.raises(ValueError, match='saved run does not match'):
        d.run(path, 'slices')


def test_full_cli_one_extraction_only_mfull_and_robustness(case, monkeypatch, tmp_path):
    path, cfg, data, folds, features = case
    events, fitted = [], []
    # Simulate absent EDA output; the orchestrator must call the existing EDA entrypoint.
    Path(cfg['data']['clean_csv']).unlink()
    Path(cfg['data']['splits_csv']).unlink()
    import src.eda

    def prepare(config_path):
        assert config_path == str(path)
        events.append('eda')
        data.to_csv(cfg['data']['clean_csv'], index=False)

    def extract(frame, config):
        events.append('extract')
        pd.testing.assert_frame_equal(frame, folds)
        return features.copy()

    def fitting(fold_frame, x, config, **kwargs):
        assert config['validation']['seed'] == 42
        assert kwargs.get('class_weight') is None
        assert list(x.columns[512:]) == train.METADATA
        fitted.append(kwargs.get('train_masks'))
        result = train.fit_oof(fold_frame, x, config, **kwargs)
        if kwargs.get('output'):
            images = Path(cfg['data']['images_dir'])
            images.mkdir()
            for name in d.select_errors(result).image_name:
                Image.new('RGB', (12, 16)).save(images / f'{name}.jpg')
        return result

    monkeypatch.setattr(src.eda, 'run', prepare)
    monkeypatch.setattr(train, 'embeddings', extract)
    monkeypatch.setattr(d, 'fit_oof', fitting)
    monkeypatch.setattr(sys, 'argv', ['src.diagnostics', '--config', str(path), '--stage', 'all'])
    d.main()
    assert events == ['eda', 'extract']
    assert len(fitted) == 7  # Original M-full + 3 split seeds + 3 train fractions.
    assert len(list((tmp_path / 'outputs/M-full').glob('*.joblib'))) == 5
    assert (tmp_path / 'outputs/baseline/embeddings.npz').is_file()
    assert not (tmp_path / 'outputs/baseline/oof_predictions.csv').exists()
    assert not (tmp_path / 'outputs/A-age').exists()
    report, figures = tmp_path / 'reports/M-full', tmp_path / 'reports/figures/M-full'
    assert {p.name for p in report.iterdir()} == {
        'age_slices.csv', 'mp_slices.csv', 'seed_robustness.csv', 'train_size_robustness.csv'}
    assert {p.name for p in figures.iterdir()} == {
        'error_analysis.png', 'age_slices.png', 'mp_slices.png',
        'seed_robustness.png', 'train_size_robustness.png'}
    seed_table = pd.read_csv(report / 'seed_robustness.csv')
    size_table = pd.read_csv(report / 'train_size_robustness.csv')
    for table, factor in ((seed_table, 'split_seed'), (size_table, 'train_fraction')):
        assert len(table) == 21
        for value, part in table.groupby(factor):
            fold_rows = part.loc[part.row_type.eq('fold')]
            assert len(fold_rows) == 5
            for metric in d.METRICS:
                assert part.loc[part.row_type.eq('mean'), metric].iloc[0] == pytest.approx(fold_rows[metric].mean())
                assert part.loc[part.row_type.eq('std'), metric].iloc[0] == pytest.approx(fold_rows[metric].std(ddof=1))
    for metric in d.METRICS:
        np.testing.assert_allclose(
            seed_table.loc[seed_table.row_type.eq('fold') & seed_table.split_seed.eq(42), metric],
            size_table.loc[size_table.row_type.eq('fold') & size_table.train_fraction.eq(1), metric])
    # Reusing saved original OOF skips original M-full fitting and reuses the disk cache.
    original_oof = (tmp_path / 'outputs/M-full/oof_predictions.csv').read_bytes()
    fitted.clear()
    d.run(str(path), 'all')
    assert events == ['eda', 'extract'] and len(fitted) == 6
    assert (tmp_path / 'outputs/M-full/oof_predictions.csv').read_bytes() == original_oof


def test_wandb_robustness_only_and_disabled(case, monkeypatch, tmp_path):
    _, cfg, _, _, _ = case
    records = [dict(split_seed=seed, fold=fold, roc_auc=.6 + fold / 100, average_precision=.2)
               for seed in d.SEEDS for fold in range(5)]
    table = d.robustness_table(records, 'split_seed')
    calls = []
    monkeypatch.setattr(pipeline.importlib, 'import_module',
                        lambda *a: pytest.fail('Disabled W&B must not be imported'))
    d.log_robustness(cfg, {}, {'seed_robustness': table}, tmp_path)

    class FakeRun:
        def __init__(self, **kwargs):
            self.options, self.summary, self.logs = kwargs, {}, []
            calls.append(self)
        def log(self, values):
            self.logs.append(values)
        def finish(self, exit_code):
            self.exit_code = exit_code

    sdk = types.SimpleNamespace(init=FakeRun, Table=lambda **kw: kw, Image=str)
    monkeypatch.setattr(pipeline.importlib, 'import_module', lambda name: sdk)
    cfg['wandb'] = dict(enabled=True, project='same-project', mode='offline')
    d.log_robustness(cfg, {'validation': cfg['validation']}, {'seed_robustness': table}, tmp_path)
    assert len(calls) == 1
    run = calls[0]
    assert run.options['project'] == 'same-project' and run.options['job_type'] == 'robustness'
    assert run.options['name'] == 'M-full-seed_robustness'
    assert len(run.logs) == 16 and run.exit_code == 0
    assert 'split_seed/42/roc_auc_std' in run.summary


def test_standalone_robustness_without_oof_or_jpegs(case, monkeypatch, tmp_path):
    path, _, _, _, features = case
    calls = []

    def extract(*args):
        calls.append('extract')
        return features.copy()

    monkeypatch.setattr(train, 'embeddings', extract)
    monkeypatch.setattr(d, 'load_result', lambda *a, **kw: pytest.fail('OOF not needed for robustness'))
    tables = d.run(path, 'robustness')
    assert calls == ['extract']
    assert set(tables) == {'seed_robustness', 'train_size_robustness'}
    assert not (tmp_path / 'outputs/M-full/oof_predictions.csv').exists()
    assert not list(tmp_path.rglob('*.joblib'))


def test_saved_oof_embeddings_mismatch_is_not_silently_reused(case, monkeypatch, tmp_path):
    path, _, _, _, features = case
    save_oof(case)
    manifest = tmp_path / 'outputs/M-full/run_config.json'
    saved = json.loads(manifest.read_text())
    saved['embeddings_sha256'] = 'different-embeddings'
    manifest.write_text(json.dumps(saved))
    monkeypatch.setattr(d, 'load_or_compute_embeddings', lambda *a: features)
    monkeypatch.setattr(d, 'fit_oof', lambda *a, **kw: pytest.fail('Existing OOF must not be overwritten'))
    with pytest.raises(ValueError, match='different embeddings'):
        d.run(path, 'all')
    assert not (tmp_path / 'reports/figures/M-full').exists()
