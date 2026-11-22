import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.train as training


@pytest.fixture
def cache_case(tmp_path, monkeypatch):
    df = pd.DataFrame({'image_name': ['a', 'b', 'c'], 'target': [0, 1, 0],
                       'patient_id': ['p1', 'p2', 'p3']})
    cfg = {'data': {'output_dir': str(tmp_path / 'outputs'),
                    'images_dir': str(tmp_path / 'missing_jpegs')},
           'model': {'weights_path': None, 'device': 'cpu'}}
    features = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    calls = []

    def fake_extract(frame, config):
        calls.append(frame.image_name.tolist())
        return features.copy()

    def forbid_fit(*args, **kwargs):
        pytest.fail('Training must never run in cache tests')

    monkeypatch.setattr(training, 'embeddings', fake_extract)
    monkeypatch.setattr(training.LogisticRegression, 'fit', forbid_fit)
    monkeypatch.setattr(training.StandardScaler, 'fit', forbid_fit)
    return df, cfg, features, calls


def test_save_and_reuse_in_current_image_order(cache_case):
    df, cfg, expected, calls = cache_case
    np.testing.assert_array_equal(training.load_or_compute_embeddings(df, cfg), expected)
    path = Path(cfg['data']['output_dir']) / 'embeddings.npz'
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['image_name'], df.image_name)
        np.testing.assert_array_equal(saved['features'], expected)
        assert saved['features'].dtype == np.float32
        assert json.loads(saved['metadata'].item()) == training.embedding_signature(cfg)
    # Fold, target, device and classifier settings do not change frozen image features.
    reordered = df.iloc[[2, 0, 1]].assign(target=1, fold=4)
    cfg['model']['device'] = 'cuda'
    cfg['logreg'] = {'C': 10}
    result = training.load_or_compute_embeddings(reordered, cfg)
    np.testing.assert_array_equal(result, expected[[2, 0, 1]])
    assert len(calls) == 1
    assert not Path(cfg['data']['images_dir']).exists()


@pytest.mark.parametrize('relative', [True, False])
def test_configured_cache_path(cache_case, tmp_path, monkeypatch, relative):
    df, cfg, expected, calls = cache_case
    monkeypatch.chdir(tmp_path)
    path = Path('custom') / 'features.npz'
    cfg['data']['embeddings_cache'] = str(path if relative else tmp_path / path)
    training.load_or_compute_embeddings(df, cfg)
    assert (tmp_path / path).is_file()
    np.testing.assert_array_equal(training.load_or_compute_embeddings(df, cfg), expected)
    assert len(calls) == 1


@pytest.mark.parametrize('damage', ['keys', 'duplicates', 'shape', 'dtype', 'nan',
                                  'metadata', 'missing_array', 'truncated'])
def test_invalid_cache_rejected_without_recomputing(cache_case, damage):
    df, cfg, _, calls = cache_case
    training.load_or_compute_embeddings(df, cfg)
    path = Path(cfg['data']['output_dir']) / 'embeddings.npz'
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if damage == 'keys':
        arrays['image_name'][0] = 'z'
    elif damage == 'duplicates':
        arrays['image_name'][0] = 'b'
    elif damage == 'shape':
        arrays['features'] = arrays['features'][:, :100]
    elif damage == 'dtype':
        arrays['features'] = arrays['features'].astype(np.float64)
    elif damage == 'nan':
        arrays['features'][0, 0] = np.nan
    elif damage == 'metadata':
        arrays['metadata'] = np.array('{}')
    elif damage == 'missing_array':
        del arrays['image_name']
    np.savez_compressed(path, **arrays)
    if damage == 'truncated':
        path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(ValueError, match='Invalid embeddings cache:.*recompute-embeddings'):
        training.load_or_compute_embeddings(df, cfg)
    assert len(calls) == 1


def test_changed_dataset_rejected(cache_case):
    df, cfg, _, calls = cache_case
    training.load_or_compute_embeddings(df, cfg)
    with pytest.raises(ValueError, match='image_name keys'):
        training.load_or_compute_embeddings(df.iloc[:2], cfg)
    with pytest.raises(ValueError, match='unique string image_name'):
        training.load_or_compute_embeddings(pd.concat([df, df.iloc[[0]]]), cfg)
    assert len(calls) == 1


def test_custom_weights_checked_by_content(cache_case, tmp_path):
    df, cfg, _, calls = cache_case
    weights = tmp_path / 'weights.pth'
    weights.write_bytes(b'first weights')
    cfg['model']['weights_path'] = str(weights)
    training.load_or_compute_embeddings(df, cfg)
    moved = tmp_path / 'moved.pth'
    weights.rename(moved)
    cfg['model']['weights_path'] = str(moved)
    training.load_or_compute_embeddings(df, cfg)
    moved.write_bytes(b'other weights')
    with pytest.raises(ValueError, match='Weights, preprocessing'):
        training.load_or_compute_embeddings(df, cfg)
    assert len(calls) == 1


def test_changed_preprocessing_or_versions_rejected(cache_case, monkeypatch):
    df, cfg, _, calls = cache_case
    training.load_or_compute_embeddings(df, cfg)
    with monkeypatch.context() as context:
        context.setattr(training, 'version', lambda package: 'changed-version')
        with pytest.raises(ValueError, match='library versions differ'):
            training.load_or_compute_embeddings(df, cfg)
    # Changing the pinned weights also changes the selected transform preset.
    monkeypatch.setattr(training, 'RESNET_WEIGHTS', 'different-preset')
    with pytest.raises(ValueError, match='Weights, preprocessing'):
        training.load_or_compute_embeddings(df, cfg)
    assert len(calls) == 1


def test_force_recompute_and_failed_write_preserve_cache(cache_case, monkeypatch):
    df, cfg, expected, calls = cache_case
    training.load_or_compute_embeddings(df, cfg)
    expected += 1
    np.testing.assert_array_equal(
        training.load_or_compute_embeddings(df, cfg, recompute=True), expected)
    assert len(calls) == 2
    path = Path(cfg['data']['output_dir']) / 'embeddings.npz'
    original = path.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError('simulated disk failure')

    monkeypatch.setattr(training.np, 'savez_compressed', fail_write)
    with pytest.raises(OSError, match='disk failure'):
        training.load_or_compute_embeddings(df, cfg, recompute=True)
    assert path.read_bytes() == original
    assert list(path.parent.iterdir()) == [path]
    np.testing.assert_array_equal(training.load_or_compute_embeddings(df, cfg), expected)


def test_failed_extraction_preserves_existing_cache(cache_case, monkeypatch):
    df, cfg, _, _ = cache_case
    training.load_or_compute_embeddings(df, cfg)
    path = Path(cfg['data']['output_dir']) / 'embeddings.npz'
    original = path.read_bytes()
    monkeypatch.setattr(training, 'embeddings', lambda frame, config: np.zeros((3, 10)))
    with pytest.raises(ValueError, match='shape'):
        training.load_or_compute_embeddings(df, cfg, recompute=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize('recompute', [False, True])
def test_pipeline_uses_cache_before_training(cache_case, tmp_path, monkeypatch, recompute):
    df, cfg, _, calls = cache_case
    clean = tmp_path / 'clean.csv'
    df.to_csv(clean, index=False)
    cfg['data'].update(clean_csv=str(clean), report_dir=str(tmp_path / 'reports'))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(training, 'make_folds', lambda frame, config: frame.assign(fold=0))

    class StopBeforeTraining(Exception):
        pass

    def stop_at_cache(frame, config, recompute=False):
        assert frame.image_name.tolist() == df.image_name.tolist()
        assert config == cfg
        assert recompute == expected_recompute
        raise StopBeforeTraining

    expected_recompute = recompute
    monkeypatch.setattr(training, 'load_or_compute_embeddings', stop_at_cache)
    # only-folds must bypass both extraction and cache access.
    assert len(training.run(path, only_folds=True)) == len(df)
    with pytest.raises(StopBeforeTraining):
        training.run(path, recompute_embeddings=recompute)
    assert not calls
