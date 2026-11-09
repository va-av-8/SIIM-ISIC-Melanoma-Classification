from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from src.common import load_data, remove_exact_duplicates
from src.eda import run as eda
from src.train import run as train
from src.evaluate import evaluate


def setup(tmp_path):
    folder = tmp_path / 'jpeg' / 'train'
    folder.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(9)
    for patient in range(20):
        for i in range(4):
            name = f'ISIC_{patient:03}_{i}'
            target = int(patient % 3 == 0 and i == 0)
            rows.append(dict(image_name=name, patient_id=f'patient_{patient:03}', target=target,
                             sex='female' if patient % 2 else 'male', age_approx=40 + patient,
                             anatom_site_general_challenge='torso', diagnosis='melanoma' if target else 'nevus',
                             benign_malignant='malignant' if target else 'benign'))
            Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(folder / f'{name}.jpg')
    # Exact duplicate within one patient for audit.
    (folder / 'ISIC_001_3.jpg').write_bytes((folder / 'ISIC_001_2.jpg').read_bytes())
    train_csv = tmp_path / 'train.csv'
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    cfg = {'data': dict(labels_csv=str(train_csv), images_dir=str(folder),
                        eda_dir=str(tmp_path / 'reports' / 'eda'),
                        clean_csv=str(tmp_path / 'reports' / 'eda' / 'clean_train.csv'),
                        splits_csv=str(tmp_path / 'splits' / 'folds.csv'),
                        output_dir=str(tmp_path / 'outputs'), report_dir=str(tmp_path / 'reports' / 'baseline')),
           'eda': {'dark_pixel_threshold': 32},
           'validation': {'n_splits': 5, 'seed': 42},
           'model': {'device': 'cpu', 'batch_size': 8, 'num_workers': 0},
           'logreg': {'C': 1.0, 'solver': 'liblinear', 'l1_ratio': 0.0,
                      'max_iter': 2000, 'tol': 1e-4, 'class_weights': [None, 'balanced']}}
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_eda(tmp_path):
    path = setup(tmp_path)
    result = eda(path)
    assert result['n_images'] == 80
    assert result['n_exact_duplicate_groups'] == 1
    assert (tmp_path / 'reports' / 'eda' / 'by_age.csv').exists()
    assert (tmp_path / 'reports' / 'eda' / 'by_sex.csv').exists()
    assert (tmp_path / 'reports' / 'eda' / 'by_diagnosis.csv').exists()
    assert (tmp_path / 'reports' / 'eda' / 'figures' / 'contrast.png').exists()
    assert 'hash' not in (tmp_path / 'reports' / 'eda' / 'exact_duplicates.csv').read_text().lower()
    clean = pd.read_csv(tmp_path / 'reports' / 'eda' / 'clean_train.csv')
    assert len(clean) == 79
    assert 'ISIC_001_3' not in set(clean.image_name)
    assert not (tmp_path / 'splits' / 'folds.csv').exists()


def test_splits_and_evaluation(tmp_path):
    path = setup(tmp_path)
    eda(path)
    folds = train(path, only_folds=True)
    assert not folds.groupby('patient_id').fold.nunique().gt(1).any()
    assert len(folds.fold.unique()) == 5
    assert len(folds) == 79
    pd.testing.assert_frame_equal(folds, train(path, only_folds=True))
    out = tmp_path / 'outputs'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(4)
    oof = []
    for w in ('none', 'balanced'):
        f = folds.copy()
        f['score'] = np.clip(.1 + .6 * f.target + rng.normal(0, .1, len(f)), 0, 1)
        f['class_weight'] = w
        oof.append(f)
    pd.concat(oof).to_csv(out / 'oof_predictions.csv', index=False)
    result = evaluate(str(path))
    assert len(result['fold_metrics']) == 10


def test_duplicate_image_id_rejected(tmp_path):
    path = setup(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    csv = Path(cfg['data']['labels_csv'])
    df = pd.read_csv(csv)
    pd.concat([df, df.iloc[[0]]]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match='image_name'):
        load_data(csv)


def test_duplicate_cleaning_rules():
    # Same patient + same target: keep one; different patient or target: drop all.
    df = pd.DataFrame([
        ('a', 'P1', 0), ('b', 'P1', 0),
        ('c', 'P1', 0), ('d', 'P2', 0),
        ('e', 'P3', 0), ('f', 'P3', 1),
        ('g', 'P4', 1),
    ], columns=['image_name', 'patient_id', 'target'])
    cleaned, excluded = remove_exact_duplicates(df, [['a', 'b'], ['c', 'd'], ['e', 'f']])
    assert cleaned.image_name.tolist() == ['a', 'g']
    assert dict(zip(excluded.image_name, excluded.reason)) == {
        'b': 'same_patient_repeat', 'c': 'different_patient',
        'd': 'different_patient', 'e': 'conflicting_target',
        'f': 'conflicting_target',
    }


def test_training_requires_eda(tmp_path):
    path = setup(tmp_path)
    with pytest.raises(FileNotFoundError, match='Run src.eda first'):
        train(path, only_folds=True)
