"""Frozen ResNet-18 embeddings + 5-fold patient-safe logistic regression."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.common import config, load_data, image_path


def make_folds(df, cfg):
    path = Path(cfg['data']['splits_csv'])
    if path.exists():
        old = pd.read_csv(path)
        if old.image_name.duplicated().any() or set(old.image_name) != set(df.image_name):
            raise ValueError('Saved folds do not match the current data')
        for col in ('patient_id', 'target'):
            if not (old.set_index('image_name').loc[df.image_name, col].to_numpy() == df[col].to_numpy()).all():
                raise ValueError(f'Saved folds have different {col} values')
        folds = old.set_index('image_name').loc[df.image_name, 'fold'].to_numpy()
    else:
        splitter = StratifiedGroupKFold(n_splits=int(cfg['validation']['n_splits']),
                                        shuffle=True, random_state=int(cfg['validation']['seed']))
        folds = np.full(len(df), -1)
        for i, (_, val) in enumerate(splitter.split(df, df.target, groups=df.patient_id)):
            folds[val] = i
    result = df[['image_name', 'patient_id', 'target']].copy()
    result['fold'] = folds
    if (result.fold == -1).any() or len(set(folds)) != cfg['validation']['n_splits']:
        raise ValueError('Invalid fold assignment')
    if result.groupby('patient_id').fold.nunique().gt(1).any():
        raise ValueError('Patient leakage between folds')
    for i, part in result.groupby('fold'):
        if part.target.nunique() < 2:
            raise ValueError(f'Fold {i} has only one class')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result.to_csv(path, index=False)
    return result


class Images:
    def __init__(self, df, folder, transform):
        self.names, self.folder, self.transform = df.image_name.tolist(), folder, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(image_path(self.folder, self.names[i])) as img:
            return self.transform(img.convert('RGB')), i


def embeddings(df, cfg):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18, ResNet18_Weights

    device_name = cfg['model']['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    device = torch.device(device_name)
    weights_path = cfg['model'].get('weights_path')
    if weights_path:
        model = resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    dataset = Images(df, cfg['data']['images_dir'], ResNet18_Weights.DEFAULT.transforms())
    loader = DataLoader(dataset, batch_size=cfg['model']['batch_size'], shuffle=False,
                        num_workers=cfg['model']['num_workers'])
    features = np.empty((len(df), 512), dtype=np.float32)
    with torch.inference_mode():
        for batch, (x, indices) in enumerate(loader, 1):
            features[indices.numpy()] = model(x.to(device)).cpu().numpy()
            if batch % 50 == 0:
                print(f'Embedding batch {batch}/{len(loader)}')
    return features


def run(path, only_folds=False):
    cfg = config(path)
    clean_path = Path(cfg['data']['clean_csv'])
    if not clean_path.exists():
        raise FileNotFoundError(f'Cleaned dataset not found: {clean_path}. Run src.eda first.')
    df = load_data(clean_path)
    folds = make_folds(df, cfg)
    report = Path(cfg['data']['report_dir'])
    report.mkdir(parents=True, exist_ok=True)
    balance = folds.groupby('fold').agg(n_images=('target', 'size'), n_positive=('target', 'sum'), n_patients=('patient_id', 'nunique'))
    balance['prevalence'] = balance.n_positive / balance.n_images
    balance.to_csv(report / 'fold_balance.csv')
    print(balance)
    if only_folds:
        return folds
    X, y = embeddings(folds, cfg), folds.target.to_numpy()
    output = Path(cfg['data']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    for class_weight in cfg['logreg']['class_weights']:
        weight = 'none' if class_weight is None else class_weight
        for fold in sorted(folds.fold.unique()):
            val = folds.fold.to_numpy() == fold
            clf = make_pipeline(StandardScaler(), LogisticRegression(
                C=cfg['logreg']['C'],
                solver=cfg['logreg']['solver'],
                l1_ratio=cfg['logreg']['l1_ratio'],
                max_iter=cfg['logreg']['max_iter'],
                tol=cfg['logreg']['tol'],
                class_weight=class_weight,
                random_state=cfg['validation']['seed']))
            clf.fit(X[~val], y[~val])
            piece = folds.loc[val].copy()
            piece['score'] = clf.predict_proba(X[val])[:, 1]
            piece['class_weight'] = weight
            predictions.append(piece)
            joblib.dump(clf, output / f'lr_{weight}_fold{fold}.joblib')
    pd.concat(predictions, ignore_index=True).to_csv(output / 'oof_predictions.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baseline.yaml')
    p.add_argument('--only-folds', action='store_true')
    args = p.parse_args()
    run(args.config, args.only_folds)
