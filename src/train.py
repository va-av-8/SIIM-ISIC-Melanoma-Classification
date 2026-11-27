"""Frozen ResNet-18 embeddings + 5-fold patient-safe logistic regression."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.common import config, load_data, image_path


# Explicit equivalent of the original ResNet-18 DEFAULT weights.
RESNET_WEIGHTS = 'IMAGENET1K_V1'
METADATA = ['age_approx', 'sex', 'anatom_site_general_challenge']
VARIANTS = {
    'B0': [], 'B1': [], 'M-full': METADATA,
    'A-age': METADATA[1:], 'A-sex': [METADATA[0], METADATA[2]],
    'A-site': METADATA[:2],
}


def variant_spec(variant):
    return {
        'stage': 'baseline' if variant in ('B0', 'B1') else
                 'hypothesis' if variant == 'M-full' else 'ablation',
        'features': VARIANTS[variant],
        'class_weight': 'balanced' if variant == 'B1' else None,
    }


def generate_folds(df, cfg, split_seed=None):
    """Generate assignments in current row order, without reading/writing saved folds."""
    splitter = StratifiedGroupKFold(
        n_splits=int(cfg['validation']['n_splits']), shuffle=True,
        random_state=int(cfg['validation']['seed'] if split_seed is None else split_seed))
    folds = np.full(len(df), -1)
    for i, (_, val) in enumerate(splitter.split(df, df.target, groups=df.patient_id)):
        folds[val] = i
    result = df[['image_name', 'patient_id', 'target']].copy()
    result['fold'] = folds
    return result


def make_folds(df, cfg):
    path = Path(cfg['data']['splits_csv'])
    if path.exists():
        old = pd.read_csv(path)
        if not {'image_name', 'patient_id', 'target', 'fold'}.issubset(old.columns):
            raise ValueError('Saved folds lack required columns')
        if old.image_name.duplicated().any() or set(old.image_name) != set(df.image_name):
            raise ValueError('Saved folds do not match the current data')
        for col in ('patient_id', 'target'):
            if not (old.set_index('image_name').loc[df.image_name, col].to_numpy() == df[col].to_numpy()).all():
                raise ValueError(f'Saved folds have different {col} values')
        folds = old.set_index('image_name').loc[df.image_name, 'fold'].to_numpy()
    else:
        folds = generate_folds(df, cfg).fold.to_numpy()
    result = df[['image_name', 'patient_id', 'target']].copy()
    result['fold'] = folds
    if (not result.fold.isin(range(cfg['validation']['n_splits'])).all()
            or len(set(folds)) != cfg['validation']['n_splits']):
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
        model = resnet18(weights=ResNet18_Weights[RESNET_WEIGHTS])
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    dataset = Images(df, cfg['data']['images_dir'], ResNet18_Weights[RESNET_WEIGHTS].transforms())
    loader = DataLoader(dataset, batch_size=cfg['model']['batch_size'], shuffle=False,
                        num_workers=cfg['model']['num_workers'])
    features = np.empty((len(df), 512), dtype=np.float32)
    with torch.inference_mode():
        for batch, (x, indices) in enumerate(loader, 1):
            features[indices.numpy()] = model(x.to(device)).cpu().numpy()
            if batch % 50 == 0:
                print(f'Embedding batch {batch}/{len(loader)}')
    return features


def embedding_signature(cfg):
    """Invalidate cached features when weights, transforms or extraction libraries change."""
    weights = f'ResNet18_Weights.{RESNET_WEIGHTS}'
    if weights_path := cfg['model'].get('weights_path'):
        digest = hashlib.sha256()
        with Path(weights_path).open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        weights = f'sha256:{digest.hexdigest()}'
    libraries = {}
    for package in ('torch', 'torchvision', 'Pillow'):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    # Bump format_version when changing feature extraction beyond the fields below.
    return dict(format_version=1, backbone='resnet18; fc=Identity; eval', weights=weights,
                preprocessing=f'RGB; ResNet18_Weights.{RESNET_WEIGHTS}.transforms()',
                libraries=libraries)


def load_or_compute_embeddings(df, cfg, recompute=False):
    """Cache raw frozen features, never the fold-specific StandardScaler output."""
    names = df.image_name
    if (names.empty or names.isna().any() or names.duplicated().any()
            or not names.map(lambda name: isinstance(name, str) and bool(name)).all()):
        raise ValueError('Embeddings require nonempty, unique string image_name values')
    names = names.to_numpy(dtype=str)
    path = Path(cfg['data'].get('embeddings_cache',
                               Path(cfg['data']['output_dir']) / 'embeddings.npz'))
    signature = embedding_signature(cfg)

    def validate_features(features, n):
        if (features.shape != (n, 512) or features.dtype != np.dtype('float32')
                or not np.isfinite(features).all()):
            raise ValueError('Expected finite float32 embeddings with shape (n_images, 512)')

    if path.exists() and not recompute:
        try:
            with np.load(path, allow_pickle=False) as saved:
                cached_names = saved['image_name']
                features = saved['features']
                metadata = json.loads(saved['metadata'].item())
            if metadata != signature:
                raise ValueError('Weights, preprocessing or extraction library versions differ')
            if (cached_names.ndim != 1 or cached_names.dtype.kind != 'U'
                    or len(set(cached_names)) != len(cached_names)
                    or set(cached_names) != set(names)):
                raise ValueError('Cached image_name keys do not match the current dataset')
            validate_features(features, len(cached_names))
            # Cache row order is independent of fold assignment and current CSV order.
            order = pd.Index(cached_names).get_indexer(names)
            print(f'Loaded embeddings: {path} ({len(names)} images)')
            return features[order]
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
            raise ValueError(f'Invalid embeddings cache: {path}. {exc}. '
                             'Use --recompute-embeddings to rebuild it from JPEGs.') from exc

    features = embeddings(df, cfg)
    validate_features(features, len(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish one complete archive; a failed extraction/write leaves the old cache intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as output:
            temporary = Path(output.name)
            np.savez_compressed(output, image_name=names, features=features,
                                metadata=np.array(json.dumps(signature, sort_keys=True)))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Saved embeddings: {path} ({len(names)} images)')
    return features


def make_logreg(cfg, class_weight=None):
    return LogisticRegression(
        **{key: cfg['logreg'][key] for key in ('C', 'solver', 'l1_ratio', 'max_iter', 'tol')},
        class_weight=class_weight, random_state=cfg['validation']['seed'])


def experiment_features(features, data, variant):
    """Select only approved metadata; all learned transforms stay inside CV."""
    columns = VARIANTS[variant]
    if not columns:
        return features, StandardScaler()
    if missing := set(columns) - set(data.columns):
        raise ValueError(f'Missing metadata for {variant}: {sorted(missing)}')
    image_columns = [f'emb_{i:03}' for i in range(512)]
    frame = pd.DataFrame(features, columns=image_columns)
    for column in columns:
        values = data[column].reset_index(drop=True)
        if column == 'age_approx':
            values = pd.to_numeric(values, errors='raise').astype(float)
            if np.isinf(values).any():
                raise ValueError('Age must be finite or missing')
        else:
            values = values.astype(object).where(values.notna(), np.nan)
        frame[column] = values
    transforms = [('image', StandardScaler(), image_columns)]
    if 'age_approx' in columns:
        transforms.append(('age', make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()),
            ['age_approx']))
    categories = [column for column in columns if column != 'age_approx']
    if categories:
        transforms.append(('categorical', make_pipeline(
            SimpleImputer(strategy='constant', fill_value='MISSING', keep_empty_features=True),
            OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categories))
    return frame, ColumnTransformer(transforms, remainder='drop')


def fit_oof(folds, features, cfg, output=None, class_weight=None, preprocessor=None,
            train_masks=None):
    """Shared CV loop; optionally save models for the standalone training command."""
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    y = folds.target.to_numpy()
    weight = 'none' if class_weight is None else class_weight
    predictions = []
    for fold in sorted(folds.fold.unique()):
        val = folds.fold.to_numpy() == fold
        train = ~val if train_masks is None else np.asarray(train_masks[fold])
        if train.shape != val.shape or train.dtype != bool or (train & val).any():
            raise ValueError('Train mask must be boolean and exclude validation rows')
        if set(folds.loc[train, 'patient_id']) & set(folds.loc[val, 'patient_id']):
            raise ValueError('Patient leakage between train subset and validation')
        if len(np.unique(y[train])) != 2:
            raise ValueError(f'Fold {fold}: training subset must contain both classes')
        transform = StandardScaler() if preprocessor is None else clone(preprocessor)
        clf = make_pipeline(transform, make_logreg(cfg, class_weight))
        train_x = features.iloc[train] if isinstance(features, pd.DataFrame) else features[train]
        val_x = features.iloc[val] if isinstance(features, pd.DataFrame) else features[val]
        clf.fit(train_x, y[train])
        piece = folds.loc[val].copy()
        piece['score'] = clf.predict_proba(val_x)[:, 1]
        piece['class_weight'] = weight
        predictions.append(piece)
        if output is not None:
            joblib.dump(clf, output / f'lr_{weight}_fold{fold}.joblib')
    return pd.concat(predictions, ignore_index=True)


def save_fold_balance(folds, report):
    report = Path(report)
    report.mkdir(parents=True, exist_ok=True)
    balance = folds.groupby('fold').agg(n_images=('target', 'size'), n_positive=('target', 'sum'),
                                       n_patients=('patient_id', 'nunique'))
    balance['prevalence'] = balance.n_positive / balance.n_images
    balance.to_csv(report / 'fold_balance.csv')
    return balance


def run(path, only_folds=False, recompute_embeddings=False):
    cfg = config(path)
    clean_path = Path(cfg['data']['clean_csv'])
    if not clean_path.exists():
        raise FileNotFoundError(f'Cleaned dataset not found: {clean_path}. Run src.eda first.')
    df = load_data(clean_path)
    folds = make_folds(df, cfg)
    print(save_fold_balance(folds, cfg['data']['report_dir']))
    if only_folds:
        return folds
    X = load_or_compute_embeddings(folds, cfg, recompute=recompute_embeddings)
    output = Path(cfg['data']['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    for class_weight in cfg['logreg']['class_weights']:
        predictions.append(fit_oof(folds, X, cfg, output, class_weight))
    pd.concat(predictions, ignore_index=True).to_csv(output / 'oof_predictions.csv', index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baseline.yaml')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--only-folds', action='store_true')
    mode.add_argument('--recompute-embeddings', action='store_true',
                      help='Rebuild the embeddings cache from JPEGs before training')
    args = p.parse_args()
    run(args.config, args.only_folds, args.recompute_embeddings)
