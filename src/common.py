"""Small shared helpers for SIIM-ISIC 2020."""
from collections import defaultdict
from pathlib import Path
import hashlib
import pandas as pd
import yaml


def config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required = {'image_name', 'patient_id', 'target'}
    if missing := required - set(df.columns):
        raise ValueError(f'Missing columns: {missing}')
    if df.image_name.isna().any() or df.image_name.duplicated().any():
        raise ValueError('Missing or duplicated image_name')
    if df.patient_id.isna().any():
        raise ValueError('Missing patient_id; cannot split by patient')
    if df.target.isna().any() or not df.target.isin([0, 1]).all():
        raise ValueError('Target must be binary (0/1)')
    return df.sort_values('image_name').reset_index(drop=True)


def image_path(images_dir, name):
    name = str(name)
    if Path(name).name != name or '\\' in name or '/' in name:
        raise ValueError(f'Invalid image_name: {name}')
    return Path(images_dir) / (name if name.lower().endswith('.jpg') else f'{name}.jpg')


def find_duplicates(df, images_dir):
    """Find only identical file bytes. Hashes are never saved."""
    seen = defaultdict(list)
    for name in df.image_name:
        h = hashlib.sha256()
        with image_path(images_dir, name).open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        seen[h.digest()].append(name)
    return [names for names in seen.values() if len(names) > 1]


def duplicate_table(df, groups):
    mapping = {name: i for i, names in enumerate(groups, 1) for name in names}
    result = df[df.image_name.isin(mapping)].copy()
    result.insert(0, 'duplicate_group', result.image_name.map(mapping))
    return result[['duplicate_group', 'image_name', 'patient_id', 'target']]



def remove_exact_duplicates(df, groups):
    """Drop repeated files; ambiguous patient/label groups are dropped entirely."""
    by_name = df.set_index('image_name')
    dropped = {}
    for names in groups:
        rows = by_name.loc[names]
        if rows.target.nunique() > 1:
            dropped.update({name: 'conflicting_target' for name in names})
        elif rows.patient_id.nunique() > 1:
            dropped.update({name: 'different_patient' for name in names})
        else:
            dropped.update({name: 'same_patient_repeat' for name in names[1:]})
    excluded = df[df.image_name.isin(dropped)].copy()
    excluded['reason'] = excluded.image_name.map(dropped)
    cleaned = df[~df.image_name.isin(dropped)].copy().reset_index(drop=True)
    return cleaned, excluded
