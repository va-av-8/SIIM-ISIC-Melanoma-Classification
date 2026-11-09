"""EDA using Kaggle train.csv and ORIGINAL JPEGs; no splitting or preprocessing."""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageStat
from tqdm import tqdm
from src.common import config, load_data, image_path, find_duplicates, duplicate_table, remove_exact_duplicates


def plot_hist(df, col, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in (0, 1):
        values = df.loc[df.target == label, col].dropna()
        if len(values):
            ax.hist(values, bins=35, density=True, alpha=.5, label=f'target={label}, n={len(values)}')
    ax.set(xlabel=col, ylabel='Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run(path):
    cfg = config(path)
    df = load_data(cfg['data']['labels_csv'])
    clean_path = Path(cfg['data']['clean_csv'])
    clean_path.unlink(missing_ok=True)  # Never reuse stale cleaned data if EDA fails.
    out = Path(cfg['data']['eda_dir'])
    figs = out / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    df.groupby('target').size().rename('n_images').to_csv(out / 'class_counts.csv')
    patient = df.groupby('patient_id').agg(n_images=('target', 'size'), n_positive=('target', 'sum'))
    patient.to_csv(out / 'patients.csv')
    missing = df.isna().sum().rename('n_missing').to_frame()
    missing['fraction'] = missing.n_missing / len(df)
    missing.to_csv(out / 'missing.csv')
    pd.DataFrame({label: group.isna().sum() for label, group in df.groupby('target')}).T.to_csv(out / 'missing_by_class.csv')

    # Every category available in Kaggle train.csv, including diagnostic fields,
    # is analysed but only image pixels are used by the baseline model.
    for col in ('sex', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant'):
        if col in df:
            group = df.assign(**{col: df[col].fillna('MISSING')}).groupby(col)['target'].agg(['size', 'sum', 'mean'])
            group.columns = ['n_images', 'n_positive', 'prevalence']
            group.to_csv(out / f'by_{col}.csv')
            if col in ('sex', 'anatom_site_general_challenge'):
                fig, ax = plt.subplots(figsize=(7, 4))
                group.prevalence.plot.bar(ax=ax)
                ax.set(ylabel='Melanoma fraction', xlabel=col)
                fig.tight_layout()
                fig.savefig(figs / f'by_{col}.png')
                plt.close(fig)
    if 'age_approx' in df:
        age = pd.to_numeric(df.age_approx, errors='coerce')
        bins = pd.cut(age, [0, 20, 30, 40, 50, 60, 70, 80, 120], include_lowest=True)
        counts = df.groupby(bins, observed=True)['target'].agg(['size', 'sum', 'mean'])
        counts.columns = ['n_images', 'n_positive', 'prevalence']
        counts.to_csv(out / 'by_age.csv')
        fig, ax = plt.subplots(figsize=(7, 4))
        counts.prevalence.plot.bar(ax=ax)
        ax.set(ylabel='Melanoma fraction', xlabel='Approximate age')
        fig.tight_layout()
        fig.savefig(figs / 'by_age.png')
        plt.close(fig)

    # Exact hashes only, never perceptual hashes or near-duplicate search.
    images_dir = Path(cfg['data']['images_dir'])
    expected = {image_path(images_dir, name).name for name in df.image_name}
    actual = {p.name for p in images_dir.glob('*.jpg')}
    pd.DataFrame({'image_name': sorted(actual - expected)}).to_csv(out / 'unreferenced_images.csv', index=False)
    pd.DataFrame({'image_name': sorted(expected - actual)}).to_csv(out / 'missing_images.csv', index=False)
    groups = find_duplicates(df, images_dir, progress=True)
    duplicate_table(df, groups).to_csv(out / 'exact_duplicates.csv', index=False)
    cleaned, excluded = remove_exact_duplicates(df, groups)
    excluded.to_csv(out / 'excluded_images.csv', index=False)

    records = []
    threshold = int(cfg['eda']['dark_pixel_threshold'])
    for name in tqdm(
        df.image_name, total=len(df), desc='Image stats', unit='img',
        file=sys.stdout, mininterval=1.0,
    ):
        with Image.open(image_path(cfg['data']['images_dir'], name)) as img:
            width, height = img.size
            gray = img.convert('L')
            stats = ImageStat.Stat(gray)
            records.append(dict(image_name=name, width=width, height=height,
                                brightness=stats.mean[0], contrast=stats.stddev[0],
                                dark_fraction=sum(gray.histogram()[:threshold]) / (width * height)))
    img_df = pd.DataFrame(records).merge(df[['image_name', 'target']], on='image_name', validate='one_to_one')
    img_df.to_csv(out / 'image_statistics.csv', index=False)
    for col in ('brightness', 'contrast', 'dark_fraction'):
        plot_hist(img_df, col, figs / f'{col}.png')
    plot_hist(patient.reset_index().assign(target=(patient.n_positive > 0).astype(int).to_numpy()), 'n_images', figs / 'images_per_patient.png')
    summary = dict(n_images=len(df), n_patients=df.patient_id.nunique(),
                   n_positive=int(df.target.sum()), prevalence=float(df.target.mean()),
                   n_exact_duplicate_groups=len(groups), n_images_in_duplicate_groups=sum(map(len, groups)),
                   n_excluded=len(excluded), n_clean=len(cleaned),
                   n_positive_clean=int(cleaned.target.sum()),
                   n_missing_images=len(expected - actual), n_unreferenced_images=len(actual - expected))
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/baseline.yaml')
    run(p.parse_args().config)
