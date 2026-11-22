import numpy as np
import pandas as pd
import pytest
import yaml

from src.evaluate import evaluate_resolutions, merge_b0_resolutions, resolution_metrics


@pytest.fixture
def inputs():
    b0 = pd.DataFrame({
        "image_name": list("abcdef"), "target": [1, 0, 1, 0, 0, 0],
        "score": [.9, .8, .7, .1, .8, .2], "class_weight": "none",
    })
    b1 = b0.assign(class_weight="balanced", score=1 - b0.score)
    stats = pd.DataFrame({
        "image_name": list("fedcba"), "width": [20, 20, 10, 10, 10, 10],
        "height": [10, 10, 20, 20, 20, 20], "target": [0, 0, 0, 1, 0, 1],
    })
    return pd.concat([b0, b1], ignore_index=True), stats


def test_metrics_and_b0_filter(inputs):
    oof, stats = inputs
    merged = merge_b0_resolutions(oof, stats)
    assert len(merged) == 6
    assert merged.score.tolist() == [.9, .8, .7, .1, .8, .2]
    result = resolution_metrics(merged, top_n=1, min_positive=2)
    row = result.iloc[0]
    assert row.resolution == "10×20"  # Preserve orientation; do not pool 20×10.
    assert row.n == 4 and row.n_positive == 2
    assert row.prevalence == .5
    assert row.average_precision == pytest.approx((1 + 2 / 3) / 2)
    assert row.ap_over_prevalence == pytest.approx(5 / 3)
    assert result.shown_in_plot.tolist() == [True, False]
    zero = result.iloc[1]
    assert zero.n == 2 and zero.n_positive == 0 and zero.prevalence == 0
    assert np.isnan(zero.average_precision) and np.isnan(zero.ap_over_prevalence)
    assert not resolution_metrics(merged, min_positive=3).shown_in_plot.any()
    with pytest.raises(ValueError, match="No B0"):
        merge_b0_resolutions(oof.loc[oof.class_weight.eq("balanced")], stats)


@pytest.mark.parametrize("side", ["oof", "stats"])
def test_duplicate_keys(inputs, side):
    oof, stats = inputs
    if side == "oof":
        oof = pd.concat([oof, oof.iloc[[0]]])
    else:
        stats = pd.concat([stats, stats.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicated image_name"):
        merge_b0_resolutions(oof, stats)


@pytest.mark.parametrize("side", ["oof", "stats"])
def test_mismatched_keys(inputs, side):
    oof, stats = inputs
    if side == "oof":
        oof = oof.loc[oof.image_name.ne("a")]
    else:
        stats = stats.loc[stats.image_name.ne("a")]
    with pytest.raises(ValueError, match="Image sets do not match"):
        merge_b0_resolutions(oof, stats)


def test_only_audited_exclusions_allowed(inputs):
    oof, stats = inputs
    extra = stats.iloc[[0]].assign(image_name="excluded")
    full_stats = pd.concat([stats, extra])
    result = merge_b0_resolutions(oof, full_stats, extra[["image_name"]])
    assert set(result.image_name) == set(oof.image_name)
    with pytest.raises(ValueError, match="differ from excluded_images"):
        merge_b0_resolutions(oof, full_stats, pd.DataFrame({"image_name": ["other"]}))


@pytest.mark.parametrize("column,value,message", [
    ("width", 0, "positive integers"), ("height", 1.5, "positive integers"),
    ("width", np.nan, "positive integers"), ("target", 1, "different target"),
    ("image_name", None, "image_name"),
])
def test_invalid_stats(inputs, column, value, message):
    oof, stats = inputs
    stats.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        merge_b0_resolutions(oof, stats)


def test_plot_selection_orders_by_size_then_resolution():
    data = pd.DataFrame({"width": [30] * 4 + [20] * 3 + [10] * 3,
                         "height": 10, "target": [0] * 4 + [1] * 6, "score": .5})
    result = resolution_metrics(data, top_n=1, min_positive=2)
    assert result.width.tolist() == [30, 10, 20]
    assert result.shown_in_plot.tolist() == [False, True, False]
    assert result.iloc[1].average_precision == 1  # All-positive group is defined.


@pytest.mark.parametrize("min_positive", [2, 10])
def test_csv_only_run_and_missing_inputs(tmp_path, inputs, min_positive):
    oof, stats = inputs
    # No JPEGs, embeddings, folds, labels CSV or model configuration are present.
    oof.to_csv(tmp_path / "oof_predictions.csv", index=False)
    stats.to_csv(tmp_path / "image_statistics.csv", index=False)
    cfg = {"data": {"output_dir": str(tmp_path), "eda_dir": str(tmp_path),
                    "report_dir": str(tmp_path / "reports"), "figure_dir": str(tmp_path / "figures")},
           "resolution_analysis": {"top_n": 1, "min_positive": min_positive}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = evaluate_resolutions(path)
    saved = pd.read_csv(tmp_path / "reports" / "resolution_metrics.csv")
    pd.testing.assert_frame_equal(saved, result)
    assert (tmp_path / "figures" / "resolution_analysis.png").read_bytes().startswith(b"\x89PNG")
    (tmp_path / "image_statistics.csv").unlink()
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "image_statistics.csv")):
        evaluate_resolutions(path)
    extra = stats.iloc[[0]].assign(image_name="excluded")
    pd.concat([stats, extra]).to_csv(tmp_path / "image_statistics.csv", index=False)
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "excluded_images.csv")):
        evaluate_resolutions(path)
    extra[["image_name"]].to_csv(tmp_path / "excluded_images.csv", index=False)
    pd.testing.assert_frame_equal(evaluate_resolutions(path), result)
    (tmp_path / "oof_predictions.csv").unlink()
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "oof_predictions.csv")):
        evaluate_resolutions(path)
