"""Independent numerical checks of the saved study, without retraining."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text())
    for name, item in manifest.items():
        path = ROOT / "data/raw" / f"{name}.geojson.gz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["gzip_sha256"]
        raw = gzip.decompress(path.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == item["raw_sha256"]
        assert len(json.loads(raw)["features"]) == item["features"]
    data = pd.read_csv(ROOT / "data/processed/streets.csv")
    pred = pd.read_csv(ROOT / "results/predictions.csv")
    metrics = pd.read_csv(ROOT / "results/model_metrics.csv")
    selected = pd.read_csv(ROOT / "results/selected_streets_10pct.csv")
    policy = pd.read_csv(ROOT / "results/policy_summary.csv")
    config = json.loads((ROOT / "results/run_config.json").read_text())
    quality = json.loads((ROOT / "results/data_quality.json").read_text())
    assert hashlib.sha256((ROOT / "analysis.py").read_bytes()).hexdigest() == config["source_hash"]
    assert pred.street_edge_id.is_unique
    assert set(pred.street_edge_id) == set(data.loc[data.cohort, "street_edge_id"])
    assert len(pred) == quality["cohort_streets"]
    assert pred.target.sum() == quality["cohort_positive"]
    assert set(config["features"]).isdisjoint({"target", "record_count", "audit_count", "region_id"})
    coords = pred[["x_m", "y_m"]].to_numpy()
    saved_checks = pd.read_csv(ROOT / "results/split_checks.csv").set_index("fold")
    for fold in sorted(pred.spatial_fold.unique()):
        test = pred.spatial_fold.to_numpy() == fold
        distances = cKDTree(coords[test]).query(coords[~test])[0]
        train = distances >= config["spatial_buffer_m"]
        assert train.sum() == saved_checks.loc[fold, "train"]
        assert test.sum() == saved_checks.loc[fold, "test"]
        assert (distances[train] >= config["spatial_buffer_m"]).all()
    for _, record in metrics.loc[metrics.fold == "pooled"].iterrows():
        p = pred[f"{record['split']}_{record['model']}"]
        assert p.between(0, 1).all()
        assert np.isclose(average_precision_score(pred.target, p), record.ap)
        assert np.isclose(roc_auc_score(pred.target, p), record.roc_auc)
        assert np.isclose(brier_score_loss(pred.target, p), record.brier)
    for strategy, part in selected.groupby("strategy"):
        assert part.street_edge_id.is_unique
        merged = part.merge(pred, on="street_edge_id", validate="one_to_one")
        assert (merged.fold == merged.spatial_fold).all()
        row = policy.loc[(policy.strategy == strategy) & np.isclose(policy.fraction, .1)].iloc[0]
        assert len(merged) == row.budget
        assert merged.target.sum() == row.mean_found
        assert np.isclose(merged.target.mean(), row.precision)
        assert np.isclose(merged.length_m.mean(), row.mean_selected_length_m)
    assert policy.precision.between(0, 1).all()
    assert policy.macro_region_coverage.between(0, 1).all()
    for strategy in policy.strategy.unique():
        choices = policy.loc[policy.strategy == strategy].sort_values("fraction")
        assert np.all(np.diff(choices.budget) > 0)
        assert np.all(np.diff(choices.mean_found) >= 0)
    for name in ("model_and_policy.png", "study_map.png", "feature_dependence.png"):
        assert (ROOT / "results" / name).stat().st_size > 10000
    print("PASS: raw-data hashes and feature counts")
    print("PASS: cohort, label counts, prohibited-feature exclusion")
    print(f"PASS: spatial holdout membership and {config['spatial_buffer_m']} m buffer")
    print("PASS: independent recomputation of pooled model metrics")
    print("PASS: each 10% strategy has the same number of unique held-out streets")
    print("PASS: selected-record totals match the published policy table")
    print("PASS: probability bounds, budgets and plot artifacts")


if __name__ == "__main__":
    main()
