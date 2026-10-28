"""Public-data learning study: predicting recorded sidewalk barriers.

Run: python analysis.py
No street imagery, travel trajectories, population estimates or repair-effect
assumptions are used. The target is an observed label, not physical ground truth.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
RAW = ROOT / "data" / "raw"
CATEGORIES = ["health", "transit", "grocery", "park", "school", "community", "library"]
NUMERIC = ["length_m", "sinuosity", "end_degree_mean"]
NUMERIC += [f"{category}_within_400m" for category in CATEGORIES]
NUMERIC += ["nearest_health_m"]
FEATURES = NUMERIC + ["way_type"]
NEGATIVE_TYPES = {"NoCurbRamp", "Obstacle", "SurfaceProblem", "NoSidewalk"}
SEED = 20261026
BUFFER_M = 500


def read_raw(name):
    with gzip.open(RAW / f"{name}.geojson.gz", "rt") as stream:
        return json.load(stream)["features"]


def xy(coordinates):
    """Local equirectangular coordinates in meters, for city-scale proximity."""
    a = np.asarray(coordinates, dtype=float)
    x = np.radians(a[..., 0] + 122.33) * 6371008.8 * np.cos(np.radians(47.61))
    y = np.radians(a[..., 1] - 47.61) * 6371008.8
    return np.stack((x, y), axis=-1)


def qualifies(p, min_agree=2, min_severity=2):
    """Predeclared record definition; never interpret missing records as safe."""
    severity = p.get("median_severity")
    return (
        p["label_type"] in NEGATIVE_TYPES
        and (p["label_type"] == "NoSidewalk" or (severity or 0) >= min_severity)
        and p["agree_count"] >= min_agree
        and p["agree_count"] > p["disagree_count"]
    )


def prepare():
    streets, clusters, places = [read_raw(x) for x in ("streets", "clusters", "places")]
    street_ids = [x["properties"]["street_edge_id"] for x in streets]
    cluster_ids = [x["properties"]["label_cluster_id"] for x in clusters]
    assert len(street_ids) == len(set(street_ids)), "Duplicate street ids"
    assert len(cluster_ids) == len(set(cluster_ids)), "Duplicate cluster ids"
    current = [
        x for x in streets
        if x["properties"]["status"] == "open" and not x["properties"].get("outdated", False)
    ]
    degrees = {}
    for feature in current:
        points = feature["geometry"]["coordinates"]
        for coord in (points[0], points[-1]):
            key = tuple(round(v, 7) for v in coord)
            degrees[key] = degrees.get(key, 0) + 1

    barriers = [x for x in clusters if qualifies(x["properties"])]
    counts = {}
    for feature in barriers:
        key = feature["properties"]["street_edge_id"]
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for feature in current:
        p = feature["properties"]
        coords = np.array(feature["geometry"]["coordinates"], dtype=float)
        projected = xy(coords)
        segment_lengths = np.linalg.norm(np.diff(projected, axis=0), axis=1)
        length = segment_lengths.sum()
        if length < 1:
            continue
        # Arc-length midpoint; not the mean of unequally spaced shape vertices.
        cumulative = np.r_[0, np.cumsum(segment_lengths)]
        midpoint = np.array([np.interp(length / 2, cumulative, projected[:, j]) for j in (0, 1)])
        straight = np.linalg.norm(projected[-1] - projected[0])
        keys = [tuple(round(float(v), 7) for v in point) for point in (coords[0], coords[-1])]
        rows.append({
            "street_edge_id": p["street_edge_id"],
            "region_id": p["region_id"],
            "region_name": p["region_name"],
            "way_type": p["way_type"],
            "length_m": length,
            "sinuosity": min(length / max(straight, 1), 10),
            "end_degree_mean": np.mean([degrees[k] for k in keys]),
            "x_m": midpoint[0], "y_m": midpoint[1],
            "audit_count": p["audit_count"],
            "record_count": counts.get(p["street_edge_id"], 0),
            "target": int(p["street_edge_id"] in counts),
        })
    frame = pd.DataFrame(rows)
    for category in CATEGORIES:
        points = [x["geometry"]["coordinates"] for x in places if x["properties"]["category"] == category]
        if not points:
            raise ValueError(f"No {category} records; cannot silently impute zero coverage.")
        tree = cKDTree(xy(points))
        frame[f"{category}_within_400m"] = tree.query_ball_point(
            frame[["x_m", "y_m"]].to_numpy(), r=400, return_length=True
        )
        if category == "health":
            frame["nearest_health_m"] = tree.query(frame[["x_m", "y_m"]].to_numpy())[0]
    frame["cohort"] = frame.audit_count >= 2
    assert not frame[FEATURES].isna().any().any()
    assert not frame.street_edge_id.duplicated().any()
    assert set(FEATURES).isdisjoint({
        "target", "record_count", "audit_count", "region_id", "region_name", "street_edge_id", "x_m", "y_m"
    })
    cohort_ids = set(frame.loc[frame.cohort, "street_edge_id"])
    eligible_barriers = [x for x in barriers if x["properties"]["street_edge_id"] in cohort_ids]
    image_years = pd.Series([
        (x["properties"].get("avg_image_capture_date") or "Unknown")[:4] for x in eligible_barriers
    ]).value_counts().sort_index()
    image_years.rename_axis("image_year").rename("clusters").to_csv(OUT / "image_years.csv")
    sensitivity = []
    for agree in (1, 2, 3):
        ids = {x["properties"]["street_edge_id"] for x in clusters if qualifies(x["properties"], agree)}
        for audits in (1, 2, 3):
            subset = frame.loc[frame.audit_count >= audits]
            positive = subset.street_edge_id.isin(ids).sum()
            sensitivity.append({
                "min_agree": agree, "min_audits": audits,
                "streets": len(subset), "positive": int(positive),
                "record_prevalence": positive / len(subset),
            })
    pd.DataFrame(sensitivity).to_csv(OUT / "definition_sensitivity.csv", index=False)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    frame.to_csv(ROOT / "data" / "processed" / "streets.csv", index=False)
    raw_known = set(street_ids)
    quality = {
        "raw_streets": len(streets), "raw_clusters": len(clusters), "raw_places": len(places),
        "open_current_streets_before_length_filter": len(current),
        "analysis_current_streets": len(frame),
        "excluded_under_1m": len(current) - len(frame),
        "cohort_streets": int(frame.cohort.sum()),
        "cohort_positive": int(frame.loc[frame.cohort, "target"].sum()),
        "cohort_regions": int(frame.loc[frame.cohort, "region_id"].nunique()),
        "qualifying_clusters_all": len(barriers),
        "qualifying_clusters_cohort": len(eligible_barriers),
        "qualifying_clusters_unknown_street_id": sum(x["properties"]["street_edge_id"] not in raw_known for x in barriers),
        "status_counts": pd.Series([x["properties"]["status"] for x in streets]).value_counts().to_dict(),
        "place_categories": pd.Series([x["properties"]["category"] for x in places]).value_counts().to_dict(),
        "cohort_image_year_counts": image_years.to_dict(),
    }
    return frame, quality


def models():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("road_type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["way_type"]),
    ])
    return {
        "logistic": Pipeline([("features", prep), ("model", LogisticRegression(max_iter=2000, C=1.0))]),
        "forest": Pipeline([("features", clone(prep)), ("model", RandomForestClassifier(
            n_estimators=180, min_samples_leaf=8, max_features=0.8, random_state=SEED, n_jobs=2,
        ))]),
    }


def scores(y, p):
    return {
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def evaluate(cohort):
    X, y = cohort[FEATURES], cohort.target.to_numpy()
    coords = cohort[["x_m", "y_m"]].to_numpy()
    groups = KMeans(n_clusters=5, n_init=20, random_state=SEED).fit_predict(coords)
    splits = {
        "random": list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y)),
        "spatial": [],
    }
    checks = []
    for group in range(5):
        test = np.flatnonzero(groups == group)
        potential = np.flatnonzero(groups != group)
        distances = cKDTree(coords[test]).query(coords[potential])[0]
        train = potential[distances >= BUFFER_M]
        assert len(train) and len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
        assert len(set(train).intersection(test)) == 0
        checks.append({
            "fold": group, "train": len(train), "test": len(test),
            "buffer_excluded": len(potential) - len(train),
            "min_train_test_distance_m": float(cKDTree(coords[test]).query(coords[train])[0].min()),
            "train_prevalence": float(y[train].mean()), "test_prevalence": float(y[test].mean()),
        })
        splits["spatial"].append((train, test))
    oof = cohort[["street_edge_id", "region_name", "region_id", "target", "length_m", "nearest_health_m", "x_m", "y_m"]].copy()
    results, importances = [], []
    for split_name, pairs in splits.items():
        oof[f"{split_name}_fold"] = -1
        for model_name, prototype in models().items():
            pred = np.full(len(y), np.nan)
            for fold, (train, test) in enumerate(pairs):
                fitted = clone(prototype).fit(X.iloc[train], y[train])
                pred[test] = fitted.predict_proba(X.iloc[test])[:, 1]
                oof.loc[test, f"{split_name}_fold"] = fold
                results.append({"split": split_name, "model": model_name, "fold": fold,
                                "n": len(test), **scores(y[test], pred[test])})
                if split_name == "spatial" and model_name == "forest":
                    # Test-fold permutation only interprets a frozen model;
                    # it is not used for model selection or hyperparameter tuning.
                    base_ap = average_precision_score(y[test], pred[test])
                    rng = np.random.default_rng(SEED + fold)
                    for feature in FEATURES:
                        drops = []
                        for _ in range(3):
                            altered = X.iloc[test].copy()
                            altered[feature] = rng.permutation(altered[feature].to_numpy())
                            perm = fitted.predict_proba(altered)[:, 1]
                            drops.append(base_ap - average_precision_score(y[test], perm))
                        importances.append({"fold": fold, "feature": feature, "ap_drop": float(np.mean(drops))})
                print(f"{split_name:7} {model_name:8} fold {fold}: AP={results[-1]['ap']:.3f}", flush=True)
            assert np.isfinite(pred).all(), "Each row must receive exactly one held-out prediction."
            oof[f"{split_name}_{model_name}"] = pred
            results.append({"split": split_name, "model": model_name, "fold": "pooled",
                            "n": len(y), **scores(y, pred)})
    pd.DataFrame(results).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "split_checks.csv", index=False)
    pd.DataFrame(importances).to_csv(OUT / "permutation_importance.csv", index=False)
    oof.to_csv(OUT / "predictions.csv", index=False)
    return oof, pd.DataFrame(results)


def equal_region_selection(part, budget):
    """Round-robin one slot per neighborhood, ranked by score within each."""
    queues = {
        region: sub.sort_values(["spatial_forest", "street_edge_id"], ascending=[False, True]).index.tolist()
        for region, sub in part.groupby("region_id")
    }
    chosen = []
    while len(chosen) < budget:
        advanced = False
        for region in sorted(queues):
            if queues[region] and len(chosen) < budget:
                chosen.append(queues[region].pop(0))
                advanced = True
        if not advanced:
            break
    return chosen


def policies(oof):
    records, selected_rows = [], []
    for fold, part in oof.groupby("spatial_fold"):
        for fraction in (0.05, 0.10, 0.20):
            budget = max(1, math.ceil(len(part) * fraction))
            strategies = {
                "longest": part.sort_values(["length_m", "street_edge_id"], ascending=[False, True]).index[:budget],
                "near_health": part.sort_values(["nearest_health_m", "street_edge_id"]).index[:budget],
                "logistic": part.sort_values(["spatial_logistic", "street_edge_id"], ascending=[False, True]).index[:budget],
                "forest": part.sort_values(["spatial_forest", "street_edge_id"], ascending=[False, True]).index[:budget],
                "forest_region_quota": equal_region_selection(part, budget),
            }
            for name, indices in strategies.items():
                chosen = oof.loc[indices]
                records.append({
                    "fold": int(fold), "fraction": fraction, "strategy": name, "repeat": 0,
                    "budget": budget, "found": int(chosen.target.sum()),
                    "region_coverage": chosen.region_id.nunique() / part.region_id.nunique(),
                    "mean_length_m": float(chosen.length_m.mean()),
                    "total_length_m": float(chosen.length_m.sum()),
                })
                if fraction == 0.10:
                    selected_rows.extend({"strategy": name, "street_edge_id": int(sid), "fold": int(fold)}
                                         for sid in chosen.street_edge_id)
            rng = np.random.default_rng(SEED + int(fold) + int(fraction * 1000))
            for repeat in range(200):
                chosen = part.loc[rng.choice(part.index, budget, replace=False)]
                records.append({
                    "fold": int(fold), "fraction": fraction, "strategy": "random", "repeat": repeat,
                    "budget": budget, "found": int(chosen.target.sum()),
                    "region_coverage": chosen.region_id.nunique() / part.region_id.nunique(),
                    "mean_length_m": float(chosen.length_m.mean()),
                    "total_length_m": float(chosen.length_m.sum()),
                })
    detail = pd.DataFrame(records)
    detail.to_csv(OUT / "policy_folds.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_streets_10pct.csv", index=False)
    summary = []
    for (fraction, strategy), part in detail.groupby(["fraction", "strategy"]):
        repetitions = part.groupby("repeat").agg(found=("found", "sum"), budget=("budget", "sum"),
                                                region_coverage=("region_coverage", "mean"),
                                                total_length_m=("total_length_m", "sum"))
        precision = repetitions.found / repetitions.budget
        summary.append({
            "fraction": fraction, "strategy": strategy,
            "budget": int(repetitions.budget.iloc[0]),
            "mean_found": float(repetitions.found.mean()),
            "precision": float(precision.mean()),
            "randomization_p05": float(precision.quantile(.05)) if strategy == "random" else None,
            "randomization_p95": float(precision.quantile(.95)) if strategy == "random" else None,
            "macro_region_coverage": float(repetitions.region_coverage.mean()),
            "mean_selected_length_m": float((repetitions.total_length_m / repetitions.budget).mean()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    return summary


def figures(frame, oof, metrics, policy):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    colors = {"logistic": "#4c78a8", "forest": "#188977"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    pooled = metrics[metrics.fold == "pooled"]
    for j, model in enumerate(("logistic", "forest")):
        values = [pooled[(pooled.split == s) & (pooled.model == model)].ap.iloc[0] for s in ("random", "spatial")]
        positions = np.arange(2) + (j - .5) * .3
        axes[0].bar(positions, values, .3, label=model.title(), color=colors[model])
        for pos, val in zip(positions, values):
            axes[0].text(pos, val + .015, f"{val:.3f}", ha="center", fontsize=9)
    axes[0].axhline(oof.target.mean(), color="gray", linestyle="--", label="Prevalence reference")
    axes[0].set(xticks=[0, 1], xticklabels=["Random split", "Spatial + 500 m buffer"],
                ylim=(0, 1), ylabel="Average precision (higher is better)", title="A. Held-out prediction")
    axes[0].legend(fontsize=8, loc="upper right")
    subset = policy[np.isclose(policy.fraction, .10)].set_index("strategy")
    order = ["random", "longest", "near_health", "logistic", "forest", "forest_region_quota"]
    labels = ["Random", "Longest\nstreet", "Near\nhealth", "Logistic", "Forest", "Forest +\nregion quota"]
    values = [subset.loc[x, "precision"] for x in order]
    axes[1].bar(range(6), values, color=["#b9bec4", "#b9bec4", "#b9bec4", "#4c78a8", "#188977", "#7856a4"])
    for j, val in enumerate(values):
        axes[1].text(j, val + .015, f"{val:.1%}", ha="center", fontsize=9)
    r = subset.loc["random"]
    axes[1].errorbar(0, r.precision, yerr=[[r.precision-r.randomization_p05], [r.randomization_p95-r.precision]],
                     color="#333333", capsize=4, fmt="none")
    axes[1].set(xticks=range(6), xticklabels=labels, ylim=(0, 1),
                ylabel="Share with a qualifying public record",
                title=f"B. Same {int(r.budget)} street-inspection slots")
    fig.tight_layout()
    fig.savefig(OUT / "model_and_policy.png")
    fig.savefig(OUT / "model_and_policy.svg")
    plt.close(fig)

    current_ids = set(frame.street_edge_id)
    cohort_ids = set(oof.street_edge_id)
    selected = set(pd.read_csv(OUT / "selected_streets_10pct.csv").query("strategy == 'forest'").street_edge_id)
    lines, sets = [], []
    for feature in read_raw("streets"):
        sid = feature["properties"]["street_edge_id"]
        if sid in current_ids:
            lines.append(xy(feature["geometry"]["coordinates"]) / 1000)
            sets.append(sid)
    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    for ax in axes:
        ax.add_collection(LineCollection(lines, colors="#d9dde1", linewidths=.45))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set(xlabel="East-west distance (km)", ylabel="North-south distance (km)")
    palette = np.array(["#31688e", "#35b779", "#e69f00", "#8c6bb1", "#ca4b4b"])
    axes[0].scatter(oof.x_m/1000, oof.y_m/1000, c=palette[oof.spatial_fold.to_numpy()], s=2, rasterized=True)
    axes[0].set_title("Five geographic test areas\n500 m train/test exclusion buffer")
    selected_lines = [line for line, sid in zip(lines, sets) if sid in selected]
    axes[1].add_collection(LineCollection(selected_lines, colors="#188977", linewidths=1.25))
    axes[1].set_title("Forest-selected inspection streets\nRetrospective held-out evaluation")
    fig.tight_layout()
    fig.savefig(OUT / "study_map.png")
    fig.savefig(OUT / "study_map.svg")
    plt.close(fig)

    imp = pd.read_csv(OUT / "permutation_importance.csv").groupby("feature").ap_drop.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(imp.index, imp.values, color="#188977")
    ax.axvline(0, color="#777777", lw=.8)
    ax.set(xlabel="Mean test AP decrease after shuffling (five spatial folds)",
           title="Model dependence on recorded features; not causal effects")
    fig.tight_layout()
    fig.savefig(OUT / "feature_dependence.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    frame, quality = prepare()
    (OUT / "data_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return
    cohort = frame.loc[frame.cohort].reset_index(drop=True)
    oof, metrics = evaluate(cohort)
    policy = policies(oof)
    figures(frame, oof, metrics, policy)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    config = {
        "seed": SEED, "features": FEATURES, "spatial_buffer_m": BUFFER_M,
        "n_spatial_groups": 5, "inspection_fractions": [.05, .1, .2],
        "random_repeats": 200, "versions": versions,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "label_definition": "negative type; severity>=2 except NoSidewalk; agree>=2 and agree>disagree",
        "cohort": "status=open; outdated=false; length>=1m; audit_count>=2",
    }
    (OUT / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("\nPooled metrics:\n", metrics.loc[metrics.fold == "pooled"].to_string(index=False))
    print("\n10% inspection budget:\n", policy[np.isclose(policy.fraction, .1)].to_string(index=False))


if __name__ == "__main__":
    main()
