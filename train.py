"""
train.py

Trains the churn classifier on data/train.parquet (see build_dataset.py),
evaluates it on a held-out split, and writes a new model version to
models/v{n}/:

  model.pkl           - the fitted scikit-learn pipeline
  metadata.json        - metrics, timestamp, feature list, training row count
  baseline_stats.json  - per-feature quantile buckets from the training set,
                          the reference distribution drift_check.py (Phase C)
                          compares live traffic against via PSI

Also updates models/current.json to point at the new version. Run directly
for the initial v1 model; retrain.py (Phase D) reuses train_model() and
evaluate() but only promotes a new version if it actually beats the
current one.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from drift import bucket_edges_to_proportions
from features import FEATURE_COLUMNS

MODELS_DIR = Path("models")
RANDOM_STATE = 42
N_QUANTILE_BUCKETS = 10  # for the PSI baseline - see drift.py in Phase C


def train_model(train_df: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict, pd.DataFrame, pd.Series]:
    X = train_df[FEATURE_COLUMNS]
    y = train_df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    return model, metrics, X_train, y_train


def evaluate(model: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "test_rows": len(y_test),
    }


def compute_baseline_stats(X_train: pd.DataFrame, buckets: int = N_QUANTILE_BUCKETS) -> dict:
    """
    Quantile bucket edges per feature, computed from the training data.
    drift_check.py buckets live traffic the same way and compares the
    resulting bucket proportions against these via PSI - this is the
    "reference distribution" a churn model's incoming traffic gets judged
    against.
    """
    stats = {}
    for column in FEATURE_COLUMNS:
        quantiles = np.linspace(0, 1, buckets + 1)
        edges = X_train[column].quantile(quantiles).tolist()
        edges[0], edges[-1] = -np.inf, np.inf  # so any live value always falls in a bucket
        # Low-cardinality features (e.g. "frequency", where many customers
        # share the same small integer value) can produce duplicate
        # quantile edges - pd.cut rejects those as invalid bin boundaries.
        # Deduplicating is the correct fix, not a workaround: a feature
        # that only takes a handful of distinct values genuinely can't
        # support N meaningfully distinct buckets, so it should end up
        # with fewer, wider ones instead of raising.
        edges = sorted(set(edges))

        # Deduped edges are no longer guaranteed to each hold ~1/buckets of
        # the training data (some quantile points collapsed into others),
        # so PSI needs the *actual* per-bucket training proportions here -
        # assuming uniformity at drift-check time would silently produce
        # wrong PSI scores for exactly the features that needed deduping.
        reference_proportions = bucket_edges_to_proportions(X_train[column], edges).tolist()

        stats[column] = {"bucket_edges": edges, "reference_proportions": reference_proportions}
    return stats


def next_version_number() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    existing = [int(p.name[1:]) for p in MODELS_DIR.glob("v*") if p.name[1:].isdigit()]
    return max(existing, default=0) + 1


def save_version(
    model: GradientBoostingClassifier, metrics: dict, baseline_stats: dict, version: int
) -> Path:
    version_dir = MODELS_DIR / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, version_dir / "model.pkl")

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (version_dir / "baseline_stats.json").write_text(json.dumps(baseline_stats, indent=2))

    (MODELS_DIR / "current.json").write_text(json.dumps({"version": version}, indent=2))
    return version_dir


def main() -> None:
    train_df = pd.read_parquet("data/train.parquet")
    model, metrics, X_train, _ = train_model(train_df)
    baseline_stats = compute_baseline_stats(X_train)

    version = next_version_number()
    version_dir = save_version(model, metrics, baseline_stats, version)

    print(f"Trained model v{version} on {len(train_df)} rows.")
    print(f"Metrics: {metrics}")
    print(f"Saved to {version_dir}, set as current.")


if __name__ == "__main__":
    main()
