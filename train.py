"""
train.py

Trains the churn classifier on data/train.parquet (see build_dataset.py),
evaluates it on the fixed out-of-time holdout (see HOLDOUT_START), and
writes a new model version to models/v{n}/:

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
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from drift import bucket_edges_to_proportions
from features import CHURN_WINDOW_DAYS, FEATURE_COLUMNS

MODELS_DIR = Path("models")
RANDOM_STATE = 42
N_QUANTILE_BUCKETS = 10  # for the PSI baseline - see drift.py in Phase C


# Every model version is scored on the same fixed, out-of-time holdout: the
# labelled snapshots from this date onward. Two reasons, both learned the
# hard way (see README "Retraining v2"):
#
# 1. A random row split leaks. The same customer appears in every monthly
#    snapshot, with near-identical features month to month, so a random
#    split puts "the same customer, one month later" in the test set. A
#    split by time can't do that.
# 2. Metrics are only comparable on the same test set. The original
#    retrain.py compared a candidate scored on a random 2011 split against
#    v1's stored score from a random 2010 split - two different test sets,
#    so the promotion decision wasn't measuring anything.
HOLDOUT_START = pd.Timestamp("2011-06-01")


def holdout_split(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits labelled snapshots into (trainable, holdout). Holdout is every
    snapshot on/after HOLDOUT_START. Trainable is every snapshot whose
    CHURN_WINDOW_DAYS label window has fully closed before HOLDOUT_START -
    so no training label depends on anything that happened during the
    holdout period. Snapshots in the gap between (label window still open
    when the holdout starts) are dropped from both.
    """
    reference = labeled["reference_date"]
    holdout = labeled[reference >= HOLDOUT_START]
    label_closes = reference + pd.Timedelta(days=CHURN_WINDOW_DAYS)
    trainable = labeled[label_closes < HOLDOUT_START]
    return trainable, holdout


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict]:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(train_df[FEATURE_COLUMNS], train_df["churned"])
    metrics = evaluate(model, test_df[FEATURE_COLUMNS], test_df["churned"])
    metrics["train_rows"] = len(train_df)
    return model, metrics


def evaluate(model: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        # Lower is better. The API returns a probability, not a label, so
        # whether "0.7" really means ~70% matters - Brier tracks that
        # (calibration + sharpness) where AUC only measures ranking.
        "brier": brier_score_loss(y_test, probabilities),
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
    for column in X_train.columns:
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


def load_holdout() -> pd.DataFrame:
    _, holdout = holdout_split(pd.read_parquet("data/live_2011_labeled.parquet"))
    return holdout


def main() -> None:
    # v1 is trained on Year 1 only - the "deployed before the drift" model
    # the rest of the project watches degrade. retrain.py builds later
    # versions on everything labelled since.
    train_df = pd.read_parquet("data/train.parquet")
    model, metrics = train_model(train_df, load_holdout())
    baseline_stats = compute_baseline_stats(train_df[FEATURE_COLUMNS])

    version = next_version_number()
    version_dir = save_version(model, metrics, baseline_stats, version)

    print(f"Trained model v{version} on {len(train_df)} rows.")
    print(f"Metrics: {metrics}")
    print(f"Saved to {version_dir}, set as current.")


if __name__ == "__main__":
    main()
