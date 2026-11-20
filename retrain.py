"""
retrain.py

Manually triggered (see .github/workflows/retrain.yml - workflow_dispatch
only, never on push) once a drift alert has fired. Retrains on every
labelled snapshot available, scores the candidate AND the current model on
the same fixed out-of-time holdout (train.HOLDOUT_START), and only promotes
the candidate if registry.is_better() says it actually wins.

Training data is Year 1 (data/train.parquet) plus the Year 2 snapshots in
data/live_2011_labeled.parquet whose 90-day label window closed before the
holdout starts (see train.holdout_split). In a real deployment those labels
wouldn't exist at scoring time - they need the forward window to resolve -
so this represents what's available once that window has passed, matching
the real-world "churn labels arrive late" constraint.

The current model is re-scored here rather than trusting the metrics stored
in its metadata.json: those were computed whenever it was trained, possibly
on a different test set, and comparing numbers from two different test sets
is how the original version of this script made meaningless promotion
decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from registry import is_better
from train import (
    MODELS_DIR,
    compute_baseline_stats,
    evaluate,
    holdout_split,
    next_version_number,
    save_version,
    train_model,
)


def load_current_model():
    version = json.loads((MODELS_DIR / "current.json").read_text())["version"]
    return version, joblib.load(MODELS_DIR / f"v{version}" / "model.pkl")


def build_training_and_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = pd.concat(
        [pd.read_parquet("data/train.parquet"), pd.read_parquet("data/live_2011_labeled.parquet")],
        ignore_index=True,
    )
    return holdout_split(labeled)


def main() -> None:
    train_df, holdout = build_training_and_holdout()
    current_version, current_model = load_current_model()

    current_metrics = evaluate(current_model, holdout[FEATURE_COLUMNS], holdout["churned"])
    candidate_model, candidate_metrics = train_model(train_df, holdout)

    dates = train_df["reference_date"]
    print(f"Training on {len(train_df)} rows, reference dates {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
    print(f"Holdout: {len(holdout)} rows from {holdout['reference_date'].min():%Y-%m-%d}")
    print(f"Current model (v{current_version}) on holdout: {current_metrics}")
    print(f"Candidate model on holdout:           {candidate_metrics}")

    if is_better(candidate_metrics, current_metrics):
        baseline_stats = compute_baseline_stats(train_df[FEATURE_COLUMNS])
        version = next_version_number()
        version_dir = save_version(candidate_model, candidate_metrics, baseline_stats, version)
        print(f"\nPromoted: v{version} beats v{current_version} on roc_auc. Saved to {version_dir}.")
        print("Redeploy the API Lambda (update-function-code) to pick up the new model.")
        _write_github_output(promoted=True, version=version)
    else:
        print(f"\nNot promoted: the candidate does not meaningfully beat v{current_version}.")
        _write_github_output(promoted=False, version=current_version)


def _write_github_output(promoted: bool, version: int) -> None:
    """Lets retrain.yml's later steps branch on the result when run in Actions."""
    import os

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a") as fh:
            fh.write(f"promoted={'true' if promoted else 'false'}\nversion={version}\n")


if __name__ == "__main__":
    main()
