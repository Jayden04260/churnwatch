"""
retrain.py

Manually triggered (see .github/workflows/retrain.yml - workflow_dispatch
only, never on push) once a drift alert has fired. Retrains on the newly-
labeled batch, evaluates against the same fixed holdout train.py used, and
only promotes the new model if registry.is_better() says it actually wins.

Uses data/live_2011_labeled.parquet (see build_dataset.py) - the same
reference dates as the "live" scoring traffic, but with real churn labels.
In a real deployment those labels wouldn't exist yet at scoring time (they
require a 90-day forward window to resolve) - this file represents what
becomes available once that window has passed, matching the real-world
"churn labels arrive late" constraint.
"""

from __future__ import annotations

import pandas as pd

from registry import is_better
from train import compute_baseline_stats, next_version_number, save_version, train_model


def load_current_metrics() -> dict:
    import json
    from pathlib import Path

    current_version = json.loads(Path("models/current.json").read_text())["version"]
    metadata = json.loads(Path(f"models/v{current_version}/metadata.json").read_text())
    return metadata["metrics"]


def main() -> None:
    new_data = pd.read_parquet("data/live_2011_labeled.parquet")
    current_metrics = load_current_metrics()

    candidate_model, candidate_metrics, X_train, _ = train_model(new_data)

    print(f"Current model metrics:   {current_metrics}")
    print(f"Candidate model metrics: {candidate_metrics}")

    if is_better(candidate_metrics, current_metrics):
        baseline_stats = compute_baseline_stats(X_train)
        version = next_version_number()
        version_dir = save_version(candidate_model, candidate_metrics, baseline_stats, version)
        print(f"\nPromoted: v{version} beats the current model on roc_auc. Saved to {version_dir}.")
        print("Redeploy the API Lambda (update-function-code) to pick up the new model.")
    else:
        print("\nNot promoted: the candidate does not meaningfully beat the current model.")


if __name__ == "__main__":
    main()
