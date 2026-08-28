"""
drift.py

Population Stability Index (PSI) - a standard, widely-used metric for
detecting distribution drift between a reference (training) population and
a current (live) one. Pure functions, no I/O - drift_check.py (the Lambda
handler) is responsible for pulling the actual reference stats and live
prediction logs and handing them to these.

PSI interpretation (industry rule of thumb, cited in most credit-risk and
MLOps literature):
    < 0.1  - no significant shift
    0.1-0.2 - moderate shift, worth watching
    > 0.2  - significant shift, action recommended

DRIFT_THRESHOLD below is set at that 0.2 boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRIFT_THRESHOLD = 0.2
# PSI is undefined (division by zero) for an empty bucket - this floor
# keeps a single empty bucket from producing NaN/inf and derailing the
# whole feature's score, without meaningfully distorting well-populated
# buckets.
EPSILON = 1e-4


def bucket_edges_to_proportions(values: pd.Series, bucket_edges: list[float]) -> np.ndarray:
    """
    Assigns each value to a bucket defined by bucket_edges (N+1 edges for
    N buckets, as produced by train.py's compute_baseline_stats), and
    returns the proportion of values falling in each bucket.
    """
    bucket_indices = pd.cut(values, bins=bucket_edges, labels=False, include_lowest=True)
    counts = bucket_indices.value_counts(sort=False).reindex(range(len(bucket_edges) - 1), fill_value=0)
    return (counts / max(len(values), 1)).to_numpy()


def psi(reference_proportions: np.ndarray, current_proportions: np.ndarray) -> float:
    """
    Population Stability Index between two bucket-proportion arrays (same
    bucket definition, i.e. same length and same edges used to produce
    both). Larger = more drift.
    """
    ref = np.clip(reference_proportions, EPSILON, None)
    cur = np.clip(current_proportions, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compute_feature_drift(live_values: pd.Series, bucket_edges: list[float], reference_proportions: list[float]) -> dict:
    """
    Convenience wrapper: given a feature's live values and its baseline
    bucket_edges + reference_proportions (both from
    models/v*/baseline_stats.json - see train.py's compute_baseline_stats),
    returns the PSI score plus whether it crosses DRIFT_THRESHOLD.

    reference_proportions must be the *actual* training-data proportion per
    bucket, not assumed uniform: train.py deduplicates quantile edges for
    low-cardinality features (e.g. "frequency"), and once edges are
    deduplicated the remaining buckets are no longer equal-sized slices of
    the training population.
    """
    current_proportions = bucket_edges_to_proportions(live_values, bucket_edges)
    score = psi(np.array(reference_proportions), current_proportions)
    return {
        "psi": score,
        "drifted": score > DRIFT_THRESHOLD,
    }
