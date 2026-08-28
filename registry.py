"""
registry.py

The promotion gate for retrain.py (Phase D): given the newly-retrained
model's metrics and the currently-deployed model's metrics, decides
whether the new one actually wins. Deliberately a single simple, legible
comparison rather than a black-box scoring function - the whole point of
a promotion gate is that a human can look at it and know exactly what
"better" means here.
"""

from __future__ import annotations

# ROC-AUC is the deciding metric: unlike accuracy, it isn't sensitive to
# the exact classification threshold or class balance, which matters here
# since the true churn rate can plausibly drift over time too, not just
# the RFM feature distributions.
DECIDING_METRIC = "roc_auc"

# A candidate must beat the current model by more than this to be promoted -
# not just any improvement, however small. Guards against promoting a
# "better" model that's really just noise from a different train/test
# split, which would otherwise flip-flop the production model on
# effectively no real signal.
MIN_IMPROVEMENT = 0.01


def is_better(candidate_metrics: dict, current_metrics: dict) -> bool:
    """
    True if candidate_metrics[DECIDING_METRIC] beats current_metrics'
    by more than MIN_IMPROVEMENT. A tie, or an improvement too small to
    be meaningful, keeps the current model in place.
    """
    return candidate_metrics[DECIDING_METRIC] - current_metrics[DECIDING_METRIC] > MIN_IMPROVEMENT
