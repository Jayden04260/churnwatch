import numpy as np
import pandas as pd

from drift import DRIFT_THRESHOLD, bucket_edges_to_proportions, compute_feature_drift, psi


def test_psi_is_zero_for_identical_distributions():
    proportions = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(proportions, proportions) == 0.0


def test_psi_increases_with_the_size_of_the_shift():
    reference = np.array([0.25, 0.25, 0.25, 0.25])
    small_shift = np.array([0.30, 0.25, 0.25, 0.20])
    large_shift = np.array([0.70, 0.10, 0.10, 0.10])

    small_psi = psi(reference, small_shift)
    large_psi = psi(reference, large_shift)

    assert 0 < small_psi < large_psi


def test_psi_matches_the_standard_below_and_above_threshold_cases():
    # A textbook "no significant shift" case (small perturbation) should
    # fall under the industry-standard 0.2 threshold, and a textbook
    # "significant shift" case (population concentrated into one bucket
    # that used to be a quarter of it) should clear it.
    reference = np.array([0.25, 0.25, 0.25, 0.25])
    minor = np.array([0.27, 0.24, 0.26, 0.23])
    major = np.array([0.85, 0.05, 0.05, 0.05])

    assert psi(reference, minor) < DRIFT_THRESHOLD
    assert psi(reference, major) > DRIFT_THRESHOLD


def test_bucket_edges_to_proportions_splits_values_correctly():
    values = pd.Series([1, 1, 5, 5, 5, 9])
    edges = [float("-inf"), 3, 7, float("inf")]  # 3 buckets: (<=3), (3,7], (>7)

    proportions = bucket_edges_to_proportions(values, edges)

    assert proportions.tolist() == [2 / 6, 3 / 6, 1 / 6]


def test_compute_feature_drift_flags_a_real_shift_against_a_uniform_baseline():
    # Baseline: 4 quantile buckets, each holding 25% of training data.
    bucket_edges = [float("-inf"), 25, 50, 75, float("inf")]
    reference_proportions = [0.25, 0.25, 0.25, 0.25]

    # Live traffic entirely in the lowest bucket - a real, large shift.
    live_values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = compute_feature_drift(live_values, bucket_edges, reference_proportions)

    assert result["drifted"] is True
    assert result["psi"] > DRIFT_THRESHOLD


def test_compute_feature_drift_does_not_flag_traffic_matching_the_baseline():
    bucket_edges = [float("-inf"), 25, 50, 75, float("inf")]
    reference_proportions = [0.25, 0.25, 0.25, 0.25]
    # Roughly a quarter of values in each bucket, matching the baseline.
    live_values = pd.Series([10, 15, 35, 40, 60, 65, 85, 90])
    result = compute_feature_drift(live_values, bucket_edges, reference_proportions)

    assert result["drifted"] is False
    assert result["psi"] < DRIFT_THRESHOLD
