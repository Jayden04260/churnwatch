import pandas as pd

from train import N_QUANTILE_BUCKETS, compute_baseline_stats, next_version_number


def test_compute_baseline_stats_has_an_entry_per_feature_with_correct_bucket_count():
    X_train = pd.DataFrame(
        {
            "recency_days": range(100),
            "frequency": range(100),
            "monetary": range(100),
        }
    )
    stats = compute_baseline_stats(X_train)

    assert set(stats.keys()) == {"recency_days", "frequency", "monetary"}
    for column_stats in stats.values():
        # N buckets means N+1 edges (including the -inf/+inf caps).
        assert len(column_stats["bucket_edges"]) == N_QUANTILE_BUCKETS + 1


def test_compute_baseline_stats_edges_cover_the_full_range():
    X_train = pd.DataFrame(
        {
            "recency_days": [10, 20, 30, 40, 50],
            "frequency": [1, 2, 3, 4, 5],
            "monetary": [100, 200, 300, 400, 500],
        }
    )
    stats = compute_baseline_stats(X_train)
    edges = stats["recency_days"]["bucket_edges"]

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    # interior edges should be non-decreasing (valid bucket boundaries)
    interior = edges[1:-1]
    assert interior == sorted(interior)


def test_compute_baseline_stats_deduplicates_edges_for_low_cardinality_features():
    # Almost all values are 1 - only a couple of quantiles can be
    # meaningfully distinct, so the edges must collapse rather than
    # produce duplicate bin boundaries (which pd.cut rejects outright).
    X_train = pd.DataFrame(
        {
            "recency_days": range(100),
            "frequency": [1] * 95 + [2, 2, 3, 4, 5],
            "monetary": range(100),
        }
    )
    stats = compute_baseline_stats(X_train)
    edges = stats["frequency"]["bucket_edges"]

    assert len(edges) == len(set(edges))  # no duplicates
    assert len(edges) <= N_QUANTILE_BUCKETS + 1


def test_next_version_number_starts_at_one_and_increments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import train

    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")

    assert next_version_number() == 1

    (tmp_path / "models" / "v1").mkdir(parents=True)
    assert next_version_number() == 2

    (tmp_path / "models" / "v2").mkdir(parents=True)
    (tmp_path / "models" / "not_a_version").mkdir(parents=True)  # should be ignored, not crash
    assert next_version_number() == 3
