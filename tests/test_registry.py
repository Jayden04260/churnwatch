from registry import MIN_IMPROVEMENT, is_better


def test_is_better_true_for_a_clear_win():
    candidate = {"roc_auc": 0.80}
    current = {"roc_auc": 0.75}
    assert is_better(candidate, current) is True


def test_is_better_false_for_a_clear_loss():
    candidate = {"roc_auc": 0.70}
    current = {"roc_auc": 0.75}
    assert is_better(candidate, current) is False


def test_is_better_false_for_an_exact_tie():
    candidate = {"roc_auc": 0.75}
    current = {"roc_auc": 0.75}
    assert is_better(candidate, current) is False


def test_is_better_false_for_an_improvement_too_small_to_be_meaningful():
    candidate = {"roc_auc": 0.75 + MIN_IMPROVEMENT / 2}
    current = {"roc_auc": 0.75}
    assert is_better(candidate, current) is False


def test_is_better_true_right_above_the_improvement_threshold():
    candidate = {"roc_auc": 0.75 + MIN_IMPROVEMENT + 0.001}
    current = {"roc_auc": 0.75}
    assert is_better(candidate, current) is True
