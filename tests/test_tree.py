"""
Unit tests for the batch decision tree (numcompute_stream/tree.py).
"""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, gini, entropy


# 1. gini / entropy

def test_gini_zero_when_pure():
    assert gini(np.array([0, 0, 0, 0])) == pytest.approx(0.0)


def test_gini_half_at_half_and_half():
    assert gini(np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_entropy_zero_when_pure():
    assert entropy(np.array([1, 1])) == pytest.approx(0.0)


def test_entropy_one_bit_at_half_and_half():
    assert entropy(np.array([0, 1])) == pytest.approx(1.0)


def test_gini_and_entropy_empty_group_is_zero():
    assert gini(np.array([], dtype=int)) == pytest.approx(0.0)
    assert entropy(np.array([], dtype=int)) == pytest.approx(0.0)


# 2. Matches classification-tree.html's 12-student dataset

def test_tree_matches_reference_root_split():
    # submitted, attended, study_group -> passed
    X = np.array([
        [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0], [1, 0, 0],
        [0, 1, 1], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0],
    ], dtype=float)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    assert tree.root.feature == 0  # "submitted" has the largest gain at the root


# 3. Numeric threshold splits

def test_tree_numeric_threshold_split():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    assert tree.root.threshold == pytest.approx(2.5)

    preds = tree.predict(np.array([[1.5], [3.5]]))
    assert np.array_equal(preds, [0, 1])


def test_duplicate_feature_values_are_valid_tie_candidates():
    X = np.array([[1.0], [1.0], [2.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    assert tree.root.threshold == pytest.approx(1.5)


def test_entropy_criterion_builds_equivalent_split():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=1, criterion="entropy")
    tree.fit(X, y)
    assert tree.root.threshold == pytest.approx(2.5)


# 4. Stop rules

def test_pure_labels_stop_immediately():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X, y)
    assert tree.root.label == 1
    assert tree.root.left is None


def test_min_samples_split_prevents_split():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=3)
    tree.fit(X, y)
    assert tree.root.label is not None  # too few samples to split


def test_constant_features_fall_back_to_leaf():
    X = np.full((4, 2), 1.0)
    y = np.array([0, 1, 0, 1])

    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X, y)
    assert tree.root.label is not None  # no split can improve on the impurity before splitting


# 5. max_features / random_state

def test_max_features_with_random_state_is_reproducible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 5))
    y = (X[:, 0] > 0).astype(int)

    tree_a = DecisionTreeClassifier(max_depth=3, max_features=2, random_state=42).fit(X, y)
    tree_b = DecisionTreeClassifier(max_depth=3, max_features=2, random_state=42).fit(X, y)

    assert np.array_equal(tree_a.predict(X), tree_b.predict(X))


# 6. Input validation and error messages

def test_fit_rejects_nan_values():
    X = np.array([[1.0], [np.nan]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="NaN or infinite"):
        DecisionTreeClassifier().fit(X, y)


def test_fit_rejects_non_binary_labels():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 2])

    with pytest.raises(ValueError, match="0 or 1"):
        DecisionTreeClassifier().fit(X, y)


def test_fit_row_count_mismatch_raises():
    with pytest.raises(ValueError, match="same number of rows"):
        DecisionTreeClassifier().fit(np.zeros((3, 1)), np.zeros(2))


def test_fit_empty_data_raises():
    with pytest.raises(ValueError, match="empty"):
        DecisionTreeClassifier().fit(np.zeros((0, 2)), np.zeros(0))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted yet"):
        DecisionTreeClassifier().predict(np.zeros((2, 1)))


def test_invalid_max_depth_raises():
    with pytest.raises(ValueError, match="max_depth must be positive"):
        DecisionTreeClassifier(max_depth=0)


def test_invalid_criterion_raises():
    with pytest.raises(ValueError, match="Unknown criterion"):
        DecisionTreeClassifier(criterion="bogus")


def test_invalid_min_samples_split_raises():
    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTreeClassifier(min_samples_split=1)


def test_invalid_max_features_raises():
    with pytest.raises(ValueError, match="max_features must be positive"):
        DecisionTreeClassifier(max_features=0)


def test_invalid_buffer_size_raises():
    with pytest.raises(ValueError, match="buffer_size must be positive"):
        DecisionTreeClassifier(buffer_size=0)


# 7. partial_fit (buffered retraining)

def test_partial_fit_does_not_retrain_below_buffer_size():
    tree = DecisionTreeClassifier(max_depth=2, buffer_size=10)
    tree.partial_fit(np.array([[1.0], [2.0]]), np.array([0, 1]))

    with pytest.raises(RuntimeError, match="not fitted yet"):
        tree.predict(np.array([[1.0]]))


def test_partial_fit_retrains_once_buffer_size_reached():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    tree.partial_fit(np.array([[1.0], [2.0]]), np.array([0, 0]))
    tree.partial_fit(np.array([[3.0], [4.0]]), np.array([1, 1]))

    assert tree.root is not None
    assert tree.root.threshold == pytest.approx(2.5)


def test_partial_fit_retrains_when_one_chunk_exceeds_buffer_size():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=2)
    tree.partial_fit(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))

    assert tree.root is not None
    assert tree.root.threshold == pytest.approx(2.5)


def test_predict_uses_last_trained_tree_between_retrains():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    tree.partial_fit(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    root_after_first_retrain = tree.root

    tree.partial_fit(np.array([[5.0]]), np.array([1]))  # below buffer_size, no retrain yet
    assert tree.root is root_after_first_retrain
    assert tree.predict(np.array([[1.5]]))[0] == 0


def test_partial_fit_matches_fit_on_the_same_data():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    streamed = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    streamed.partial_fit(X, y)

    batch = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert np.array_equal(streamed.predict(X), batch.predict(X))


def test_partial_fit_rejects_nan_chunk():
    tree = DecisionTreeClassifier(buffer_size=4)
    with pytest.raises(ValueError, match="NaN or infinite"):
        tree.partial_fit(np.array([[np.nan]]), np.array([0]))


def test_partial_fit_feature_dimension_mismatch_raises():
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(np.zeros((2, 2)), np.array([0, 1]))

    with pytest.raises(ValueError, match="dimension mismatch"):
        tree.partial_fit(np.zeros((2, 3)), np.array([0, 1]))
