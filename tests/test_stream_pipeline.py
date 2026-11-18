"""
Unit tests for the streaming pipeline (numcompute_stream/pipeline.py).
"""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StreamingStandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


# 1. partial_fit chaining a transformer with a final model

def test_partial_fit_matches_manual_composition():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scale", StreamingStandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=1, buffer_size=4)),
    ])
    pipe.partial_fit(X, y)

    scaler = StreamingStandardScaler()
    X_scaled = scaler.partial_fit(X).transform(X)
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    tree.partial_fit(X_scaled, y)

    assert np.array_equal(pipe.predict(X), tree.predict(X_scaled))


def test_predict_transforms_new_data_through_the_scaler():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])

    pipe = Pipeline([
        ("scale", StreamingStandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=1, buffer_size=4)),
    ])
    pipe.partial_fit(X, y)

    preds = pipe.predict(np.array([[1.0], [4.0]]))
    assert np.array_equal(preds, [0, 1])


def test_predict_before_model_retrains_raises():
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("scale", StreamingStandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=1, buffer_size=10)),
    ])
    pipe.partial_fit(X, y)  # below buffer_size, model not trained yet

    with pytest.raises(RuntimeError, match="not fitted yet"):
        pipe.predict(X)


def test_model_retrains_once_buffer_size_reached_across_chunks():
    pipe = Pipeline([
        ("scale", StreamingStandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=1, buffer_size=4)),
    ])
    pipe.partial_fit(np.array([[1.0], [2.0]]), np.array([0, 0]))
    pipe.partial_fit(np.array([[3.0], [4.0]]), np.array([1, 1]))

    preds = pipe.predict(np.array([[1.0], [4.0]]))
    assert np.array_equal(preds, [0, 1])


# 2. transform() only

def test_transform_only_pipeline_chains_in_order():
    X = np.array([[2.0], [4.0]])

    pipe = Pipeline([
        ("scale1", StreamingStandardScaler(with_std=False)),
        ("scale2", StreamingStandardScaler(with_std=False)),
    ])
    pipe.partial_fit(X)  # no y_chunk: both steps are transformers

    out = pipe.transform(X)
    expected = X - np.mean(X) - np.mean(X - np.mean(X))
    assert np.allclose(out, expected)


def test_predict_raises_when_final_step_has_no_predict():
    pipe = Pipeline([
        ("scale1", StreamingStandardScaler()),
        ("scale2", StreamingStandardScaler()),
    ])
    pipe.partial_fit(np.array([[1.0], [2.0]]))

    with pytest.raises(AttributeError, match="does not implement 'predict'"):
        pipe.predict(np.array([[1.0]]))


def test_transform_raises_when_a_step_has_no_transform():
    pipe = Pipeline([
        ("scale", StreamingStandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=2)),
    ])
    pipe.partial_fit(np.array([[1.0], [2.0]]), np.array([0, 1]))

    with pytest.raises(AttributeError, match="does not implement 'transform'"):
        pipe.transform(np.array([[1.0]]))
