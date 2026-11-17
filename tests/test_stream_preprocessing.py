"""
Unit tests for streaming preprocessing (numcompute_stream/preprocessing.py).
"""

import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.preprocessing import (
    StreamingStandardScaler,
    StreamingImputer,
    StreamingOneHotEncoder,
)

# 1. StreamingStandardScaler

def test_streaming_scaler_matches_batch():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(20, 3)) for _ in range(3)]
    scaler = StreamingStandardScaler()
    for chunk in chunks:
        scaler.partial_fit(chunk)

    full = np.vstack(chunks)
    batch_scaler = StandardScaler().fit(full)
    assert np.allclose(scaler.mean_, batch_scaler.mean_)
    assert np.allclose(scaler.scale_, batch_scaler.scale_)


def test_streaming_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted yet"):
        StreamingStandardScaler().transform(np.zeros((2, 2)))


def test_streaming_scaler_zero_variance_chunk_scales_by_one():
    scaler = StreamingStandardScaler()
    scaler.partial_fit(np.full((5, 2), 3.0))

    assert scaler.scale_[0] == pytest.approx(1.0)
    out = scaler.transform(np.full((1, 2), 3.0))
    assert np.allclose(out, 0.0)


def test_streaming_scaler_with_mean_false_skips_centering():
    scaler = StreamingStandardScaler(with_mean=False)
    scaler.partial_fit(np.array([[1.0], [3.0]]))

    assert scaler.mean_ is None
    out = scaler.transform(np.array([[3.0]]))
    assert out[0, 0] != 0.0


def test_streaming_scaler_dimension_mismatch_raises():
    scaler = StreamingStandardScaler()
    scaler.partial_fit(np.zeros((3, 2)))

    with pytest.raises(ValueError, match="dimension mismatch"):
        scaler.transform(np.zeros((3, 3)))


# 2. StreamingImputer

def test_streaming_imputer_mean_matches_batch():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 6.0]])
    imputer = StreamingImputer()
    imputer.partial_fit(X[:2])
    imputer.partial_fit(X[2:])

    batch_imputer = SimpleImputer(strategy="mean").fit(X)
    assert np.allclose(imputer.statistics_, batch_imputer.statistics_)


def test_streaming_imputer_fills_nan_with_running_mean():
    imputer = StreamingImputer()
    imputer.partial_fit(np.array([[1.0], [3.0]]))

    out = imputer.transform(np.array([[np.nan]]))
    assert out[0, 0] == pytest.approx(2.0)


def test_streaming_imputer_all_nan_column_fills_zero():
    imputer = StreamingImputer()
    imputer.partial_fit(np.array([[np.nan, 1.0], [np.nan, 2.0]]))

    assert imputer.statistics_[0] == pytest.approx(0.0)


def test_streaming_imputer_constant_strategy():
    imputer = StreamingImputer(strategy="constant", fill_value=-1.0)
    imputer.partial_fit(np.array([[np.nan, 2.0]]))

    out = imputer.transform(np.array([[np.nan, np.nan]]))
    assert np.allclose(out, [-1.0, -1.0])


def test_streaming_imputer_constant_without_fill_value_raises():
    with pytest.raises(ValueError, match="fill_value must be provided"):
        StreamingImputer(strategy="constant")


def test_streaming_imputer_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        StreamingImputer(strategy="median")


def test_streaming_imputer_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted yet"):
        StreamingImputer().transform(np.zeros((2, 2)))


# 3. StreamingOneHotEncoder

def test_streaming_encoder_grows_categories_midstream():
    encoder = StreamingOneHotEncoder()
    encoder.partial_fit(np.array([["a"], ["b"]]))
    first_out = encoder.transform(np.array([["a"]]))
    assert first_out.shape == (1, 2)

    encoder.partial_fit(np.array([["c"]]))  # new category shows up mid-stream
    second_out = encoder.transform(np.array([["c"]]))
    assert second_out.shape == (1, 3)
    assert second_out[0].sum() == 1


def test_streaming_encoder_matches_expected_columns():
    encoder = StreamingOneHotEncoder()
    encoder.partial_fit(np.array([[0], [1], [0]]))

    out = encoder.transform(np.array([[0], [1]]))
    assert np.array_equal(out, [[1.0, 0.0], [0.0, 1.0]])


def test_streaming_encoder_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted yet"):
        StreamingOneHotEncoder().transform(np.zeros((2, 1)))


def test_streaming_encoder_dimension_mismatch_raises():
    encoder = StreamingOneHotEncoder()
    encoder.partial_fit(np.array([["a", "x"]]))

    with pytest.raises(ValueError, match="dimension mismatch"):
        encoder.transform(np.array([["a"]]))
