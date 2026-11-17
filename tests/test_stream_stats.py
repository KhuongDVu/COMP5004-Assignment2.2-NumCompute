"""
Unit tests for streaming statistics (numcompute_stream/stats.py).
"""

import numpy as np
import pytest

from numcompute_stream.stats import StreamStats


# 1. Running mean / variance

def test_running_mean_var_matches_batch():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(20, 3)) for _ in range(3)]
    stats = StreamStats()
    for chunk in chunks:
        stats.update_stats(chunk)

    full = np.vstack(chunks)
    assert np.allclose(stats.mean, np.mean(full, axis=0))
    assert np.allclose(stats.var, np.var(full, axis=0))
    assert stats.count == full.shape[0]


def test_single_feature_1d_input():
    stats = StreamStats()
    stats.update_stats(np.array([1.0, 2.0, 3.0]))
    stats.update_stats(np.array([4.0, 5.0]))

    assert stats.mean.shape == (1,)
    assert np.allclose(stats.mean, [3.0])
    assert stats.count == 5


def test_zero_variance_chunk_no_nan_or_inf():
    stats = StreamStats()
    stats.update_stats(np.full((10, 2), 5.0))
    stats.update_stats(np.array([[5.0, 1.0], [5.0, 3.0]]))

    assert not np.isnan(stats.var).any()
    assert not np.isinf(stats.var).any()
    assert stats.var[0] == pytest.approx(0.0)


# Rows containing NaN are excluded from mean/variance
def test_nan_rows_dropped_from_mean_var():
    stats = StreamStats()
    stats.update_stats(np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]]))

    assert stats.count == 2
    assert np.allclose(stats.mean, [2.0, 3.0])

# mismatch dimension
def test_feature_dimension_mismatch_raises():
    stats = StreamStats()
    stats.update_stats(np.zeros((3, 2)))

    with pytest.raises(ValueError, match="dimension mismatch"):
        stats.update_stats(np.zeros((3, 3)))


# 2. Histogram / quantiles

# Histogram counts (no window) accumulate across every chunk
def test_histogram_counts_sum_to_valid_rows():
    stats = StreamStats(bins=5, bin_range=(0.0, 10.0))
    stats.update_stats(np.array([1.0, 2.0, 9.0]).reshape(-1, 1))
    stats.update_stats(np.array([3.0, np.nan]).reshape(-1, 1))

    counts, edges = stats.histogram()
    assert counts.sum() == 4  # drop NaN
    assert edges.shape == (6,)


def test_sliding_window_histogram_drops_old_chunks():
    stats = StreamStats(bins=4, bin_range=(0.0, 4.0), window=1)
    stats.update_stats(np.array([0.5, 0.5, 0.5]).reshape(-1, 1))
    counts, _ = stats.histogram()
    assert counts.sum() == 3

    stats.update_stats(np.array([3.5]).reshape(-1, 1))
    counts, _ = stats.histogram()
    assert counts.sum() == 1


def test_quantiles_scalar_input_returns_float():
    stats = StreamStats(bins=10, bin_range=(0.0, 10.0))
    stats.update_stats(np.arange(10.0).reshape(-1, 1))

    result = stats.quantiles(0.5)
    assert isinstance(result, float)

# 3. Constructor validation

def test_invalid_bins_or_window_raises():
    with pytest.raises(ValueError, match="bins must be positive"):
        StreamStats(bins=0)
    with pytest.raises(ValueError, match="window must be positive"):
        StreamStats(window=0)


def test_hist_feature_out_of_bounds_raises():
    stats = StreamStats(hist_feature=5)
    with pytest.raises(ValueError, match="out of bounds"):
        stats.update_stats(np.zeros((3, 2)))
