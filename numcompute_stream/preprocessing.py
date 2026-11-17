"""
Streaming (chunk-based) preprocessing transformers: a running-stats scaler,
an online imputer, and a one-hot encoder with incrementally growing categories.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

from numcompute_stream.stats import StreamStats


class StreamingStandardScaler:
    """
    Standardize features online using a running mean/variance.

    Reuse StreamStats' vectorised parallel-Welford merge to fold
    each chunk's mean/variance into the running totals. Rows containing any
    NaN are excluded from the running statistics

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.

    Attributes
    ----------
    mean_ : Optional[np.ndarray]
        Running per-feature mean. Shape (D,). None if with_mean is False.
    scale_ : Optional[np.ndarray]
        Running per-feature standard deviation. Shape (D,). Zero-variance
        features are scaled by 1.0 instead of 0.0. None if with_std is False.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self._stats = StreamStats()
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def partial_fit(self, X_chunk: np.ndarray) -> "StreamingStandardScaler":
        """
        Fold a new chunk of rows into the running mean/variance.

        Parameters
        ----------
        X_chunk : np.ndarray
            New rows, shape (n, D) or (n,) for a single feature.

        Returns
        -------
        StreamingStandardScaler
            self, to allow chaining.

        Raises
        ------
        ValueError
            If X_chunk is not 1D/2D, or its feature count does not match a
            previous chunk's.

        Complexity
        ----------
        Time: O(n * D). Space: O(D) extra for the running totals.
        """
        self._stats.update_stats(X_chunk)

        if self.with_mean:
            self.mean_ = self._stats.mean
        else:
            self.mean_ = None

        if self.with_std:
            scale = np.sqrt(self._stats.var)
            self.scale_ = np.where(scale == 0.0, 1.0, scale)
        else:
            self.scale_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Center and/or scale X using the running mean/variance.

        Parameters
        ----------
        X : np.ndarray
            Data to transform, shape (n, D).

        Returns
        -------
        np.ndarray
            Transformed array, shape (n, D).

        Raises
        ------
        RuntimeError
            If called before any partial_fit.
        ValueError
            If X is not a 2D array or its feature count does not match.

        Complexity
        ----------
        Time: O(n * D). Space: O(n * D) for the returned array.
        """
        if self._stats.count == 0:
            raise RuntimeError("StreamingStandardScaler is not fitted yet; call partial_fit first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expect a 2D array")
        if X.shape[1] != self._stats.mean.shape[0]:
            raise ValueError("Feature dimension mismatch")

        X_out = X.copy()
        if self.with_mean:
            X_out -= self.mean_
        if self.with_std:
            X_out /= self.scale_

        return X_out

    def partial_fit_transform(self, X_chunk: np.ndarray) -> np.ndarray:
        # fold a chunk into the running stats then transform that same chunk
        return self.partial_fit(X_chunk).transform(X_chunk)


class StreamingImputer:
    """
    Fill missing values online using a running per-column mean (or a fixed constant).

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. Supported: 'mean', 'constant'.
    fill_value : Optional[float], default=None
        Value used when strategy='constant'.

    Attributes
    ----------
    statistics_ : Optional[np.ndarray]
        Running per-feature fill values. Shape (D,). None until the first chunk.
    """

    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None) -> None:
        if strategy not in ("mean", "constant"):
            raise ValueError("Unknown strategy. Choose 'mean' or 'constant'.")
        if strategy == "constant" and fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None
        self._sum: Optional[np.ndarray] = None
        self._count: Optional[np.ndarray] = None

    def partial_fit(self, X_chunk: np.ndarray) -> "StreamingImputer":
        """
        Fold a new chunk of rows into the running fill statistics.

        Parameters
        ----------
        X_chunk : np.ndarray
            New rows, possibly containing NaN, shape (n, D).

        Returns
        -------
        StreamingImputer
            self, to allow chaining.

        Raises
        ------
        ValueError
            If X_chunk is not a 2D array, or its feature count does not
            match a previous chunk's.

        Complexity
        ----------
        Time: O(n * D). Space: O(D) extra for the running totals.
        """
        X_chunk = np.asarray(X_chunk, dtype=np.float64)
        if X_chunk.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X_chunk.ndim}D array instead.")
        if self.statistics_ is not None and X_chunk.shape[1] != self.statistics_.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.statistics_.shape[0]}, got {X_chunk.shape[1]}."
            )

        if self.strategy == "constant":
            self.statistics_ = np.full(X_chunk.shape[1], self.fill_value, dtype=np.float64)
            return self

        if self._sum is None:
            self._sum = np.zeros(X_chunk.shape[1], dtype=np.float64)
            self._count = np.zeros(X_chunk.shape[1], dtype=np.int64)

        # only allow valid entries in each column's running sum/count
        valid = np.logical_not(np.isnan(X_chunk))
        self._sum += np.nansum(X_chunk, axis=0)
        self._count += valid.sum(axis=0)

        with np.errstate(invalid="ignore"):
            mean = self._sum / self._count
        mean[self._count == 0] = 0.0
        self.statistics_ = mean

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Replace NaNs in X with the running fill statistics.

        Parameters
        ----------
        X : np.ndarray
            Data to impute, shape (n, D).

        Returns
        -------
        np.ndarray
            Imputed array, shape (n, D).

        Raises
        ------
        RuntimeError
            If called before any partial_fit.
        ValueError
            If X is not a 2D array or its feature count does not match.

        Complexity
        ----------
        Time: O(n * D). Space: O(n * D) for the returned array.
        """
        if self.statistics_ is None:
            raise RuntimeError("StreamingImputer is not fitted yet; call partial_fit first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expect a 2D array")
        if X.shape[1] != self.statistics_.shape[0]:
            raise ValueError("Feature dimension mismatch.")

        X_out = X.copy()
        nan_mask = np.isnan(X_out)
        X_out = np.where(nan_mask, self.statistics_, X_out)

        return X_out

    def partial_fit_transform(self, X_chunk: np.ndarray) -> np.ndarray:
        return self.partial_fit(X_chunk).transform(X_chunk)


class StreamingOneHotEncoder:
    """
    One-hot encode categorical features, discovering new categories as chunks arrive.

    Attributes
    ----------
    categories_ : Optional[List[np.ndarray]]
        Per-feature sorted unique categories discovered so far. None until
        the first chunk.
    """

    def __init__(self) -> None:
        self.categories_: Optional[List[np.ndarray]] = None

    def partial_fit(self, X_chunk: np.ndarray) -> "StreamingOneHotEncoder":
        """
        Grow each column's category set with any values seen for the first time.

        Parameters
        ----------
        X_chunk : np.ndarray
            New rows, shape (n, D).

        Returns
        -------
        StreamingOneHotEncoder
            self, to allow chaining.

        Raises
        ------
        ValueError
            If X_chunk is not a 2D array, or its feature count does not
            match a previous chunk's.

        Complexity
        ----------
        Time: O(D * n log n) for sorting each column's unique values.
        Space: O(C) where C is the total number of categories seen.
        """
        X_chunk = np.asarray(X_chunk)
        if X_chunk.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X_chunk.ndim}D array instead.")

        if self.categories_ is None:
            self.categories_ = [np.unique(X_chunk[:, col]) for col in range(X_chunk.shape[1])]
        else:
            if X_chunk.shape[1] != len(self.categories_):
                raise ValueError(
                    f"Feature dimension mismatch: expected {len(self.categories_)} features, got {X_chunk.shape[1]}."
                )
            self.categories_ = [
                np.union1d(self.categories_[col], np.unique(X_chunk[:, col]))
                for col in range(X_chunk.shape[1])
            ]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X into one-hot numeric representation using broadcasting.

        Parameters
        ----------
        X : np.ndarray
            Data to encode, shape (n, D).

        Returns
        -------
        np.ndarray
            Binary one-hot encoded matrix, shape (n, total_categories_so_far).

        Raises
        ------
        RuntimeError
            If called before any partial_fit.
        ValueError
            If X is not a 2D array or its feature count does not match.

        Complexity
        ----------
        Time: O(n * C) where C is the total number of categories so far.
        Space: O(n * C) for the output array.
        """
        if self.categories_ is None:
            raise RuntimeError("StreamingOneHotEncoder is not fitted yet; call partial_fit first.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expect a 2D array")
        if X.shape[1] != len(self.categories_):
            raise ValueError("Feature dimension mismatch.")

        one_hot_blocks = []
        for col_idx, cats in enumerate(self.categories_):
            col_data = X[:, col_idx : col_idx + 1]
            encoded_block = (col_data == cats.reshape(1, -1)).astype(np.float64)
            one_hot_blocks.append(encoded_block)

        if not one_hot_blocks:
            return np.empty((X.shape[0], 0), dtype=np.float64)

        return np.hstack(one_hot_blocks)

    def partial_fit_transform(self, X_chunk: np.ndarray) -> np.ndarray:
        return self.partial_fit(X_chunk).transform(X_chunk)
