"""
Streaming (chunk-based) statistics: running mean/variance and an approximate
running histogram/quantiles, updated as chunks arrive.
"""

from __future__ import annotations
from collections import deque
from typing import Optional, Sequence, Tuple, Union
import numpy as np


class StreamStats:
    """
    Online per-feature mean, variance, and histogram

    Running mean/variance use the parallel merge of Welford's algorithm: each chunk's own mean/var is computed in one vectorised pass,
    then merged into the running totals in O(D) time, D = feature number
    
    Parameters
    ----------
    bins : int, default=20
        Number of bins used for the running histogram / quantile estimate.
    bin_range : Optional[Tuple[float, float]], default=None
        Fixed (min, max) range for the histogram. If None, the range is
        taken from the first chunk seen and reused for every later chunk.
    window : Optional[int], default=None
        If given, only the last window chunks' histograms are kept and older chunks are dropped. 
        If None, the histogram accumulates over the whole stream.
    hist_feature : int, default=0
        Column index the histogram/quantiles track. Mean/var cover every feature

    Attributes
    ----------
    count : int
        Total number of valid (non-NaN) rows seen so far.
    mean : Optional[np.ndarray]
        Running per-feature mean. Shape (D,). None until the first update.
    var : Optional[np.ndarray]
        Running per-feature population variance (ddof=0). Shape (D,).
    """

    def __init__(
        self,
        bins: int = 20,
        bin_range: Optional[Tuple[float, float]] = None,
        window: Optional[int] = None,
        hist_feature: int = 0,
    ) -> None:
        if bins <= 0:
            raise ValueError("bins must be positive.")
        if window is not None and window <= 0:
            raise ValueError("window must be positive.")

        self.bins = bins
        self.bin_range = bin_range
        self.window = window
        self.hist_feature = hist_feature

        self.count = 0
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None

        self._bin_edges: Optional[np.ndarray] = None
        self._hist_counts: Optional[np.ndarray] = None
        self._hist_chunks: deque = deque(maxlen=window)

    def update_stats(self, X_chunk: np.ndarray) -> StreamStats:
        """
        Fold a new chunk of rows into the running mean/variance/histogram.

        Parameters
        ----------
        X_chunk : np.ndarray
            New rows, shape (n, D) or (n,) for a single feature. Rows
            containing NaN are dropped before updating mean/variance; the
            histogram drops NaNs in its own tracked column the same way.
        Returns
        -------
        StreamStats
            self, to allow chaining.

        Raises
        ------
        ValueError
            If X_chunk is not 1D/2D, or its feature count does not match a
            previous chunk's, or hist_feature is out of bounds for it.

        Complexity
        ----------
        Time: O(n * D) for the chunk's own mean/var/histogram.
        Space: O(D) extra for the running totals.
        """
        X_chunk = np.asarray(X_chunk, dtype=np.float64)
        if X_chunk.ndim == 1:
            X_chunk = X_chunk.reshape(-1, 1)
        if X_chunk.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D array, got {X_chunk.ndim}D array instead.")

        if self.mean is not None and X_chunk.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.mean.shape[0]}, got {X_chunk.shape[1]}."
            )
        if not (0 <= self.hist_feature < X_chunk.shape[1]):
            raise ValueError(
                f"hist_feature {self.hist_feature} is out of bounds for {X_chunk.shape[1]} features."
            )

        self._update_histogram(X_chunk)

        valid_rows = np.logical_not(np.isnan(X_chunk).any(axis=1))
        X_valid = X_chunk[valid_rows]
        n_new = X_valid.shape[0]
        if n_new == 0:
            return self

        chunk_mean = np.mean(X_valid, axis=0)
        chunk_var = np.var(X_valid, axis=0)  # ddof=0

        if self.count == 0:
            self.mean = chunk_mean
            self.var = chunk_var
        else:
            n_old = self.count
            total = n_old + n_new
            delta = chunk_mean - self.mean

            new_mean = self.mean + delta * (n_new / total)
            # parallel variance merge, vectorised over features
            m2_old = self.var * n_old
            m2_new = chunk_var * n_new
            m2_total = m2_old + m2_new + delta ** 2 * n_old * n_new / total

            self.mean = new_mean
            self.var = m2_total / total

        self.count += n_new
        return self

    def _update_histogram(self, X_chunk: np.ndarray) -> None:
        """
        Fold one chunk's tracked column into the running histogram.

        The bin range is fixed after the first chunk (or from `bin_range`).
        Later values that fall outside it are clamped into the nearest
        boundary bin rather than silently dropped, so `histogram()`'s and
        `quantiles()`'s total count always matches the valid rows seen.
        """
        col = X_chunk[:, self.hist_feature]
        col = col[np.logical_not(np.isnan(col))]
        if col.size == 0:
            return

        if self.bin_range is not None:
            edges = np.linspace(self.bin_range[0], self.bin_range[1], self.bins + 1)
        elif self._bin_edges is None:
            lo, hi = float(col.min()), float(col.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, self.bins + 1)
        else:
            edges = self._bin_edges

        col = np.clip(col, edges[0], edges[-1])
        counts, edges = np.histogram(col, bins=edges)
        self._bin_edges = edges

        if self.window is None:
            if self._hist_counts is None:
                self._hist_counts = counts
            else:
                self._hist_counts = self._hist_counts + counts
        else:
            self._hist_chunks.append(counts)
            self._hist_counts = np.sum(self._hist_chunks, axis=0)

    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the running histogram of the tracked column (`hist_feature`).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            `(counts, bin_edges)`, shapes (bins,) and (bins + 1,).

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        """
        if self._hist_counts is None:
            raise RuntimeError("No data seen, call update_stats first")
        return self._hist_counts, self._bin_edges

    def quantiles(self, q: Union[float, Sequence[float]]) -> Union[float, np.ndarray]:
        """
        Approximate quantile(s) of the tracked column from the running histogram.

        Parameters
        ----------
        q : float or Sequence[float]
            Quantile(s) in [0, 1].

        Returns
        -------
        float or np.ndarray
            Approximate quantile value(s), linearly interpolated from the cumulative histogram

        Raises
        ------
        ValueError
            If q is outside [0, 1].
        RuntimeError
            If no chunk seen

        Complexity
        ----------
        Time: O(bins + len(q)). Space: O(bins).
        """
        counts, edges = self.histogram()
        q_arr = np.asarray(q, dtype=np.float64)
        q_scalar = q_arr.ndim == 0
        q_flat = np.atleast_1d(q_arr)
        if np.any((q_flat < 0) | (q_flat > 1)):
            raise ValueError("q must be between 0 and 1.")

        total = counts.sum()
        if total == 0:
            result = np.full(q_flat.shape, np.nan)
        else:
            cum_counts = np.concatenate([[0.0], np.cumsum(counts)])
            cum_frac = cum_counts / total
            result = np.interp(q_flat, cum_frac, edges)

        if q_scalar:
            return float(result[0])
        else:
            return result
