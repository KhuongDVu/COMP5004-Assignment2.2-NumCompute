"""
Streaming classification metrics: chunk-wise accuracy, confusion matrix, precision/recall/F1, and AUC
"""

from __future__ import annotations
from collections import deque
from typing import Deque, Optional, Tuple, Type
import numpy as np

from numcompute.metrics import roc_curve
from numcompute.metrics import auc as auc_score


def _check_1d_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shared shape guard for every streaming metric below."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Inputs must have the same shape.")
    return y_true, y_pred


class StreamingAccuracy:
    """
    Running accuracy accumulated chunk by chunk.

    Attributes
    ----------
    correct : int
        Total number of correct predictions seen so far.
    total : int
        Total number of predictions seen so far.
    """

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamingAccuracy":
        """
        Fold one chunk of predictions into the running accuracy.

        Parameters
        ----------
        y_true_chunk : np.ndarray
            Actual labels for this chunk, shape (n,).
        y_pred_chunk : np.ndarray
            Predicted labels for this chunk, shape (n,).

        Returns
        -------
        StreamingAccuracy
            self, to allow chaining.

        Raises
        ------
        ValueError
            If inputs are not 1D or have different shapes.

        Complexity
        ----------
        Time: O(n). Space: O(1).
        """
        y_true_chunk, y_pred_chunk = _check_1d_same_shape(y_true_chunk, y_pred_chunk)
        self.correct += int(np.sum(y_true_chunk == y_pred_chunk))
        self.total += y_true_chunk.size
        return self

    def reset(self) -> None:
        """Clear the running correct/total counts."""
        self.correct = 0
        self.total = 0

    def result(self) -> float:
        """
        Return the accuracy over every chunk seen so far.

        Returns
        -------
        float
            correct / total.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        """
        if self.total == 0:
            raise RuntimeError("No data seen yet; call update first.")
        else:
            return self.correct / self.total


class StreamingConfusionMatrix:
    """
    Confusion matrix accumulated chunk by chunk.

    Attributes
    ----------
    classes : Optional[np.ndarray]
        Sorted class labels seen so far. None until the first update.
    matrix : Optional[np.ndarray]
        Running confusion matrix, shape (n_classes, n_classes). Rows are
        actual labels, columns are predicted labels, ordered like `classes`.
    """

    def __init__(self) -> None:
        self.classes: Optional[np.ndarray] = None
        self.matrix: Optional[np.ndarray] = None

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamingConfusionMatrix":
        """
        Fold one chunk of predictions into the running confusion matrix.

        Parameters
        ----------
        y_true_chunk : np.ndarray
            Actual labels for this chunk, shape (n,).
        y_pred_chunk : np.ndarray
            Predicted labels for this chunk, shape (n,).

        Returns
        -------
        StreamingConfusionMatrix
            self, to allow chaining.

        Raises
        ------
        ValueError
            If inputs are not 1D or have different shapes.

        Complexity
        ----------
        Time: O(n + c^2), c = num of distinct classes
        Space: O(c^2) for the matrix.
        """
        y_true_chunk, y_pred_chunk = _check_1d_same_shape(y_true_chunk, y_pred_chunk)
        chunk_classes = np.unique(np.concatenate((y_true_chunk, y_pred_chunk)))

        if self.classes is None:
            new_classes = chunk_classes
        else:
            new_classes = np.union1d(self.classes, chunk_classes)

        if self.matrix is None:
            self.matrix = np.zeros((new_classes.size, new_classes.size), dtype=int)
        elif new_classes.size != self.classes.size:
            # copy old counts into their new positions in the grown matrix
            grown = np.zeros((new_classes.size, new_classes.size), dtype=int)
            old_positions = np.searchsorted(new_classes, self.classes)
            grown[np.ix_(old_positions, old_positions)] = self.matrix
            self.matrix = grown

        self.classes = new_classes
        # map label to position in the matrix
        true_indices = np.searchsorted(self.classes, y_true_chunk)
        predicted_indices = np.searchsorted(self.classes, y_pred_chunk)
        np.add.at(self.matrix, (true_indices, predicted_indices), 1)
        return self

    def reset(self) -> None:
        """Clear the running classes/matrix."""
        self.classes = None
        self.matrix = None

    def result(self) -> np.ndarray:
        """
        Return the confusion matrix over every chunk seen so far.

        Returns
        -------
        np.ndarray
            Confusion matrix, shape (n_classes, n_classes).

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        """
        if self.matrix is None:
            raise RuntimeError("No data seen yet; call update first.")
        else:
            return self.matrix


class StreamingPrecisionRecallF1:
    """
    Running precision, recall, and F1 for binary (0/1) classification.

    Attributes
    ----------
    true_positive, false_positive, false_negative : int
        Running true-positive, false-positive, and false-negative counts.
    """

    def __init__(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "StreamingPrecisionRecallF1":
        """
        Fold one chunk of predictions into the running true/false positive/negative counts.

        Parameters
        ----------
        y_true_chunk : np.ndarray
            Actual binary labels (0 or 1) for this chunk, shape (n,).
        y_pred_chunk : np.ndarray
            Predicted binary labels (0 or 1) for this chunk, shape (n,).

        Returns
        -------
        StreamingPrecisionRecallF1
            self, to allow chaining.

        Raises
        ------
        ValueError
            If inputs are not 1D, have different shapes, or contain labels
            other than 0 and 1.

        Complexity
        ----------
        Time: O(n). Space: O(1).
        """
        y_true_chunk, y_pred_chunk = _check_1d_same_shape(y_true_chunk, y_pred_chunk)
        true_is_binary = np.all(np.isin(y_true_chunk, [0, 1]))
        pred_is_binary = np.all(np.isin(y_pred_chunk, [0, 1]))
        if not (true_is_binary and pred_is_binary):
            raise ValueError("Labels must be 0 or 1.")

        # count correct and incorrect positive predictions in this chunk
        self.true_positive += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 1)))
        self.false_positive += int(np.sum((y_true_chunk == 0) & (y_pred_chunk == 1)))
        self.false_negative += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 0)))
        return self

    def reset(self) -> None:
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def result(self) -> Tuple[float, float, float]:
        """
        Return (precision, recall, f1) over every chunk seen so far.

        Returns
        -------
        Tuple[float, float, float]
            precision, recall, f1. Each is 0.0 if its own denominator is 0.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        """
        if self.true_positive + self.false_positive + self.false_negative == 0:
            raise RuntimeError("No data seen yet; call update first.")

        if self.true_positive + self.false_positive == 0:
            precision = 0.0
        else:
            precision = self.true_positive / (self.true_positive + self.false_positive)

        if self.true_positive + self.false_negative == 0:
            recall = 0.0
        else:
            recall = self.true_positive / (self.true_positive + self.false_negative)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1


class StreamingAUC:
    """
    Running AUC for binary classification.

    this class recomputes the ROC curve/AUC on `result()` using `numcompute.metrics.roc_curve`/`auc`.
    """

    def __init__(self) -> None:
        self._y_true: list = []
        self._y_score: list = []

    def update(self, y_true_chunk: np.ndarray, y_score_chunk: np.ndarray) -> "StreamingAUC":
        """
        Append one chunk of (label, score) pairs to the running buffer.

        Parameters
        ----------
        y_true_chunk : np.ndarray
            Actual binary labels (0 or 1) for this chunk, shape (n,).
        y_score_chunk : np.ndarray
            Prediction scores for this chunk, shape (n,).

        Returns
        -------
        StreamingAUC
            self, to allow chaining.

        Raises
        ------
        ValueError
            If inputs are not 1D or have different shapes.

        Complexity
        ----------
        Time: O(n) to append. Space: O(N) total, N = rows seen so far.
        """
        y_true_chunk, y_score_chunk = _check_1d_same_shape(y_true_chunk, y_score_chunk)
        self._y_true.append(np.asarray(y_true_chunk, dtype=float))
        self._y_score.append(np.asarray(y_score_chunk, dtype=float))
        return self

    def reset(self) -> None:
        self._y_true = []
        self._y_score = []

    def result(self) -> float:
        """
        Return the AUC over every (label, score) pair seen so far.

        Returns
        -------
        float
            Area under the ROC curve.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.
        ValueError
            If only one class has been seen so far (raised by `roc_curve`).

        Complexity
        ----------
        Time: O(N log N), N = rows seen so far. Space: O(N).
        """
        if not self._y_true:
            raise RuntimeError("No data seen yet; call update first.")

        y_true = np.concatenate(self._y_true)
        y_score = np.concatenate(self._y_score)
        # get ROC curve, then integrate it to get the AUC
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        return auc_score(false_positive_rate, true_positive_rate)


class RollingMetric:
    """
    Wrap a streaming metric class to only consider the last `window` chunks.

    Parameters
    ----------
    metric_cls : type
        A streaming metric class from this module (e.g. StreamingAccuracy),
        constructible with no arguments and exposing update()/result().
    window : int
        Number of most recent chunks to keep.
    """

    def __init__(self, metric_cls: Type, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.metric_cls = metric_cls
        self.window = window
        self._chunks: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=window)

    def update(self, y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray) -> "RollingMetric":
        """
        Record one chunk, dropping the oldest chunk once `window` is exceeded.

        Parameters
        ----------
        y_true_chunk : np.ndarray
            Actual labels/scores for this chunk, shape (n,).
        y_pred_chunk : np.ndarray
            Predicted labels/scores for this chunk, shape (n,).

        Returns
        -------
        RollingMetric
            self, to allow chaining.

        Raises
        ------
        ValueError
            If inputs are not 1D or have different shapes.

        Complexity
        ----------
        Time: O(n). Space: O(window) chunks kept.
        """
        y_true_chunk, y_pred_chunk = _check_1d_same_shape(y_true_chunk, y_pred_chunk)
        self._chunks.append((y_true_chunk, y_pred_chunk))
        return self

    def reset(self) -> None:
        self._chunks.clear()

    def result(self):
        """
        Recompute the wrapped metric from scratch over the chunks still in the window.

        Returns
        -------
        Whatever `metric_cls().result()` returns.

        Raises
        ------
        RuntimeError
            If no chunk has been seen yet.

        Complexity
        ----------
        Time: O(window * n) to replay the window. Space: O(window * n).
        """
        if not self._chunks:
            raise RuntimeError("No data seen yet; call update first.")

        metric = self.metric_cls()
        for y_true_chunk, y_pred_chunk in self._chunks:
            metric.update(y_true_chunk, y_pred_chunk)
        return metric.result()
