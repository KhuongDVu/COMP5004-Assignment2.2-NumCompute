"""
StreamTrainer:  Manage pipeline chunk by chunk, logging per-chunk accuracy, cumulative accuracy, and an approximate memory footprint.
"""

from __future__ import annotations
from typing import Any, Dict, List
import sys
import numpy as np

from numcompute_stream.metrics import StreamingAccuracy


class StreamTrainer:
    """
    Train and evaluate a streaming estimator one chunk at a time

    Parameters
    ----------
    model : Any
        A streaming estimator or Pipeline implementing partial_fit and predict.

    Attributes
    ----------
    history : List[Dict[str, float]]
        One entry per fit_chunk/score_chunk call, in call order: chunk_accuracy,
        cumulative_accuracy, memory_bytes.

    Raises
    ------
    TypeError
        If model does not implement both partial_fit and predict.
    """

    def __init__(self, model: Any) -> None:
        if not hasattr(model, "partial_fit") or not hasattr(model, "predict"):
            raise TypeError("model must implement both 'partial_fit' and 'predict'.")

        self.model = model
        self.history: List[Dict[str, float]] = []
        self._cumulative_accuracy = StreamingAccuracy()

    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Fold one chunk into the model, then score and log this same chunk.

        Parameters
        ----------
        X_chunk : np.ndarray
            New feature rows, shape (n, D).
        y_chunk : np.ndarray
            New labels, shape (n,).

        Returns
        -------
        Dict[str, float]
            The log entry appended to self.history (see score_chunk).

        Raises
        ------
        Whatever model.partial_fit raises for this kind of chunk (e.g.
        shape mismatches, NaNs).

        Complexity
        ----------
        Time: O(model.partial_fit time + score_chunk time).
        """
        self.model.partial_fit(X_chunk, y_chunk)
        return self.score_chunk(X_chunk, y_chunk)

    def score_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Score the model on a chunk, without training on it, and log the result.

        Parameters
        ----------
        X_chunk : np.ndarray
            Feature rows to score, shape (n, D).
        y_chunk : np.ndarray
            True labels for this chunk, shape (n,).

        Returns
        -------
        Dict[str, float]
            Log entry with keys:
            - chunk_accuracy: accuracy on this chunk alone.
            - cumulative_accuracy: accuracy over every chunk scored so far
              (via fit_chunk or score_chunk).
            - memory_bytes: approximate memory footprint of the model's
              numpy-backed state (see _estimate_memory_bytes).

        Raises
        ------
        ValueError
            If the chunk is empty.

        Complexity
        ----------
        Time: O(model.predict time + n). Space: O(1) beyond the prediction.
        """
        y_chunk = np.asarray(y_chunk)
        if y_chunk.shape[0] == 0:
            raise ValueError("Cannot score an empty chunk.")

        y_predicted = self.model.predict(X_chunk)

        # chunk accuracy = correctly predicted rows / chunk size
        chunk_correct = np.sum(y_chunk == y_predicted)
        chunk_accuracy = chunk_correct / y_chunk.shape[0]

        self._cumulative_accuracy.update(y_chunk, y_predicted)
        cumulative_accuracy = self._cumulative_accuracy.result()

        log_entry = {
            "chunk_accuracy": float(chunk_accuracy),
            "cumulative_accuracy": float(cumulative_accuracy),
            "memory_bytes": self._estimate_memory_bytes(),
        }
        self.history.append(log_entry)
        return log_entry

    def _estimate_memory_bytes(self) -> int:
        """
        Approximate the model's memory footprint from its own attributes.

        Only looks at the model (or, for a Pipeline, each of its steps)
        one level deep: numpy arrays are counted via nbytes, lists of numpy
        arrays (e.g. the tree's retraining buffer) via the sum of their
        nbytes, and everything else via sys.getsizeof. This is an estimate,
        not an exact accounting of every nested object.

        Returns
        -------
        int
            Approximate size in bytes.

        Complexity
        ----------
        Time: O(number of attributes across the model's step(s)).
        """
        steps = [("model", self.model)]
        if hasattr(self.model, "steps"):
            steps = self.model.steps

        total_bytes = 0
        for _, step in steps:
            for value in vars(step).values():
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes
                elif isinstance(value, list):
                    total_bytes += sum(item.nbytes for item in value if isinstance(item, np.ndarray))
                else:
                    total_bytes += sys.getsizeof(value)

        return total_bytes
