"""
Streaming pipeline: chain transformers that support partial_fit/transform
with a final streaming model, driven one chunk at a time.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Pipeline:
    """
    Sequentially chain streaming transformers with a final streaming model.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transformer_or_model) tuples chaining the pipeline.
        Every step but the last must implement partial_fit and transform.

    Raises
    ------
    ValueError
        If steps is empty.
    TypeError
        If an intermediate step is missing partial_fit or transform, or the
        final step is missing partial_fit.
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline steps cannot be empty.")

        for name, step in steps[:-1]:
            if not hasattr(step, "partial_fit") or not hasattr(step, "transform"):
                raise TypeError(f"Intermediate step '{name}' must implement both 'partial_fit' and 'transform'.")

        final_name, final_step = steps[-1]
        if not hasattr(final_step, "partial_fit"):
            raise TypeError(f"Final step '{final_name}' must implement 'partial_fit'.")

        self.steps = steps

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline steps by name as a dictionary."""
        return dict(self.steps)

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: Optional[np.ndarray] = None) -> "Pipeline":
        """
        Fold one chunk through every transformer, then into the final step.

        Parameters
        ----------
        X_chunk : np.ndarray
            New rows, shape (n, D).
        y_chunk : Optional[np.ndarray], default=None
            Targets for this chunk, shape (n,). Only passed on to the final
            step; omit it when the final step is itself a transformer.

        Returns
        -------
        Pipeline
            self, to allow chaining.

        Raises
        ------
        Whatever each step's own partial_fit/transform raises (e.g. shape
        mismatches, NaNs) for the kind of chunk it was given.

        Complexity
        ----------
        Time: O(sum of each step's partial_fit + transform time).
        Space: O(size of the largest intermediate transformed chunk).
        """
        X_curr = X_chunk
        for _, step in self.steps[:-1]:
            step.partial_fit(X_curr)
            X_curr = step.transform(X_curr)

        _, last_step = self.steps[-1]
        if y_chunk is None:
            last_step.partial_fit(X_curr)
        else:
            last_step.partial_fit(X_curr, y_chunk)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Pass data through every step's transform, in order.

        Parameters
        ----------
        X : np.ndarray
            Data to transform, shape (n, D).

        Returns
        -------
        np.ndarray
            Transformed data from the final step.

        Raises
        ------
        AttributeError
            If any step (including the final one) does not implement
            transform.
        """
        X_curr = X
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise AttributeError(f"Step '{name}' does not implement 'transform'. Use predict() instead.")
            X_curr = step.transform(X_curr)

        return X_curr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X through every intermediate step, then predict with the final step.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on, shape (n, D).

        Returns
        -------
        np.ndarray
            Predictions from the final step.

        Raises
        ------
        AttributeError
            If the final step does not implement predict.
        """
        X_curr = X
        for _, step in self.steps[:-1]:
            X_curr = step.transform(X_curr)

        name, last_step = self.steps[-1]
        if not hasattr(last_step, "predict"):
            raise AttributeError(f"Final step '{name}' does not implement 'predict'.")

        return last_step.predict(X_curr)
