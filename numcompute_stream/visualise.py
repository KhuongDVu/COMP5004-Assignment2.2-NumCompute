from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from numcompute_stream.metrics import _check_1d_same_shape


def plot_metric_over_time(
    metric_values: Sequence[float],
    title: str = "Metric over time",
    ylabel: str = "Metric",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a metric across chunks as a line chart.

    Parameters
    ----------
    metric_values : Sequence[float]
        One value per chunk, shape (n_chunks,).
    title : str, default='Metric over time'
        Plot title.
    ylabel : str, default='Metric'
        Y-axis label.
    save_path : Optional[str], default=None
        If given, the figure is saved to this path. 

    Returns
    -------
    plt.Figure
        The created figure.

    Raises
    ------
    ValueError
        If metric_values is not 1D or is empty.

    Complexity
    ----------
    Time: O(n_chunks) to draw the line. Space: O(n_chunks).
    """
    metric_values = np.asarray(metric_values, dtype=np.float64)
    if metric_values.ndim != 1:
        raise ValueError(f"Expected a 1D array of metric values, got {metric_values.ndim}D array instead.")
    if metric_values.size == 0:
        raise ValueError("Cannot plot an empty sequence of metric values.")

    chunk_indices = np.arange(metric_values.size)

    fig, ax = plt.subplots()
    ax.plot(chunk_indices, metric_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def compare_models(
    metric1: Sequence[float],
    metric2: Sequence[float],
    labels: Tuple[str, str],
    title: str = "Model comparison",
    ylabel: str = "Metric",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare two models' streaming metrics over chunks.

    Parameters
    ----------
    metric1 : Sequence[float]
        First model's per-chunk metric values, shape (n_chunks_1,).
    metric2 : Sequence[float]
        Second model's per-chunk metric values, shape (n_chunks_2,).
        May have a different length than metric1.
    labels : Tuple[str, str]
        Legend labels for (metric1, metric2).
    title : str, default='Model comparison'
        Plot title.
    ylabel : str, default='Metric'
        Y-axis label.
    save_path : Optional[str], default=None
        If given, the figure is saved to this path

    Returns
    -------
    plt.Figure
        The created figure.

    Raises
    ------
    ValueError
        If either metric sequence is not 1D or is empty, or labels does not
        contain exactly 2 entries.

    Complexity
    ----------
    Time: O(n_chunks_1 + n_chunks_2). Space: O(n_chunks_1 + n_chunks_2).
    """
    metric1 = np.asarray(metric1, dtype=np.float64)
    metric2 = np.asarray(metric2, dtype=np.float64)
    if metric1.ndim != 1 or metric2.ndim != 1:
        raise ValueError("Expected 1D arrays of metric values for both models.")
    if metric1.size == 0 or metric2.size == 0:
        raise ValueError("Cannot plot an empty sequence of metric values.")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly 2 entries, one per model.")

    fig, ax = plt.subplots()
    ax.plot(np.arange(metric1.size), metric1, marker="o", label=labels[0])
    ax.plot(np.arange(metric2.size), metric2, marker="o", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path)

    return fig


def plot_predictions_vs_ground_truth(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    title: str = "Predictions vs ground truth",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise predictions against actual labels for one chunk.

    Parameters
    ----------
    y_true : Sequence[float]
        Actual labels, shape (n,).
    y_pred : Sequence[float]
        Predicted labels, shape (n,).
    title : str, default='Predictions vs ground truth'
        Plot title.
    save_path : Optional[str], default=None
        If given, the figure is saved to this path

    Returns
    -------
    plt.Figure
        The created figure.

    Raises
    ------
    ValueError
        If y_true/y_pred are not 1D or have different shapes.

    Complexity
    ----------
    Time: O(n). Space: O(n).
    """
    y_true, y_pred = _check_1d_same_shape(y_true, y_pred)
    sample_indices = np.arange(y_true.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(sample_indices, y_true, marker="o", facecolors="none", edgecolors="tab:blue", label="Ground truth")
    ax.scatter(sample_indices, y_pred, marker="x", color="tab:orange", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Label")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path)

    return fig
