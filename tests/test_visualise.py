"""
Unit tests for visualisation helpers (numcompute_stream/visualise.py).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


# 1. plot_metric_over_time

def test_plot_metric_over_time_returns_figure():
    fig = plot_metric_over_time([0.5, 0.6, 0.7])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_metric_over_time_sets_title_and_ylabel():
    fig = plot_metric_over_time([0.1, 0.2], title="Accuracy", ylabel="Accuracy")
    ax = fig.axes[0]
    assert ax.get_title() == "Accuracy"
    assert ax.get_ylabel() == "Accuracy"
    plt.close(fig)


def test_plot_metric_over_time_saves_to_file(tmp_path):
    save_path = tmp_path / "metric.png"
    fig = plot_metric_over_time([0.1, 0.2, 0.3], save_path=str(save_path))
    assert save_path.exists()
    plt.close(fig)


def test_plot_metric_over_time_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        plot_metric_over_time([])


def test_plot_metric_over_time_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        plot_metric_over_time([[0.1, 0.2], [0.3, 0.4]])


# 2. compare_models

def test_compare_models_returns_figure_with_two_lines():
    fig = compare_models([0.5, 0.6], [0.4, 0.5, 0.6], labels=("A", "B"))
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    plt.close(fig)


def test_compare_models_legend_uses_labels():
    fig = compare_models([0.5, 0.6], [0.4, 0.5], labels=("Buffer=10", "Buffer=100"))
    legend_texts = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert legend_texts == ["Buffer=10", "Buffer=100"]
    plt.close(fig)


def test_compare_models_rejects_empty_metric():
    with pytest.raises(ValueError, match="empty"):
        compare_models([], [0.1], labels=("A", "B"))


def test_compare_models_rejects_wrong_number_of_labels():
    with pytest.raises(ValueError, match="exactly 2"):
        compare_models([0.1], [0.2], labels=("A",))


def test_compare_models_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        compare_models([[0.1, 0.2]], [0.1, 0.2], labels=("A", "B"))


# 3. plot_predictions_vs_ground_truth

def test_plot_predictions_vs_ground_truth_returns_figure():
    fig = plot_predictions_vs_ground_truth([0, 1, 0, 1], [0, 1, 1, 1])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_predictions_vs_ground_truth_plots_both_series():
    fig = plot_predictions_vs_ground_truth([0, 1], [0, 0])
    ax = fig.axes[0]
    assert len(ax.collections) == 2  # one scatter series each for ground truth and prediction
    plt.close(fig)


def test_plot_predictions_vs_ground_truth_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        plot_predictions_vs_ground_truth([0, 1, 1], [0, 1])


def test_plot_predictions_vs_ground_truth_rejects_2d_input():
    with pytest.raises(ValueError, match="1D"):
        plot_predictions_vs_ground_truth([[0, 1]], [[0, 1]])
