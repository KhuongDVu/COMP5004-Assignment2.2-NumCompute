"""
Unit tests for the streaming trainer (numcompute_stream/stream.py).
"""

import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer
from numcompute_stream.preprocessing import StreamingStandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


# 1. Construction validation

def test_model_missing_partial_fit_or_predict_raises():
    with pytest.raises(TypeError, match="must implement both 'partial_fit' and 'predict'"):
        StreamTrainer(object())


# 2. fit_chunk / score_chunk logging

def test_fit_chunk_logs_chunk_accuracy():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    trainer = StreamTrainer(tree)

    log_entry = trainer.fit_chunk(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))

    assert log_entry["chunk_accuracy"] == pytest.approx(1.0)
    assert log_entry["cumulative_accuracy"] == pytest.approx(1.0)
    assert log_entry["memory_bytes"] > 0


def test_history_grows_in_call_order():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    trainer = StreamTrainer(tree)

    trainer.fit_chunk(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    trainer.fit_chunk(np.array([[1.0], [4.0]]), np.array([0, 1]))

    assert len(trainer.history) == 2
    assert trainer.history[0] is not trainer.history[1]


def test_cumulative_accuracy_matches_manual_running_average():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=4)
    trainer = StreamTrainer(tree)

    trainer.fit_chunk(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))  # all correct
    second_log = trainer.fit_chunk(np.array([[1.0], [1.0]]), np.array([0, 1]))  # one wrong

    # cumulative accuracy = correct so far / total so far = (4 + 1) / (4 + 2)
    assert second_log["cumulative_accuracy"] == pytest.approx(5 / 6)


def test_score_chunk_does_not_retrain_the_model():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=1)
    trainer = StreamTrainer(tree)
    trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0, 1]))  # buffer_size=1, retrains immediately
    root_after_fit = tree.root

    trainer.score_chunk(np.array([[1.0], [2.0]]), np.array([0, 1]))
    trainer.score_chunk(np.array([[1.0], [2.0]]), np.array([0, 1]))

    assert tree.root is root_after_fit


def test_score_chunk_on_empty_chunk_raises():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=1)
    trainer = StreamTrainer(tree)
    trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0, 1]))

    with pytest.raises(ValueError, match="empty chunk"):
        trainer.score_chunk(np.zeros((0, 1)), np.zeros(0))


# 3. Memory footprint

def test_memory_bytes_reflects_buffered_data():
    tree = DecisionTreeClassifier(max_depth=1, buffer_size=1000)
    tree.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))  # pre-train so predict works while buffering
    trainer = StreamTrainer(tree)

    small_log = trainer.fit_chunk(np.array([[1.0], [2.0]]), np.array([0, 1]))
    larger_log = trainer.fit_chunk(np.zeros((50, 1)), np.zeros(50, dtype=int))

    assert larger_log["memory_bytes"] > small_log["memory_bytes"]

