"""
Reproducible benchmarks for numcompute_stream: Python loops versus NumPy
split search in the decision tree, and buffer_size's effect on streaming
retraining time and accuracy.
"""

import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

# run from the project root instead
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from numcompute.benchmarking import compare_implementations
from numcompute_stream.tree import DecisionTreeClassifier


def best_split_for_feature_loop(column, y, impurity_fn):
    """
    Loop-based equivalent of DecisionTreeClassifier._best_split_for_feature:
    tries every candidate threshold one at a time, recomputing each side's
    impurity from scratch, instead of a vectorised cumulative-count pass.

    Parameters
    ----------
    column : np.ndarray
        One feature's values, shape (n,).
    y : np.ndarray
        Binary labels (0 or 1), shape (n,).
    impurity_fn : callable
        Impurity function (gini or entropy) applied to a group of labels.

    Returns
    -------
    Tuple[Optional[float], float]
        (best_threshold, best_gain). best_threshold is None if no split
        improves on the impurity before splitting.

    Complexity
    ----------
    Time: O(n^2) — n candidate cuts, each re-scanning its two sides.
    Space: O(n).
    """
    n_samples = len(column)
    order = np.argsort(column, kind="stable")
    sorted_column = column[order]
    sorted_labels = y[order]

    impurity_before = impurity_fn(y)
    best_threshold = None
    best_gain = 0.0

    for cut in range(n_samples - 1):
        if sorted_column[cut] == sorted_column[cut + 1]:
            continue

        left_labels = sorted_labels[:cut + 1]
        right_labels = sorted_labels[cut + 1:]
        n_left = cut + 1
        n_right = n_samples - n_left

        impurity_left = impurity_fn(left_labels)
        impurity_right = impurity_fn(right_labels)
        # gain = impurity before - the weighted impurity after
        impurity_after = (n_left * impurity_left + n_right * impurity_right) / n_samples
        gain = impurity_before - impurity_after

        if gain > best_gain:
            best_gain = gain
            best_threshold = (sorted_column[cut] + sorted_column[cut + 1]) / 2.0

    return best_threshold, best_gain


def make_streaming_dataset(n_samples, n_features=4, noise=0.1, seed=0):
    """
    Generate a reproducible synthetic binary classification dataset.

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    n_features : int, default=4
        Number of feature columns.
    noise : float, default=0.1
        Fraction of labels flipped at random, so no tree fits perfectly.
    seed : int, default=0
        Seed for the random generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X of shape (n_samples, n_features), y of shape (n_samples,).

    Complexity
    ----------
    Time: O(n_samples * n_features). Space: O(n_samples * n_features).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_label = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    flip = rng.random(n_samples) < noise
    y = np.where(flip, 1 - true_label, true_label)
    return X, y


def run_split_search_benchmark():
    print("Benchmark: Python loop vs NumPy decision tree split search")
    print("Each implementation: repeat=3, number=10")
    print("-" * 65)
    print(f"{'Samples':>10} {'Loop (ms)':>12} {'NumPy (ms)':>12} {'Speedup':>12}")

    tree = DecisionTreeClassifier(max_depth=1, buffer_size=1)
    rng = np.random.default_rng(1)
    results = []

    for n in [20, 100, 300, 600]:
        column = rng.normal(size=n)
        y = rng.integers(0, 2, size=n)

        loop_fn = lambda column, y: best_split_for_feature_loop(column, y, tree.impurity)
        vec_fn = lambda column, y: tree._best_split_for_feature(column, y)

        # Verify both implementations agree before timing them
        loop_threshold, loop_gain = loop_fn(column, y)
        vec_threshold, vec_gain = vec_fn(column, y)

        if not np.isclose(loop_threshold, vec_threshold) or not np.isclose(loop_gain, vec_gain):
            raise ValueError("Loop and vectorised split search produced different results.")

        timings = compare_implementations(loop_fn, vec_fn, column, y, repeat=3)

        row = {
            "samples": n,
            "loop_ms": timings["loop_time"] * 1000,
            "numpy_ms": timings["vec_time"] * 1000,
            "speedup": timings["speedup"],
        }
        results.append(row)

        print(
            f"{n:>10} "
            f"{row['loop_ms']:>12.4f} "
            f"{row['numpy_ms']:>12.4f} "
            f"{row['speedup']:>11.2f}x"
        )

    return results


def run_buffer_size_sweep():
    print("\nBenchmark: buffer_size sweep (streaming retraining time vs accuracy)")
    print("-" * 65)
    print(f"{'buffer_size':>12} {'n_retrains':>12} {'avg_retrain_ms':>16} {'test_accuracy':>14}")

    n_train = 5000
    chunk_size = 50
    X_train, y_train = make_streaming_dataset(n_train, seed=2)
    X_test, y_test = make_streaming_dataset(1000, seed=3)

    results = []

    for buffer_size in [50, 200, 500, 1000, 2000]:
        tree = DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)
        retrain_times = []

        for start in range(0, n_train, chunk_size):
            X_chunk = X_train[start:start + chunk_size]
            y_chunk = y_train[start:start + chunk_size]

            buffer_count_before = tree._buffer_count
            start_time = time.perf_counter()
            tree.partial_fit(X_chunk, y_chunk)
            elapsed = time.perf_counter() - start_time

            # retrain when buffer reaches buffer_size, according to partial_fit's retrain check
            if buffer_count_before + X_chunk.shape[0] >= buffer_size:
                retrain_times.append(elapsed)

        predictions = tree.predict(X_test)
        test_accuracy = np.mean(predictions == y_test)
        avg_retrain_ms = (np.mean(retrain_times) * 1000) if retrain_times else float("nan")

        row = {
            "buffer_size": buffer_size,
            "n_retrains": len(retrain_times),
            "avg_retrain_ms": avg_retrain_ms,
            "test_accuracy": test_accuracy,
        }
        results.append(row)

        print(
            f"{buffer_size:>12} "
            f"{row['n_retrains']:>12} "
            f"{avg_retrain_ms:>16.4f} "
            f"{test_accuracy:>14.4f}"
        )

    return results


def main():
    environment = (
        f"Python: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\n"
        f"Platform: {platform.platform()}\n"
        f"Machine: {platform.machine()}\n"
    )
    print(environment)

    split_search_results = run_split_search_benchmark()
    buffer_sweep_results = run_buffer_size_sweep()

    output_dir = Path(__file__).resolve().parent

    with (output_dir / "stream_split_search_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=split_search_results[0].keys())
        writer.writeheader()
        writer.writerows(split_search_results)

    with (output_dir / "stream_buffer_sweep_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=buffer_sweep_results[0].keys())
        writer.writeheader()
        writer.writerows(buffer_sweep_results)

    (output_dir / "stream_environment.txt").write_text(environment, encoding="utf-8")

    print("\nResults saved in benchmark/stream_split_search_results.csv")
    print("Results saved in benchmark/stream_buffer_sweep_results.csv")
    print("Environment saved in benchmark/stream_environment.txt")


if __name__ == "__main__":
    main()
