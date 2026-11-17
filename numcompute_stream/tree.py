"""
A from-scratch binary decision tree classifier (Gini or entropy), generalising
classification-tree.html's 0/1-feature design to numeric threshold splits.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def gini(y: np.ndarray) -> float:
    """
    Gini impurity of a group of binary (0/1) labels: 0 when everyone agrees, 0.5 at half and half.

    Parameters
    ----------
    y : np.ndarray
        Binary labels (0 or 1), shape (n,).

    Returns
    -------
    float
        Gini impurity.

    Complexity
    ----------
    Time: O(n). Space: O(1).
    """
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return float(1.0 - (p ** 2).sum())


def entropy(y: np.ndarray) -> float:
    """
    Entropy of a group of binary (0/1) labels, in bits: 0 when everyone agrees, 1 at half and half.

    Parameters
    ----------
    y : np.ndarray
        Binary labels (0 or 1), shape (n,).

    Returns
    -------
    float
        Entropy in bits.

    Complexity
    ----------
    Time: O(n). Space: O(1).
    """
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]  # log2(0) is undefined
    return float(-(p * np.log2(p)).sum())


def _gini_from_counts(n: np.ndarray, n_positive: np.ndarray) -> np.ndarray:
    """
    Vectorised Gini impurity for many candidate splits at once, from counts.

    Parameters
    ----------
    n : np.ndarray
        Number of samples on one side of each candidate split.
    n_positive : np.ndarray
        Number of label-1 samples on that same side of each candidate split.

    Returns
    -------
    np.ndarray
        Gini impurity for each candidate split, same shape as n.

    Complexity
    ----------
    Time: O(len(n)). Space: O(len(n)).
    """
    positive_fraction = n_positive / n
    negative_fraction = 1.0 - positive_fraction
    return 1.0 - (negative_fraction ** 2 + positive_fraction ** 2)


def _entropy_from_counts(n: np.ndarray, n_positive: np.ndarray) -> np.ndarray:
    """
    Vectorised entropy

    Parameters
    ----------
    n : np.ndarray
        Number of samples on one side of each candidate split.
    n_positive : np.ndarray
        Number of label-1 samples on that same side of each candidate split.

    Returns
    -------
    np.ndarray
        Entropy for each candidate split, same shape as n.

    Complexity
    ----------
    Time: O(len(n)). Space: O(len(n)).
    """
    positive_fraction = n_positive / n
    negative_fraction = 1.0 - positive_fraction

    # log2(0) is undefined, so each side's = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        negative_term = np.where(negative_fraction > 0, negative_fraction * np.log2(negative_fraction), 0.0)
        positive_term = np.where(positive_fraction > 0, positive_fraction * np.log2(positive_fraction), 0.0)

    return -(negative_term + positive_term)


class Node:
    """A question with two children, or a leaf with an answer."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        label: Optional[int] = None,
    ) -> None:
        self.feature = feature    
        self.threshold = threshold
        self.left = left         
        self.right = right 
        self.label = label


class DecisionTreeClassifier:
    """
    Depth-limited binary decision tree for binary classification.

    Parameters
    ----------
    max_depth : int, default=2
        Maximum depth of the tree. The root is depth 0.
    criterion : str, default='gini'
        Impurity measure used to score splits. Supported: 'gini', 'entropy'.
    min_samples_split : int, default=2
        A node with fewer rows than this is never split.
    max_features : Optional[int], default=None
        If given, only this many randomly chosen features are considered at
        each split. If None, every feature is considered.
    random_state : Optional[int], default=None
        Seed for the random feature subset chosen when max_features is set.

    Attributes
    ----------
    root : Optional[Node]
        The trained tree's root node. None until fit is called.
    """

    def __init__(
        self,
        max_depth: int = 2,
        criterion: str = "gini",
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("Unknown criterion. Choose 'gini' or 'entropy'.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if max_features is not None and max_features <= 0:
            raise ValueError("max_features must be positive.")

        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.impurity = gini if criterion == "gini" else entropy
        self._impurity_from_counts = _gini_from_counts if criterion == "gini" else _entropy_from_counts

        self.root: Optional[Node] = None

    def _best_split_for_feature(self, column: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], float]:
        """
        Search every candidate threshold for one feature column.
        Candidates are evaluated all at once via cumulative class counts along the sorted column

        Parameters
        ----------
        column : np.ndarray
            One feature's values, shape (n,).
        y : np.ndarray
            Binary labels (0 or 1), shape (n,).

        Returns
        -------
        Tuple[Optional[float], float]
            (best_threshold, best_gain). best_threshold is None if no split
            improves on the impurity before splitting.

        Complexity
        ----------
        Time: O(n log n) for the sort. Space: O(n).
        """
        n_samples = column.shape[0]
        order = np.argsort(column, kind="stable")
        sorted_column = column[order]
        sorted_labels = y[order]

        # running count of label-1 from sorted col start
        cumulative_positive = np.cumsum(sorted_labels)
        total_positive = cumulative_positive[-1]

        # cut between rows with different values
        distinct_positions = np.flatnonzero(sorted_column[:-1] != sorted_column[1:])
        if distinct_positions.size == 0:
            return None, 0.0

        n_left = distinct_positions + 1
        n_right = n_samples - n_left
        positive_left = cumulative_positive[distinct_positions]
        positive_right = total_positive - positive_left

        impurity_left = self._impurity_from_counts(n_left, positive_left)
        impurity_right = self._impurity_from_counts(n_right, positive_right)

        # gain = impurity before - the weighted impurity after
        impurity_before = self.impurity(y)
        impurity_after = (n_left * impurity_left + n_right * impurity_right) / n_samples
        gains = impurity_before - impurity_after

        best_index = np.argmax(gains)
        best_gain = gains[best_index]
        if best_gain <= 0.0:
            return None, 0.0

        cut = distinct_positions[best_index]
        best_threshold = (sorted_column[cut] + sorted_column[cut + 1]) / 2.0
        return float(best_threshold), float(best_gain)

    def _best_feature(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Ask every feature, keep the one whose gain is largest.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n, D).
        y : np.ndarray
            Binary labels (0 or 1), shape (n,).

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            (best_feature, best_threshold). best_feature is None if no
            feature has a split that improves on the impurity before
            splitting.

        Complexity
        ----------
        Time: O(D * n log n) across the candidate features.
        Space: O(n) for the per-feature sort.
        """
        n_features = X.shape[1]
        if self.max_features is None or self.max_features >= n_features:
            candidate_features = np.arange(n_features)
        else:
            candidate_features = self._rng.choice(n_features, size=self.max_features, replace=False)

        best_feature, best_threshold, best_gain = None, None, 0.0
        for feature in candidate_features:
            threshold, gain = self._best_split_for_feature(X[:, feature], y)
            if threshold is None:
                continue
            if gain > best_gain:
                best_feature, best_threshold, best_gain = int(feature), threshold, gain

        return best_feature, best_threshold

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively grow the tree: check the stop rules, else split on the best feature.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for this node, shape (n, D).
        y : np.ndarray
            Binary labels (0 or 1) for this node, shape (n,).
        depth : int
            Depth of this node, the root is depth 0.

        Returns
        -------
        Node
            The built subtree, rooted at this node.

        Complexity
        ----------
        Time: O(D * n log n) per node for the feature search.
        """
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return Node(label=np.bincount(y).argmax())

        feature, threshold = self._best_feature(X, y)
        if feature is None:
            return Node(label=np.bincount(y).argmax())

        mask = X[:, feature] <= threshold
        not_mask = np.logical_not(mask)
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[not_mask], y[not_mask], depth + 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Train the tree from scratch on the given data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n, D).
        y : np.ndarray
            Binary labels (0 or 1), shape (n,).

        Returns
        -------
        DecisionTreeClassifier
            self, to allow chaining.

        Raises
        ------
        ValueError
            If X is not 2D, y is not 1D, their row counts differ, X is
            empty, X contains non-finite values, or y contains labels other
            than 0 or 1.

        Complexity
        ----------
        Time: O(max_depth * D * n log n). Space: O(n) per level of recursion.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array for X, got {X.ndim}D array instead.")
        if y.ndim != 1:
            raise ValueError(f"Expected a 1D array for y, got {y.ndim}D array instead.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows: got {X.shape[0]} and {y.shape[0]}.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only labels 0 or 1.")

        y = y.astype(np.int64)
        self.root = self._build(X, y, 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for X by walking the trained tree.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n, D).

        Returns
        -------
        np.ndarray
            Predicted binary labels, shape (n,).

        Raises
        ------
        RuntimeError
            If called before fit.
        ValueError
            If X is not a 2D array or contains non-finite values.

        Complexity
        ----------
        Time: O(n * max_depth) to walk each row down the tree. Space: O(n).
        """
        if self.root is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet, call fit first.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Expect a 2D array")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values.")

        predictions = np.empty(X.shape[0], dtype=np.int64)
        for row_idx, row in enumerate(X):
            node = self.root
            while node.label is None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[row_idx] = node.label

        return predictions
