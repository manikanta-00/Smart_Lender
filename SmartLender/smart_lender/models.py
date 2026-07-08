"""
models.py
---------
Implements, FROM SCRATCH using only Python's standard library:

    - DecisionTreeClassifier   (CART, Gini impurity)
    - RandomForestClassifier   (bagging + feature subsampling of the tree above)
    - KNNClassifier            (k-nearest neighbours, euclidean distance)
    - GradientBoostingClassifier
        (an XGBoost-style additive ensemble of regression trees trained on
         the logistic-loss gradient -- same idea as XGBoost's boosting loop,
         just without the C++ engine / regularization extras)

These stand in for scikit-learn's DecisionTreeClassifier, RandomForestClassifier,
KNeighborsClassifier, and the xgboost.XGBClassifier mentioned in the project
brief, but require ZERO pip installs.
"""

import math
import random
from collections import Counter


# --------------------------------------------------------------------------
# Shared impurity helpers
# --------------------------------------------------------------------------
def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    impurity = 1.0
    for c in counts.values():
        p = c / n
        impurity -= p * p
    return impurity


def mse(values):
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


# --------------------------------------------------------------------------
# Decision Tree (classification) -- CART with Gini impurity
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=6, min_samples_split=6, n_features=None, random_state=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # number of features to consider per split (for RF)
        self.rng = random.Random(random_state)
        self.root = None
        self.n_total_features = None

    def fit(self, X, y):
        self.n_total_features = len(X[0])
        self.root = self._grow(X, y, depth=0)
        return self

    def _candidate_thresholds(self, X, fi, max_candidates=16):
        values = sorted(set(row[fi] for row in X))
        if len(values) < 2:
            return []
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        if len(thresholds) > max_candidates:
            step = len(thresholds) / max_candidates
            thresholds = [thresholds[int(i * step)] for i in range(max_candidates)]
        return thresholds

    def _best_split(self, X, y, feature_indices):
        n = len(y)
        parent_impurity = gini(y)
        best_gain, best_feat, best_thresh = 0.0, None, None
        for fi in feature_indices:
            for t in self._candidate_thresholds(X, fi):
                left_y = [y[i] for i in range(n) if X[i][fi] <= t]
                right_y = [y[i] for i in range(n) if X[i][fi] > t]
                if not left_y or not right_y:
                    continue
                p_left = len(left_y) / n
                p_right = 1 - p_left
                gain = parent_impurity - (p_left * gini(left_y) + p_right * gini(right_y))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, fi, t
        return best_feat, best_thresh, best_gain

    def _grow(self, X, y, depth):
        n = len(y)
        if n == 0:
            return _Node(value=0)
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]
        if depth >= self.max_depth or n < self.min_samples_split or len(counts) == 1:
            return _Node(value=majority)

        if self.n_features:
            k = min(self.n_features, self.n_total_features)
            feature_indices = self.rng.sample(range(self.n_total_features), k)
        else:
            feature_indices = list(range(self.n_total_features))

        feat, thresh, gain = self._best_split(X, y, feature_indices)
        if feat is None or gain <= 0:
            return _Node(value=majority)

        left_X, left_y, right_X, right_y = [], [], [], []
        for i in range(n):
            if X[i][feat] <= thresh:
                left_X.append(X[i]); left_y.append(y[i])
            else:
                right_X.append(X[i]); right_y.append(y[i])

        left_node = self._grow(left_X, left_y, depth + 1)
        right_node = self._grow(right_X, right_y, depth + 1)
        return _Node(feature_index=feat, threshold=thresh, left=left_node, right=right_node)

    def _predict_one(self, row):
        node = self.root
        while not node.is_leaf():
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return [self._predict_one(row) for row in X]


# --------------------------------------------------------------------------
# Random Forest -- bagging ensemble of the tree above
# --------------------------------------------------------------------------
class RandomForestClassifier:
    def __init__(self, n_estimators=25, max_depth=7, min_samples_split=6, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        n_features = len(X[0])
        n_sub_features = max(1, int(math.sqrt(n_features)))
        rng = random.Random(self.random_state)
        self.trees = []
        for i in range(self.n_estimators):
            sample_idx = [rng.randrange(n) for _ in range(n)]
            X_sample = [X[j] for j in sample_idx]
            y_sample = [y[j] for j in sample_idx]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_sub_features,
                random_state=self.random_state + i,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        results = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(len(self.trees))]
            results.append(Counter(votes).most_common(1)[0][0])
        return results

    def predict_proba_positive(self, X):
        """Fraction of trees voting class 1 -- used as a pseudo-probability."""
        all_preds = [tree.predict(X) for tree in self.trees]
        probs = []
        for i in range(len(X)):
            votes = [all_preds[t][i] for t in range(len(self.trees))]
            probs.append(sum(votes) / len(votes))
        return probs


# --------------------------------------------------------------------------
# K-Nearest Neighbours
# --------------------------------------------------------------------------
class KNNClassifier:
    def __init__(self, k=7):
        self.k = k
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    @staticmethod
    def _distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def _predict_one(self, row):
        dists = sorted(range(len(self.X)), key=lambda i: self._distance(row, self.X[i]))[: self.k]
        votes = [self.y[i] for i in dists]
        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict_one(row) for row in X]


# --------------------------------------------------------------------------
# Regression tree used internally by Gradient Boosting
# --------------------------------------------------------------------------
class _RegTreeNode:
    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class _DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, residuals):
        self.n_features = len(X[0])
        self.root = self._grow(X, residuals, depth=0)
        return self

    def _candidate_thresholds(self, X, fi, max_candidates=12):
        values = sorted(set(row[fi] for row in X))
        if len(values) < 2:
            return []
        thresholds = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        if len(thresholds) > max_candidates:
            step = len(thresholds) / max_candidates
            thresholds = [thresholds[int(i * step)] for i in range(max_candidates)]
        return thresholds

    def _best_split(self, X, residuals):
        n = len(residuals)
        parent_mse = mse(residuals)
        best_gain, best_feat, best_thresh = 1e-9, None, None
        for fi in range(self.n_features):
            for t in self._candidate_thresholds(X, fi):
                left = [residuals[i] for i in range(n) if X[i][fi] <= t]
                right = [residuals[i] for i in range(n) if X[i][fi] > t]
                if not left or not right:
                    continue
                p_left = len(left) / n
                p_right = 1 - p_left
                gain = parent_mse - (p_left * mse(left) + p_right * mse(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, fi, t
        return best_feat, best_thresh

    def _grow(self, X, residuals, depth):
        n = len(residuals)
        if n == 0:
            return _RegTreeNode(value=0.0)
        leaf_value = sum(residuals) / n
        if depth >= self.max_depth or n < self.min_samples_split:
            return _RegTreeNode(value=leaf_value)
        feat, thresh = self._best_split(X, residuals)
        if feat is None:
            return _RegTreeNode(value=leaf_value)
        left_X, left_r, right_X, right_r = [], [], [], []
        for i in range(n):
            if X[i][feat] <= thresh:
                left_X.append(X[i]); left_r.append(residuals[i])
            else:
                right_X.append(X[i]); right_r.append(residuals[i])
        left_node = self._grow(left_X, left_r, depth + 1)
        right_node = self._grow(right_X, right_r, depth + 1)
        return _RegTreeNode(feature_index=feat, threshold=thresh, left=left_node, right=right_node)

    def _predict_one(self, row):
        node = self.root
        while not node.is_leaf():
            node = node.left if row[node.feature_index] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return [self._predict_one(row) for row in X]


# --------------------------------------------------------------------------
# Gradient Boosting Classifier (XGBoost-style additive boosting)
# --------------------------------------------------------------------------
class GradientBoostingClassifier:
    """
    Fits an additive ensemble of shallow regression trees to the negative
    gradient of the logistic loss -- the same core algorithm XGBoost uses
    (minus its regularization / histogram-binning speedups).
    """

    def __init__(self, n_estimators=60, learning_rate=0.15, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = 0.0

    @staticmethod
    def _sigmoid(x):
        x = max(-35, min(35, x))
        return 1.0 / (1.0 + math.exp(-x))

    def fit(self, X, y):
        n = len(y)
        pos = max(sum(y), 1)
        neg = max(n - sum(y), 1)
        self.init_pred = math.log(pos / neg)
        F = [self.init_pred] * n
        self.trees = []
        for _ in range(self.n_estimators):
            p = [self._sigmoid(f) for f in F]
            residuals = [y[i] - p[i] for i in range(n)]
            tree = _DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            preds = tree.predict(X)
            F = [F[i] + self.learning_rate * preds[i] for i in range(n)]
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        F = [self.init_pred] * len(X)
        for tree in self.trees:
            preds = tree.predict(X)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X))]
        return [self._sigmoid(f) for f in F]

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
