"""Tests for app.ml.classifiers — training functions and metric computation."""
import math
import numpy as np
import pytest

from app.ml.classifiers import (
    TrainedModel,
    _compute_metrics,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_svm,
    train_neural_network,
    train_all_classifiers,
)
from app.ml.preprocessor import DataPreprocessor


# ─── Shared training data fixture ────────────────────────────────────────────

@pytest.fixture(scope="module")
def split_data(synthetic_csv):
    """Pre-split matrices for classifier tests (module-scoped for speed)."""
    pp = DataPreprocessor()
    df = pp.load_and_clean(synthetic_csv)
    return pp.fit_transform(df)


# ─── _compute_metrics ─────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_all_metrics_present(self, split_data):
        from sklearn.linear_model import LogisticRegression
        X_tr, X_te, y_tr, y_te = split_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_tr, y_tr)
        metrics = _compute_metrics(model, X_te, y_te, "lr")
        for key in ("accuracy", "precision", "recall", "f1", "auc_roc",
                    "roc_fpr", "roc_tpr", "roc_thresholds", "optimal_threshold"):
            assert key in metrics

    def test_metrics_in_valid_range(self, split_data):
        from sklearn.linear_model import LogisticRegression
        X_tr, X_te, y_tr, y_te = split_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_tr, y_tr)
        m = _compute_metrics(model, X_te, y_te, "lr")
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["auc_roc"] <= 1.0
        assert 0.1 <= m["optimal_threshold"] <= 0.9

    def test_no_inf_in_thresholds(self, split_data):
        """Infinite thresholds (sklearn artefact) must be filtered out."""
        from sklearn.linear_model import LogisticRegression
        X_tr, X_te, y_tr, y_te = split_data
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_tr, y_tr)
        m = _compute_metrics(model, X_te, y_te, "lr")
        for t in m["roc_thresholds"]:
            assert math.isfinite(t), f"Non-finite threshold found: {t}"

    def test_works_with_decision_function_model(self, split_data):
        """SVC without probability also goes through the decision_function branch."""
        from sklearn.svm import SVC
        X_tr, X_te, y_tr, y_te = split_data
        model = SVC(kernel="linear", probability=False, random_state=42)
        model.fit(X_tr[:100], y_tr[:100])
        m = _compute_metrics(model, X_te, y_te, "svc_no_prob")
        assert "auc_roc" in m


# ─── Individual trainer functions ─────────────────────────────────────────────

class TestTrainLogisticRegression:
    def test_returns_trained_model(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_logistic_regression(X_tr, X_te, y_tr, y_te)
        assert isinstance(tm, TrainedModel)
        assert tm.name == "Regresión Logística"

    def test_feature_importances_populated(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_logistic_regression(X_tr, X_te, y_tr, y_te)
        assert len(tm.feature_importances) == X_tr.shape[1]
        assert all(v >= 0 for v in tm.feature_importances.values())

    def test_auc_roc_in_valid_range(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_logistic_regression(X_tr, X_te, y_tr, y_te)
        assert 0.0 <= tm.auc_roc <= 1.0

    def test_training_time_recorded(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_logistic_regression(X_tr, X_te, y_tr, y_te)
        assert tm.training_time_ms >= 0


class TestTrainDecisionTree:
    def test_returns_trained_model(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_decision_tree(X_tr, X_te, y_tr, y_te)
        assert isinstance(tm, TrainedModel)
        assert tm.name == "Árbol de Decisión"

    def test_feature_importances_sum_approx_one(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_decision_tree(X_tr, X_te, y_tr, y_te)
        total = sum(tm.feature_importances.values())
        assert abs(total - 1.0) < 0.01


class TestTrainRandomForest:
    def test_returns_trained_model(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_random_forest(X_tr, X_te, y_tr, y_te)
        assert isinstance(tm, TrainedModel)
        assert tm.name == "Random Forest"

    def test_roc_curve_not_empty(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_random_forest(X_tr, X_te, y_tr, y_te)
        assert len(tm.roc_fpr) > 0
        assert len(tm.roc_tpr) == len(tm.roc_fpr)


class TestTrainSVM:
    def test_returns_trained_model(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_svm(X_tr, X_te, y_tr, y_te)
        assert isinstance(tm, TrainedModel)
        assert tm.name == "SVM"

    def test_feature_importances_are_zeros(self, split_data):
        """SVM has no native feature importances — they default to 0."""
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_svm(X_tr, X_te, y_tr, y_te)
        assert all(v == 0.0 for v in tm.feature_importances.values())


class TestTrainNeuralNetwork:
    def test_returns_trained_model(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_neural_network(X_tr, X_te, y_tr, y_te)
        assert isinstance(tm, TrainedModel)
        assert tm.name == "Red Neuronal (MLP)"

    def test_importances_sum_approx_one(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        tm = train_neural_network(X_tr, X_te, y_tr, y_te)
        total = sum(tm.feature_importances.values())
        assert abs(total - 1.0) < 0.05


# ─── train_all_classifiers ────────────────────────────────────────────────────

class TestTrainAllClassifiers:
    def test_returns_five_models(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        models = train_all_classifiers(X_tr, X_te, y_tr, y_te)
        assert len(models) == 5

    def test_all_are_trained_model_instances(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        models = train_all_classifiers(X_tr, X_te, y_tr, y_te)
        assert all(isinstance(m, TrainedModel) for m in models)

    def test_unique_model_names(self, split_data):
        X_tr, X_te, y_tr, y_te = split_data
        models = train_all_classifiers(X_tr, X_te, y_tr, y_te)
        names = [m.name for m in models]
        assert len(names) == len(set(names))
