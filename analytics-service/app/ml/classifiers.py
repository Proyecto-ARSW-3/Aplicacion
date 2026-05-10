import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from dataclasses import dataclass, field


@dataclass
class TrainedModel:
    name: str
    model: object
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    training_time_ms: int
    feature_importances: dict[str, float] = field(default_factory=dict)
    roc_thresholds: list[float] = field(default_factory=list)
    roc_fpr: list[float] = field(default_factory=list)
    roc_tpr: list[float] = field(default_factory=list)
    optimal_threshold: float = 0.5


def _compute_metrics(model, X_test, y_test, model_name: str) -> dict:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)

    y_pred = (y_proba >= 0.5).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = float(thresholds[optimal_idx]) if len(thresholds) > optimal_idx else 0.5
    optimal_threshold = max(0.1, min(0.9, optimal_threshold))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
        "optimal_threshold": optimal_threshold,
    }


def train_logistic_regression(X_train, X_test, y_train, y_test) -> TrainedModel:
    start = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train, y_train)
    elapsed = int((time.time() - start) * 1000)

    metrics = _compute_metrics(model, X_test, y_test, "Logistic Regression")

    importances = {
        col: float(abs(coef))
        for col, coef in zip(X_train.columns, model.coef_[0])
    }

    return TrainedModel(
        name="Regresión Logística",
        model=model,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc_roc=metrics["auc_roc"],
        training_time_ms=elapsed,
        feature_importances=importances,
        roc_fpr=metrics["roc_fpr"],
        roc_tpr=metrics["roc_tpr"],
        roc_thresholds=metrics["roc_thresholds"],
        optimal_threshold=metrics["optimal_threshold"],
    )


def train_decision_tree(X_train, X_test, y_train, y_test) -> TrainedModel:
    start = time.time()
    model = DecisionTreeClassifier(max_depth=8, min_samples_split=20, random_state=42)
    model.fit(X_train, y_train)
    elapsed = int((time.time() - start) * 1000)

    metrics = _compute_metrics(model, X_test, y_test, "Decision Tree")

    importances = dict(zip(X_train.columns, model.feature_importances_))

    return TrainedModel(
        name="Árbol de Decisión",
        model=model,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc_roc=metrics["auc_roc"],
        training_time_ms=elapsed,
        feature_importances=importances,
        roc_fpr=metrics["roc_fpr"],
        roc_tpr=metrics["roc_tpr"],
        roc_thresholds=metrics["roc_thresholds"],
        optimal_threshold=metrics["optimal_threshold"],
    )


def train_random_forest(X_train, X_test, y_train, y_test) -> TrainedModel:
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = int((time.time() - start) * 1000)

    metrics = _compute_metrics(model, X_test, y_test, "Random Forest")

    importances = dict(zip(X_train.columns, model.feature_importances_))

    return TrainedModel(
        name="Random Forest",
        model=model,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc_roc=metrics["auc_roc"],
        training_time_ms=elapsed,
        feature_importances=importances,
        roc_fpr=metrics["roc_fpr"],
        roc_tpr=metrics["roc_tpr"],
        roc_thresholds=metrics["roc_thresholds"],
        optimal_threshold=metrics["optimal_threshold"],
    )


def train_svm(X_train, X_test, y_train, y_test) -> TrainedModel:
    start = time.time()
    model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    # Use a subset if dataset is large to keep training reasonable
    if len(X_train) > 3000:
        idx = np.random.choice(len(X_train), 3000, replace=False)
        X_train_sub = X_train.iloc[idx]
        y_train_sub = y_train[idx] if hasattr(y_train, '__getitem__') else np.array(y_train)[idx]
    else:
        X_train_sub, y_train_sub = X_train, y_train

    model.fit(X_train_sub, y_train_sub)
    elapsed = int((time.time() - start) * 1000)

    metrics = _compute_metrics(model, X_test, y_test, "SVM")

    # SVM has no direct feature importances; use zeros
    importances = {col: 0.0 for col in X_train.columns}

    return TrainedModel(
        name="SVM",
        model=model,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc_roc=metrics["auc_roc"],
        training_time_ms=elapsed,
        feature_importances=importances,
        roc_fpr=metrics["roc_fpr"],
        roc_tpr=metrics["roc_tpr"],
        roc_thresholds=metrics["roc_thresholds"],
        optimal_threshold=metrics["optimal_threshold"],
    )


def train_neural_network(X_train, X_test, y_train, y_test) -> TrainedModel:
    start = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)
    elapsed = int((time.time() - start) * 1000)

    metrics = _compute_metrics(model, X_test, y_test, "Neural Network")

    # Approximate importance using input layer weights
    w = np.abs(model.coefs_[0])
    col_importance = w.sum(axis=1)
    total = col_importance.sum() or 1.0
    importances = {col: float(val / total) for col, val in zip(X_train.columns, col_importance)}

    return TrainedModel(
        name="Red Neuronal (MLP)",
        model=model,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        auc_roc=metrics["auc_roc"],
        training_time_ms=elapsed,
        feature_importances=importances,
        roc_fpr=metrics["roc_fpr"],
        roc_tpr=metrics["roc_tpr"],
        roc_thresholds=metrics["roc_thresholds"],
        optimal_threshold=metrics["optimal_threshold"],
    )


def train_all_classifiers(X_train, X_test, y_train, y_test) -> list[TrainedModel]:
    trainers = [
        train_logistic_regression,
        train_decision_tree,
        train_random_forest,
        train_svm,
        train_neural_network,
    ]
    return [fn(X_train, X_test, y_train, y_test) for fn in trainers]
