"""Central model manager: loads data, trains all models, and serves predictions."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from app.ml.preprocessor import DataPreprocessor
from app.ml.classifiers import TrainedModel, train_all_classifiers
from app.ml.survival import SurvivalResults, compute_kaplan_meier


class ModelManager:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models: list[TrainedModel] = []
        self.best_model: Optional[TrainedModel] = None
        self.survival_results: Optional[SurvivalResults] = None
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_all_proba: Optional[pd.DataFrame] = None
        self.threshold: float = 0.5
        self.is_trained: bool = False
        self.total_samples: int = 0
        self.dropout_samples: int = 0

    def train(self, dataset_path: str) -> dict:
        """Load dataset, preprocess, train all classifiers, compute survival."""
        self.df_raw = self.preprocessor.load_and_clean(dataset_path)

        self.total_samples = len(self.df_raw)
        self.dropout_samples = int((self.df_raw["Target"] == 0).sum())

        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(self.df_raw)

        self.models = train_all_classifiers(X_train, X_test, y_train, y_test)

        # Select best model by AUC-ROC
        self.best_model = max(self.models, key=lambda m: m.auc_roc)

        # Set optimal threshold from best model
        self.threshold = self.best_model.optimal_threshold

        # Compute survival analysis
        self.survival_results = compute_kaplan_meier(self.df_raw)

        # Pre-compute predictions for all students
        self._compute_all_predictions()

        self.is_trained = True
        return {
            "models_trained": [m.name for m in self.models],
            "total_samples": self.total_samples,
            "dropout_samples": self.dropout_samples,
        }

    def _compute_all_predictions(self):
        """Pre-compute dropout probabilities for every student in the dataset."""
        X_full = self.preprocessor.get_full_dataset_features(self.df_raw)

        if self.best_model and hasattr(self.best_model.model, "predict_proba"):
            proba = self.best_model.model.predict_proba(X_full)[:, 1]
        else:
            proba = np.zeros(len(X_full))

        # Map target back to labels
        target_map = {0: "Desertor", 1: "Graduado", 2: "Activo"}
        labels = self.df_raw["Target"].map(target_map)

        self.df_all_proba = self.df_raw.copy()
        self.df_all_proba["dropout_proba"] = proba
        self.df_all_proba["actual_class"] = labels

    def get_students_at_risk(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """Return student records with risk scores, filtered by threshold."""
        if self.df_all_proba is None:
            return pd.DataFrame()

        t = threshold if threshold is not None else self.threshold
        df = self.df_all_proba.copy()
        df["is_at_risk"] = df["dropout_proba"] >= t
        df["risk_level"] = df["dropout_proba"].apply(_risk_label)
        return df.sort_values("dropout_proba", ascending=False)

    def update_threshold(self, new_threshold: float) -> dict:
        """Update classification threshold and return updated stats."""
        self.threshold = new_threshold
        df = self.get_students_at_risk(new_threshold)

        # Compute sensitivity/specificity if we have binary labels
        actual_dropout = (self.df_all_proba["actual_class"] == "Desertor").values
        predicted_dropout = (df["dropout_proba"] >= new_threshold).values

        tp = int(np.sum(actual_dropout & predicted_dropout))
        fn = int(np.sum(actual_dropout & ~predicted_dropout))
        fp = int(np.sum(~actual_dropout & predicted_dropout))
        tn = int(np.sum(~actual_dropout & ~predicted_dropout))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return {
            "threshold": new_threshold,
            "at_risk_count": int(df["is_at_risk"].sum()),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
        }

    def predict_single(self, features: dict) -> dict:
        """Predict dropout probability for a single student."""
        if not self.is_trained or self.best_model is None:
            return {"error": "Models not trained yet"}

        X = pd.DataFrame([features])
        # Align columns with trained feature set
        for col in self.preprocessor.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.preprocessor.feature_names]
        X_scaled = self.preprocessor.transform(X)

        if hasattr(self.best_model.model, "predict_proba"):
            proba = float(self.best_model.model.predict_proba(X_scaled)[0, 1])
        else:
            proba = 0.5

        # Top risk factors from feature importances
        fi = self.best_model.feature_importances
        top_factors = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "dropout_probability": round(proba, 4),
            "risk_level": _risk_label(proba),
            "recommended_threshold": round(self.threshold, 2),
            "prediction": "Desertor" if proba >= self.threshold else "No Desertor",
            "top_risk_factors": [
                {"feature": self.preprocessor.get_display_name(f), "importance": round(v, 4)}
                for f, v in top_factors
            ],
        }

    def get_roc_analysis(self, model_name: Optional[str] = None) -> list[dict]:
        """Return ROC curve points for a model (default: best model)."""
        model = self.best_model
        if model_name:
            found = next((m for m in self.models if m.name == model_name), None)
            if found:
                model = found

        if not model or not model.roc_thresholds:
            return []

        fpr = model.roc_fpr
        tpr = model.roc_tpr
        thresholds = model.roc_thresholds

        min_len = min(len(fpr), len(tpr), len(thresholds))
        points = []
        for i in range(min_len):
            youden = tpr[i] - fpr[i]
            points.append({
                "threshold": round(float(thresholds[i]), 3),
                "sensitivity": round(float(tpr[i]), 4),
                "one_minus_specificity": round(float(fpr[i]), 4),
                "youden_index": round(float(youden), 4),
            })
        return points


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "alto"
    elif prob >= 0.4:
        return "medio"
    return "bajo"


# Singleton
manager = ModelManager()
