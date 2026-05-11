"""Tests for app.ml.model_manager — ModelManager orchestration."""
import math
import pytest

from app.ml.model_manager import ModelManager, _risk_label


# ─── _risk_label helper ───────────────────────────────────────────────────────

class TestRiskLabel:
    def test_alto_at_high_prob(self):
        assert _risk_label(0.8) == "alto"
        assert _risk_label(0.7) == "alto"

    def test_medio_at_mid_prob(self):
        assert _risk_label(0.5) == "medio"
        assert _risk_label(0.4) == "medio"

    def test_bajo_at_low_prob(self):
        assert _risk_label(0.0) == "bajo"
        assert _risk_label(0.39) == "bajo"

    def test_boundary_070_is_alto(self):
        assert _risk_label(0.70) == "alto"

    def test_boundary_040_is_medio(self):
        assert _risk_label(0.40) == "medio"


# ─── ModelManager.train ───────────────────────────────────────────────────────

class TestModelManagerTrain:
    def test_is_trained_after_train(self, trained_manager):
        assert trained_manager.is_trained

    def test_returns_correct_keys(self, synthetic_csv):
        mgr = ModelManager()
        result = mgr.train(synthetic_csv)
        assert "models_trained" in result
        assert "total_samples" in result
        assert "dropout_samples" in result

    def test_five_models_trained(self, trained_manager):
        assert len(trained_manager.models) == 5

    def test_best_model_selected(self, trained_manager):
        assert trained_manager.best_model is not None

    def test_threshold_set_from_best_model(self, trained_manager):
        assert 0.0 < trained_manager.threshold < 1.0

    def test_df_all_proba_populated(self, trained_manager):
        assert trained_manager.df_all_proba is not None
        assert "dropout_proba" in trained_manager.df_all_proba.columns
        assert "actual_class" in trained_manager.df_all_proba.columns

    def test_survival_results_present(self, trained_manager):
        assert trained_manager.survival_results is not None

    def test_total_samples_matches_df(self, trained_manager):
        assert trained_manager.total_samples == len(trained_manager.df_raw)


# ─── ModelManager.get_students_at_risk ────────────────────────────────────────

class TestGetStudentsAtRisk:
    def test_returns_dataframe(self, trained_manager):
        df = trained_manager.get_students_at_risk()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_sorted_descending_by_proba(self, trained_manager):
        df = trained_manager.get_students_at_risk()
        probas = df["dropout_proba"].tolist()
        assert probas == sorted(probas, reverse=True)

    def test_is_at_risk_column_uses_threshold(self, trained_manager):
        tau = 0.5
        df = trained_manager.get_students_at_risk(tau)
        assert "is_at_risk" in df.columns
        assert (df[df["dropout_proba"] >= tau]["is_at_risk"] == True).all()

    def test_risk_level_column_present(self, trained_manager):
        df = trained_manager.get_students_at_risk()
        assert "risk_level" in df.columns
        assert set(df["risk_level"].unique()).issubset({"alto", "medio", "bajo"})

    def test_empty_when_no_predictions(self):
        mgr = ModelManager()
        df = mgr.get_students_at_risk()
        assert df.empty


# ─── ModelManager.update_threshold ────────────────────────────────────────────

class TestUpdateThreshold:
    def test_returns_expected_keys(self, trained_manager):
        old_tau = trained_manager.threshold
        result = trained_manager.update_threshold(0.5)
        trained_manager.threshold = old_tau  # restore
        for key in ("threshold", "at_risk_count", "sensitivity", "specificity", "precision"):
            assert key in result

    def test_at_risk_count_decreases_with_higher_threshold(self, trained_manager):
        old_tau = trained_manager.threshold
        low = trained_manager.update_threshold(0.2)
        high = trained_manager.update_threshold(0.8)
        trained_manager.threshold = old_tau
        assert low["at_risk_count"] >= high["at_risk_count"]

    def test_metrics_in_valid_range(self, trained_manager):
        old_tau = trained_manager.threshold
        result = trained_manager.update_threshold(0.5)
        trained_manager.threshold = old_tau
        assert 0.0 <= result["sensitivity"] <= 1.0
        assert 0.0 <= result["specificity"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0


# ─── ModelManager.predict_single ─────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_expected_keys(self, trained_manager):
        features = {
            "Age at enrollment": 22,
            "Scholarship holder": 0,
            "Debtor": 1,
            "Curricular units 1st sem (approved)": 3,
            "Curricular units 1st sem (grade)": 8.0,
        }
        result = trained_manager.predict_single(features)
        for key in ("dropout_probability", "risk_level", "prediction", "top_risk_factors"):
            assert key in result

    def test_probability_in_range(self, trained_manager):
        result = trained_manager.predict_single({})
        assert 0.0 <= result["dropout_probability"] <= 1.0

    def test_risk_level_valid(self, trained_manager):
        result = trained_manager.predict_single({})
        assert result["risk_level"] in ("alto", "medio", "bajo")

    def test_prediction_label_valid(self, trained_manager):
        result = trained_manager.predict_single({})
        assert result["prediction"] in ("Desertor", "No Desertor")

    def test_untrained_returns_error(self):
        mgr = ModelManager()
        result = mgr.predict_single({})
        assert "error" in result


# ─── ModelManager.get_roc_analysis ────────────────────────────────────────────

class TestGetRocAnalysis:
    def test_returns_list(self, trained_manager):
        points = trained_manager.get_roc_analysis()
        assert isinstance(points, list)

    def test_points_have_expected_keys(self, trained_manager):
        points = trained_manager.get_roc_analysis()
        if points:
            for key in ("threshold", "sensitivity", "one_minus_specificity", "youden_index"):
                assert key in points[0]

    def test_no_infinite_thresholds(self, trained_manager):
        points = trained_manager.get_roc_analysis()
        for pt in points:
            assert math.isfinite(pt["threshold"])

    def test_filter_by_model_name(self, trained_manager):
        name = trained_manager.best_model.name
        points = trained_manager.get_roc_analysis(model_name=name)
        assert isinstance(points, list)

    def test_unknown_model_name_uses_best(self, trained_manager):
        points = trained_manager.get_roc_analysis(model_name="NonExistentModel")
        # Falls back to best_model
        assert isinstance(points, list)

    def test_empty_when_no_roc_data(self):
        mgr = ModelManager()
        assert mgr.get_roc_analysis() == []
