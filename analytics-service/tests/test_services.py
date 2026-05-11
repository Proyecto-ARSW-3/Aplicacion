"""Tests for analytics_service and prediction_service."""
import math
import pytest


# ─── analytics_service helpers ───────────────────────────────────────────────

class TestSafeHelper:
    """Test the private _safe() helper in analytics_service."""
    def test_normal_float(self):
        from app.services.analytics_service import _safe
        assert _safe(3.14) == pytest.approx(3.14)

    def test_inf_returns_default(self):
        from app.services.analytics_service import _safe
        assert _safe(float("inf")) == 0.0

    def test_nan_returns_default(self):
        from app.services.analytics_service import _safe
        assert _safe(float("nan")) == 0.0

    def test_none_returns_default(self):
        from app.services.analytics_service import _safe
        assert _safe(None) == 0.0

    def test_string_returns_default(self):
        from app.services.analytics_service import _safe
        assert _safe("oops") == 0.0

    def test_custom_default(self):
        from app.services.analytics_service import _safe
        assert _safe(float("inf"), default=99.0) == 99.0


# ─── get_overview ─────────────────────────────────────────────────────────────

class TestGetOverview:
    def test_returns_empty_when_untrained(self):
        from app.services.analytics_service import get_overview
        from app.ml.model_manager import ModelManager
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = ModelManager()  # untrained
        result = get_overview()
        svc.manager = original
        assert result == {}

    def test_returns_required_keys(self, trained_manager):
        from app.services.analytics_service import get_overview
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_overview()
        svc.manager = original
        for key in ("total_students", "dropout_count", "graduate_count",
                    "dropout_rate", "at_risk_count", "best_model", "best_model_accuracy"):
            assert key in result

    def test_dropout_rate_is_fraction(self, trained_manager):
        from app.services.analytics_service import get_overview
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_overview()
        svc.manager = original
        assert 0.0 <= result["dropout_rate"] <= 1.0

    def test_counts_are_consistent(self, trained_manager):
        from app.services.analytics_service import get_overview
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_overview()
        svc.manager = original
        total = result["dropout_count"] + result["graduate_count"] + result["enrolled_count"]
        assert total == result["total_students"]


# ─── get_survival_data ────────────────────────────────────────────────────────

class TestGetSurvivalData:
    def test_returns_empty_when_no_results(self):
        from app.services.analytics_service import get_survival_data
        from app.ml.model_manager import ModelManager
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = ModelManager()
        result = get_survival_data()
        svc.manager = original
        assert result == {}

    def test_returns_required_keys(self, trained_manager):
        from app.services.analytics_service import get_survival_data
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_survival_data()
        svc.manager = original
        for key in ("kaplan_meier_points", "at_risk_by_semester", "median_survival", "cox_summary"):
            assert key in result

    def test_no_inf_nan_in_cox_summary(self, trained_manager):
        from app.services.analytics_service import get_survival_data
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_survival_data()
        svc.manager = original
        for row in result.get("cox_summary", []):
            for val in row.values():
                if isinstance(val, float):
                    assert math.isfinite(val)

    def test_median_survival_is_finite(self, trained_manager):
        from app.services.analytics_service import get_survival_data
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_survival_data()
        svc.manager = original
        assert math.isfinite(result["median_survival"])


# ─── get_risk_distribution ────────────────────────────────────────────────────

class TestGetRiskDistribution:
    def test_returns_empty_when_no_proba(self):
        from app.services.analytics_service import get_risk_distribution
        from app.ml.model_manager import ModelManager
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = ModelManager()
        result = get_risk_distribution()
        svc.manager = original
        assert result == []

    def test_returns_ten_buckets(self, trained_manager):
        from app.services.analytics_service import get_risk_distribution
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_risk_distribution()
        svc.manager = original
        assert len(result) == 10

    def test_bucket_structure(self, trained_manager):
        from app.services.analytics_service import get_risk_distribution
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_risk_distribution()
        svc.manager = original
        for bucket in result:
            assert "range" in bucket
            assert "count" in bucket
            assert bucket["count"] >= 0

    def test_counts_sum_to_total(self, trained_manager):
        from app.services.analytics_service import get_risk_distribution
        import app.services.analytics_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_risk_distribution()
        total_in_buckets = sum(b["count"] for b in result)
        svc.manager = original
        assert total_in_buckets == trained_manager.total_samples


# ─── get_students_list ────────────────────────────────────────────────────────

class TestGetStudentsList:
    def test_returns_empty_when_untrained(self):
        from app.services.prediction_service import get_students_list
        from app.ml.model_manager import ModelManager
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = ModelManager()
        result = get_students_list(None, 1, 20, None)
        svc.manager = original
        assert result["total"] == 0
        assert result["students"] == []

    def test_pagination(self, trained_manager):
        from app.services.prediction_service import get_students_list
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_students_list(None, 1, 5, None)
        svc.manager = original
        assert len(result["students"]) <= 5
        assert result["page"] == 1
        assert result["page_size"] == 5

    def test_risk_filter_alto(self, trained_manager):
        from app.services.prediction_service import get_students_list
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_students_list(None, 1, 100, "alto")
        svc.manager = original
        for s in result["students"]:
            assert s["risk_level"] == "alto"

    def test_student_has_required_fields(self, trained_manager):
        from app.services.prediction_service import get_students_list
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_students_list(None, 1, 1, None)
        svc.manager = original
        if result["students"]:
            s = result["students"][0]
            for field in ("id", "dropout_probability", "risk_level",
                          "predicted_class", "actual_class"):
                assert field in s


# ─── get_model_comparison ─────────────────────────────────────────────────────

class TestGetModelComparison:
    def test_returns_five_entries(self, trained_manager):
        from app.services.prediction_service import get_model_comparison
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_model_comparison()
        svc.manager = original
        assert len(result) == 5

    def test_each_entry_has_metrics(self, trained_manager):
        from app.services.prediction_service import get_model_comparison
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_model_comparison()
        svc.manager = original
        for entry in result:
            for key in ("name", "accuracy", "precision", "recall", "f1_score", "auc_roc"):
                assert key in entry


# ─── get_feature_importance ───────────────────────────────────────────────────

class TestGetFeatureImportance:
    def test_returns_at_most_fifteen(self, trained_manager):
        from app.services.prediction_service import get_feature_importance
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_feature_importance(None)
        svc.manager = original
        assert len(result) <= 15

    def test_sorted_descending(self, trained_manager):
        from app.services.prediction_service import get_feature_importance
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        result = get_feature_importance(None)
        svc.manager = original
        importances = [r["importance"] for r in result]
        assert importances == sorted(importances, reverse=True)

    def test_filter_by_model_name(self, trained_manager):
        from app.services.prediction_service import get_feature_importance
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = trained_manager
        name = trained_manager.models[0].name
        result = get_feature_importance(name)
        svc.manager = original
        assert len(result) <= 15

    def test_no_model_returns_empty(self):
        from app.services.prediction_service import get_feature_importance
        from app.ml.model_manager import ModelManager
        import app.services.prediction_service as svc
        original = svc.manager
        svc.manager = ModelManager()
        result = get_feature_importance(None)
        svc.manager = original
        assert result == []
