"""Tests for app.ml.survival — Kaplan-Meier and Cox PH analysis."""
import math
import numpy as np
import pandas as pd
import pytest

from app.ml.survival import (
    _safe_float,
    _estimate_duration_and_event,
    compute_kaplan_meier,
    SurvivalResults,
)


# ─── _safe_float ──────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_normal_value(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_integer_input(self):
        assert _safe_float(5) == 5.0

    def test_infinity_returns_default(self):
        assert _safe_float(float("inf")) == 0.0

    def test_neg_infinity_returns_default(self):
        assert _safe_float(float("-inf")) == 0.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_string_returns_default(self):
        assert _safe_float("not a number") == 0.0

    def test_custom_default(self):
        assert _safe_float(float("inf"), default=99.0) == 99.0

    def test_zero(self):
        assert _safe_float(0.0) == 0.0

    def test_negative_value(self):
        assert _safe_float(-1.5) == pytest.approx(-1.5)


# ─── _estimate_duration_and_event ─────────────────────────────────────────────

class TestEstimateDurationAndEvent:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame({
            "Target": [0, 0, 0, 1, 1, 2],
            "Curricular units 1st sem (approved)": [0, 3, 10, 8, 6, 5],
            "Curricular units 2nd sem (approved)": [0, 2, 5, 7, 4, 3],
        })

    def test_dropout_event_flag(self, base_df):
        result = _estimate_duration_and_event(base_df)
        dropout_rows = result[result["Target"] == 0]
        assert (dropout_rows["event"] == 1).all()

    def test_graduate_censored(self, base_df):
        result = _estimate_duration_and_event(base_df)
        grad_rows = result[result["Target"] == 1]
        assert (grad_rows["event"] == 0).all()
        assert (grad_rows["duration"] == 10).all()

    def test_enrolled_censored(self, base_df):
        result = _estimate_duration_and_event(base_df)
        enrolled_rows = result[result["Target"] == 2]
        assert (enrolled_rows["event"] == 0).all()
        assert (enrolled_rows["duration"] == 4).all()

    def test_dropout_duration_based_on_approved(self, base_df):
        result = _estimate_duration_and_event(base_df)
        # Row with 0 approved → duration 1
        zero_approved = result[(result["Target"] == 0) &
                               (result["Curricular units 1st sem (approved)"] == 0) &
                               (result["Curricular units 2nd sem (approved)"] == 0)]
        if len(zero_approved) > 0:
            assert zero_approved.iloc[0]["duration"] == 1

    def test_missing_target_column_returns_empty(self):
        df = pd.DataFrame({"NotTarget": [0, 1]})
        result = _estimate_duration_and_event(df)
        assert result.empty

    def test_adds_duration_and_event_columns(self, base_df):
        result = _estimate_duration_and_event(base_df)
        assert "duration" in result.columns
        assert "event" in result.columns


# ─── compute_kaplan_meier ─────────────────────────────────────────────────────

class TestComputeKaplanMeier:
    def test_returns_survival_results(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        assert isinstance(result, SurvivalResults)

    def test_kaplan_meier_points_not_empty(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        assert len(result.kaplan_meier_points) > 0

    def test_survival_probs_in_range(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        for pt in result.kaplan_meier_points:
            assert 0.0 <= pt["survival_probability"] <= 1.0

    def test_median_survival_is_finite(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        assert math.isfinite(result.median_survival)

    def test_at_risk_by_semester_not_empty(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        assert len(result.at_risk_by_semester) > 0

    def test_empty_dataframe_returns_empty_results(self):
        result = compute_kaplan_meier(pd.DataFrame())
        assert result.kaplan_meier_points == []
        assert result.cox_summary == []

    def test_cox_summary_has_valid_structure(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        for row in result.cox_summary:
            assert "covariate" in row
            assert "exp_coef" in row
            assert math.isfinite(row["exp_coef"])
            assert "p_value" in row

    def test_no_inf_or_nan_in_cox_summary(self, synthetic_csv):
        from app.ml.preprocessor import DataPreprocessor
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        result = compute_kaplan_meier(df)
        for row in result.cox_summary:
            for key, val in row.items():
                if isinstance(val, float):
                    assert math.isfinite(val), f"Non-finite value in Cox summary: {key}={val}"
