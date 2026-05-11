"""
Tests for all API route handlers using FastAPI TestClient.

These tests use the `client` fixture from conftest, which trains the global
manager singleton before the TestClient starts. Tests that need an untrained
state temporarily toggle `manager.is_trained`.
"""
import io
import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _set_untrained(monkeypatch):
    """Temporarily mark the global manager as untrained."""
    from app.ml import model_manager as mm
    monkeypatch.setattr(mm.manager, "is_trained", False)


# ─── /api/health ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok_when_trained(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["trained"] is True

    def test_health_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["trained"] is False


# ─── /api/analytics/* ─────────────────────────────────────────────────────────

class TestAnalyticsRoutes:
    def test_overview_returns_200(self, client):
        r = client.get("/api/analytics/overview")
        assert r.status_code == 200
        data = r.json()
        assert "total_students" in data
        assert "best_model" in data

    def test_overview_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/analytics/overview")
        assert r.status_code == 503

    def test_survival_returns_200(self, client):
        r = client.get("/api/analytics/survival")
        assert r.status_code == 200
        data = r.json()
        assert "kaplan_meier_points" in data
        assert "cox_summary" in data

    def test_survival_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/analytics/survival")
        assert r.status_code == 503

    def test_risk_distribution_returns_200(self, client):
        r = client.get("/api/analytics/risk-distribution")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 10

    def test_risk_distribution_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/analytics/risk-distribution")
        assert r.status_code == 503

    def test_roc_returns_200(self, client):
        r = client.get("/api/analytics/roc")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_roc_with_model_name(self, client):
        from app.ml import manager
        name = manager.best_model.name
        r = client.get(f"/api/analytics/roc?model_name={name}")
        assert r.status_code == 200

    def test_roc_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/analytics/roc")
        assert r.status_code == 503


# ─── /api/models/* ────────────────────────────────────────────────────────────

class TestModelsRoutes:
    def test_comparison_returns_five_models(self, client):
        r = client.get("/api/models/comparison")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5

    def test_comparison_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/models/comparison")
        assert r.status_code == 503

    def test_feature_importance_returns_list(self, client):
        r = client.get("/api/models/feature-importance")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) <= 15

    def test_feature_importance_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/models/feature-importance")
        assert r.status_code == 503

    def test_list_returns_model_info(self, client):
        r = client.get("/api/models/list")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        for entry in data:
            assert "name" in entry
            assert "auc_roc" in entry
            assert "is_best" in entry

    def test_list_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/models/list")
        assert r.status_code == 503


# ─── /api/predictions/* ───────────────────────────────────────────────────────

class TestPredictionsRoutes:
    def test_students_returns_paginated(self, client):
        r = client.get("/api/predictions/students?page=1&page_size=10")
        assert r.status_code == 200
        data = r.json()
        assert "students" in data
        assert "total" in data
        assert len(data["students"]) <= 10

    def test_students_filter_by_risk_level(self, client):
        r = client.get("/api/predictions/students?risk_filter=alto&page_size=50")
        assert r.status_code == 200
        data = r.json()
        for s in data["students"]:
            assert s["risk_level"] == "alto"

    def test_students_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/predictions/students")
        assert r.status_code == 503

    def test_threshold_get(self, client):
        r = client.get("/api/predictions/threshold")
        assert r.status_code == 200
        data = r.json()
        assert "threshold" in data
        assert "optimal_threshold" in data

    def test_threshold_post_updates(self, client):
        r = client.post("/api/predictions/threshold", json={"threshold": 0.4})
        assert r.status_code == 200
        data = r.json()
        assert data["threshold"] == pytest.approx(0.4)
        assert "sensitivity" in data
        # Restore default threshold
        client.post("/api/predictions/threshold", json={"threshold": 0.5})

    def test_threshold_post_503_when_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.post("/api/predictions/threshold", json={"threshold": 0.5})
        assert r.status_code == 503

    def test_predict_returns_probability(self, client):
        payload = {
            "age_at_enrollment": 22,
            "gender": 1,
            "scholarship_holder": 0,
            "debtor": 1,
            "tuition_fees_up_to_date": 0,
            "displaced": 0,
            "educational_special_needs": 0,
            "marital_status": 1,
            "application_mode": 1,
            "course": 1,
            "daytime_evening_attendance": 1,
            "previous_qualification_grade": 130.0,
            "admission_grade": 120.0,
            "curricular_units_1st_sem_approved": 3,
            "curricular_units_1st_sem_grade": 10.0,
            "curricular_units_2nd_sem_approved": 2,
            "curricular_units_2nd_sem_grade": 9.0,
            "unemployment_rate": 10.5,
            "inflation_rate": 1.2,
            "gdp": 1.5,
        }
        r = client.post("/api/predictions/predict", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "dropout_probability" in data
        assert 0.0 <= data["dropout_probability"] <= 1.0
        assert data["risk_level"] in ("alto", "medio", "bajo")

    def test_predict_503_when_untrained(self, client, monkeypatch):
        payload = {
            "age_at_enrollment": 22, "gender": 1, "scholarship_holder": 0,
            "debtor": 0, "tuition_fees_up_to_date": 1, "displaced": 0,
            "educational_special_needs": 0, "marital_status": 1,
            "application_mode": 1, "course": 1, "daytime_evening_attendance": 1,
            "previous_qualification_grade": 130.0, "admission_grade": 120.0,
            "curricular_units_1st_sem_approved": 5,
            "curricular_units_1st_sem_grade": 12.0,
            "curricular_units_2nd_sem_approved": 5,
            "curricular_units_2nd_sem_grade": 12.0,
            "unemployment_rate": 10.0, "inflation_rate": 1.0, "gdp": 1.0,
        }
        _set_untrained(monkeypatch)
        r = client.post("/api/predictions/predict", json=payload)
        assert r.status_code == 503


# ─── /api/data/* ──────────────────────────────────────────────────────────────

class TestDataRoutes:
    def test_status_returns_200(self, client):
        r = client.get("/api/data/status")
        assert r.status_code == 200
        assert "status" in r.json()

    def test_info_trained(self, client):
        r = client.get("/api/data/info")
        assert r.status_code == 200
        data = r.json()
        assert data["trained"] is True
        assert "total_rows" in data

    def test_info_untrained(self, client, monkeypatch):
        _set_untrained(monkeypatch)
        r = client.get("/api/data/info")
        assert r.status_code == 200
        assert r.json()["trained"] is False

    def test_upload_rejects_non_csv(self, client):
        fake_file = io.BytesIO(b"not a csv")
        r = client.post(
            "/api/data/upload",
            files={"file": ("data.json", fake_file, "application/json")},
        )
        assert r.status_code == 422

    def test_upload_accepts_csv(self, client, synthetic_csv):
        with open(synthetic_csv, "rb") as f:
            r = client.post(
                "/api/data/upload",
                files={"file": ("dataset.csv", f, "text/csv")},
            )
        assert r.status_code == 200
        assert "uploaded" in r.json()["message"].lower()

    def test_train_404_when_no_dataset(self, client, monkeypatch, tmp_path):
        """Simulate missing dataset by overriding settings path."""
        from app.core.config import settings
        monkeypatch.setattr(settings, "dataset_path", str(tmp_path / "nonexistent.csv"))
        r = client.post("/api/data/train")
        assert r.status_code == 404
