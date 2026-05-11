"""
Shared pytest fixtures for the analytics-service test suite.

Session-scoped fixtures are used for expensive operations (CSV creation,
model training) so they run only once per test session.
"""
import pytest
import numpy as np
import pandas as pd


# ─── Synthetic dataset ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_df():
    """300-row DataFrame matching the expected dataset schema."""
    rng = np.random.default_rng(42)
    n = 300

    df = pd.DataFrame({
        # Categorical (already-encoded as integers in the UCI dataset)
        "Marital status": rng.integers(1, 7, n),
        "Application mode": rng.integers(1, 18, n),
        "Application order": rng.integers(0, 10, n),
        "Course": rng.integers(1, 20, n),
        "Daytime/evening attendance": rng.integers(0, 2, n),
        "Previous qualification": rng.integers(1, 40, n),
        "Nacionality": rng.integers(1, 100, n),
        "Mother's qualification": rng.integers(1, 40, n),
        "Father's qualification": rng.integers(1, 40, n),
        "Mother's occupation": rng.integers(0, 10, n),
        "Father's occupation": rng.integers(0, 10, n),
        "Displaced": rng.integers(0, 2, n),
        "Educational special needs": rng.integers(0, 2, n),
        "Debtor": rng.integers(0, 2, n),
        "Tuition fees up to date": rng.integers(0, 2, n),
        "Gender": rng.integers(0, 2, n),
        "Scholarship holder": rng.integers(0, 2, n),
        "International": rng.integers(0, 2, n),
        # Numeric
        "Previous qualification (grade)": rng.uniform(95, 190, n),
        "Admission grade": rng.uniform(95, 190, n),
        "Age at enrollment": rng.integers(17, 55, n),
        "Curricular units 1st sem (credited)": rng.integers(0, 20, n),
        "Curricular units 1st sem (enrolled)": rng.integers(0, 10, n),
        "Curricular units 1st sem (evaluations)": rng.integers(0, 20, n),
        "Curricular units 1st sem (approved)": rng.integers(0, 10, n),
        "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
        "Curricular units 1st sem (without evaluations)": rng.integers(0, 5, n),
        "Curricular units 2nd sem (credited)": rng.integers(0, 20, n),
        "Curricular units 2nd sem (enrolled)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (evaluations)": rng.integers(0, 20, n),
        "Curricular units 2nd sem (approved)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (grade)": rng.uniform(0, 20, n),
        "Curricular units 2nd sem (without evaluations)": rng.integers(0, 5, n),
        "Unemployment rate": rng.uniform(7, 17, n),
        "Inflation rate": rng.uniform(0, 5, n),
        "GDP": rng.uniform(-4, 4, n),
        # Target as strings (as in the real CSV)
        "Target": rng.choice(["Dropout", "Graduate", "Enrolled"], n, p=[0.35, 0.50, 0.15]),
    })
    return df


@pytest.fixture(scope="session")
def synthetic_csv(synthetic_df, tmp_path_factory):
    """Write the synthetic DataFrame to a temp CSV (semicolon-separated, as UCI format)."""
    tmp_dir = tmp_path_factory.mktemp("data")
    path = tmp_dir / "dataset.csv"
    synthetic_df.to_csv(path, sep=";", index=False)
    return str(path)


# ─── ModelManager (isolated instance for unit tests) ─────────────────────────

@pytest.fixture(scope="session")
def trained_manager(synthetic_csv):
    """
    A freshly constructed ModelManager trained on the synthetic dataset.
    This is NOT the global singleton — it is isolated for unit tests.
    """
    from app.ml.model_manager import ModelManager
    mgr = ModelManager()
    mgr.train(synthetic_csv)
    return mgr


# ─── Global singleton (used by API routes) ───────────────────────────────────

@pytest.fixture(scope="session")
def _trained_global_manager(synthetic_csv):
    """
    Train the module-level singleton so that API route handlers
    see an already-trained manager when the TestClient makes requests.
    """
    from app.ml import model_manager as mm_module
    if not mm_module.manager.is_trained:
        mm_module.manager.train(synthetic_csv)
    return mm_module.manager


@pytest.fixture(scope="session")
def client(_trained_global_manager):
    """FastAPI TestClient with a trained global manager."""
    from fastapi.testclient import TestClient
    from app.main import app
    # Disable lifespan (auto-training) during tests to avoid re-training
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
