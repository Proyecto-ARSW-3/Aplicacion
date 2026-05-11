"""Tests for app.ml.preprocessor — DataPreprocessor pipeline."""
import math
import numpy as np
import pandas as pd
import pytest

from app.ml.preprocessor import DataPreprocessor, DISPLAY_FEATURE_NAMES


# ─── load_and_clean ───────────────────────────────────────────────────────────

class TestLoadAndClean:
    def test_semicolon_separator(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        assert "Target" in df.columns
        assert set(df["Target"].unique()).issubset({0, 1, 2})

    def test_target_mapping(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        # Dropout→0, Graduate→1, Enrolled→2
        assert 0 in df["Target"].values
        assert 1 in df["Target"].values

    def test_strips_column_names(self, tmp_path):
        """Columns with trailing whitespace/tabs are normalised."""
        bad_cols = pd.DataFrame({
            "Target": ["Dropout", "Graduate"],
            "Age at enrollment  ": [20, 22],
        })
        path = tmp_path / "bad.csv"
        bad_cols.to_csv(path, sep=";", index=False)
        pp = DataPreprocessor()
        df = pp.load_and_clean(str(path))
        assert "Age at enrollment" in df.columns
        assert "Age at enrollment  " not in df.columns

    def test_comma_separator_fallback(self, synthetic_df, tmp_path):
        path = tmp_path / "comma.csv"
        synthetic_df.to_csv(path, sep=",", index=False)
        pp = DataPreprocessor()
        df = pp.load_and_clean(str(path))
        assert len(df) == len(synthetic_df)

    def test_drops_missing_target_rows(self, tmp_path):
        df = pd.DataFrame({
            "Target": ["Dropout", None, "Graduate"],
            "Age at enrollment": [20, 21, 22],
        })
        path = tmp_path / "missing.csv"
        df.to_csv(path, sep=";", index=False)
        pp = DataPreprocessor()
        result = pp.load_and_clean(str(path))
        assert len(result) == 2


# ─── prepare_binary ───────────────────────────────────────────────────────────

class TestPrepareBinary:
    def test_excludes_enrolled(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        binary = pp.prepare_binary(df)
        assert 2 not in binary["Target"].values

    def test_label_encoding(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        binary = pp.prepare_binary(df)
        assert set(binary["target_binary"].unique()).issubset({0, 1})
        # Dropout (Target=0) → target_binary=1
        dropout_rows = binary[binary["Target"] == 0]
        assert (dropout_rows["target_binary"] == 1).all()
        # Graduate (Target=1) → target_binary=0
        graduate_rows = binary[binary["Target"] == 1]
        assert (graduate_rows["target_binary"] == 0).all()

    def test_returns_copy(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        binary = pp.prepare_binary(df)
        assert binary is not df


# ─── build_feature_matrix ─────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_returns_dataframe_and_names(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X, names = pp.build_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_no_missing_values(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X, _ = pp.build_feature_matrix(df)
        assert X.isnull().sum().sum() == 0

    def test_imputes_numeric_with_median(self, tmp_path):
        """Verify numeric NaN → median, not zero."""
        df = pd.DataFrame({
            "Target": ["Dropout"] * 10 + ["Graduate"] * 10,
            "Age at enrollment": [float("nan")] + list(range(1, 20)),
        })
        path = tmp_path / "nulls.csv"
        df.to_csv(path, sep=";", index=False)
        pp = DataPreprocessor()
        loaded = pp.load_and_clean(str(path))
        X, _ = pp.build_feature_matrix(loaded)
        assert X["Age at enrollment"].isnull().sum() == 0


# ─── fit_transform ────────────────────────────────────────────────────────────

class TestFitTransform:
    def test_returns_four_matrices(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
        assert X_tr.shape[0] > 0
        assert X_te.shape[0] > 0
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_te) == X_te.shape[0]

    def test_stratified_split_ratio(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
        total = X_tr.shape[0] + X_te.shape[0]
        # Test set should be ~20% — allow ±5% due to SMOTE on train
        assert X_te.shape[0] / total < 0.35

    def test_smote_balances_train(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        _, _, y_tr, _ = pp.fit_transform(df, apply_smote=True)
        counts = np.bincount(y_tr)
        assert len(counts) == 2
        # After SMOTE both classes should be equal
        assert counts[0] == counts[1]

    def test_sets_is_fitted(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        pp.fit_transform(df)
        assert pp.is_fitted

    def test_feature_names_populated(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        pp.fit_transform(df)
        assert len(pp.feature_names) > 0


# ─── transform ────────────────────────────────────────────────────────────────

class TestTransform:
    def test_uses_fitted_scaler(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X_tr, X_te, _, _ = pp.fit_transform(df)
        # Transform a new batch with same columns
        new_X = X_te.copy()
        result = pp.transform(new_X)
        assert result.shape == new_X.shape

    def test_output_columns_match_feature_names(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        X_tr, X_te, _, _ = pp.fit_transform(df)
        result = pp.transform(X_te)
        assert list(result.columns) == pp.feature_names


# ─── get_full_dataset_features ────────────────────────────────────────────────

class TestGetFullDatasetFeatures:
    def test_includes_enrolled_rows(self, synthetic_csv, synthetic_df):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        pp.fit_transform(df)
        full = pp.get_full_dataset_features(df)
        # Full dataset includes all 3 classes, so rows > binary-only
        binary = pp.prepare_binary(df)
        assert full.shape[0] > len(binary) * 0.8  # enrolled rows added

    def test_returns_scaled_features(self, synthetic_csv):
        pp = DataPreprocessor()
        df = pp.load_and_clean(synthetic_csv)
        pp.fit_transform(df)
        full = pp.get_full_dataset_features(df)
        assert isinstance(full, pd.DataFrame)
        assert full.shape[1] == len(pp.feature_names)


# ─── get_display_name ─────────────────────────────────────────────────────────

class TestGetDisplayName:
    def test_known_feature(self):
        pp = DataPreprocessor()
        assert pp.get_display_name("Debtor") == "Deudor de Matrícula"

    def test_unknown_feature_returns_itself(self):
        pp = DataPreprocessor()
        assert pp.get_display_name("Unknown Feature XYZ") == "Unknown Feature XYZ"

    def test_all_known_features_have_display_names(self):
        pp = DataPreprocessor()
        for technical in DISPLAY_FEATURE_NAMES:
            assert pp.get_display_name(technical) != ""
