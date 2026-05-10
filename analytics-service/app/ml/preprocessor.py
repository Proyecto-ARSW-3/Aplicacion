import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path


NUMERIC_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

DISPLAY_FEATURE_NAMES = {
    "Curricular units 1st sem (grade)": "Promedio Académico Sem 1",
    "Curricular units 2nd sem (grade)": "Promedio Académico Sem 2",
    "Curricular units 1st sem (approved)": "Materias Aprobadas Sem 1",
    "Curricular units 2nd sem (approved)": "Materias Aprobadas Sem 2",
    "Scholarship holder": "Estatus de Becario",
    "Debtor": "Deudor de Matrícula",
    "Tuition fees up to date": "Matrícula al Día",
    "Age at enrollment": "Edad de Ingreso",
    "Unemployment rate": "Tasa de Desempleo",
    "GDP": "PIB",
    "Admission grade": "Nota de Admisión",
    "Gender": "Género",
    "Displaced": "Desplazado",
    "Previous qualification (grade)": "Calificación Previa",
    "Inflation rate": "Tasa de Inflación",
}


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.is_fitted = False

    def load_and_clean(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception:
            df = pd.read_csv(path)

        # Normalize column names (remove trailing tabs/spaces)
        df.columns = [c.strip() for c in df.columns]

        # Map target to numeric
        target_map = {"Dropout": 0, "Graduate": 1, "Enrolled": 2}
        df["Target"] = df["Target"].map(target_map)

        # Drop rows with missing target
        df = df.dropna(subset=["Target"])
        df["Target"] = df["Target"].astype(int)

        return df

    def prepare_binary(self, df: pd.DataFrame):
        """Prepare binary classification: Dropout (1) vs Not-Dropout (0)."""
        df_binary = df.copy()
        # Exclude 'Enrolled' students for binary classification (focus on known outcomes)
        df_binary = df_binary[df_binary["Target"] != 2].copy()
        # 1 = Dropout, 0 = Graduate
        df_binary["target_binary"] = (df_binary["Target"] == 0).astype(int)
        return df_binary

    def build_feature_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Build the feature matrix from available columns."""
        available_numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
        available_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

        X = df[available_numeric + available_categorical].copy()

        # Fill missing values
        for col in available_numeric:
            X[col] = X[col].fillna(X[col].median())
        for col in available_categorical:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)

        feature_names = list(X.columns)
        return X, feature_names

    def fit_transform(self, df: pd.DataFrame, apply_smote: bool = True):
        """Full preprocessing pipeline for training."""
        df_binary = self.prepare_binary(df)
        X, feature_names = self.build_feature_matrix(df_binary)
        y = df_binary["target_binary"].values

        self.feature_names = feature_names

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        if apply_smote:
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception:
                pass  # If SMOTE fails, continue without it

        self.is_fitted = True
        return X_train, X_test, y_train, y_test

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.feature_names)

    def get_full_dataset_features(self, df: pd.DataFrame):
        """Get features for the full dataset (all three classes)."""
        X, _ = self.build_feature_matrix(df)
        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
            return pd.DataFrame(X_scaled, columns=self.feature_names)
        return X

    def get_display_name(self, feature: str) -> str:
        return DISPLAY_FEATURE_NAMES.get(feature, feature)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DataPreprocessor":
        return joblib.load(path)
