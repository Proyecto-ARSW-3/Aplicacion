from pydantic import BaseModel
from typing import Optional


class OverviewResponse(BaseModel):
    total_students: int
    dropout_count: int
    graduate_count: int
    enrolled_count: int
    dropout_rate: float
    at_risk_count: int
    at_risk_rate: float
    best_model: str
    best_model_accuracy: float


class ModelMetrics(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_time_ms: int


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class SurvivalPoint(BaseModel):
    semester: int
    survival_probability: float
    at_risk: int
    events: int


class RiskStudent(BaseModel):
    id: int
    age_at_enrollment: int
    gender: str
    scholarship_holder: bool
    debtor: bool
    tuition_fees_up_to_date: bool
    sem1_approved: int
    sem1_grade: float
    sem2_approved: int
    sem2_grade: float
    dropout_probability: float
    risk_level: str
    predicted_class: str
    actual_class: Optional[str] = None


class ThresholdRequest(BaseModel):
    threshold: float


class ThresholdResponse(BaseModel):
    threshold: float
    at_risk_count: int
    sensitivity: float
    specificity: float
    precision: float


class RocPoint(BaseModel):
    threshold: float
    sensitivity: float
    one_minus_specificity: float
    youden_index: float


class PredictionRequest(BaseModel):
    age_at_enrollment: int
    gender: int
    scholarship_holder: int
    debtor: int
    tuition_fees_up_to_date: int
    displaced: int
    educational_special_needs: int
    marital_status: int
    application_mode: int
    course: int
    daytime_evening_attendance: int
    previous_qualification_grade: float
    admission_grade: float
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    unemployment_rate: float
    inflation_rate: float
    gdp: float


class PredictionResponse(BaseModel):
    dropout_probability: float
    risk_level: str
    recommended_threshold: float
    prediction: str
    top_risk_factors: list[FeatureImportance]


class TrainingStatus(BaseModel):
    status: str
    message: str
    models_trained: list[str]
    total_samples: int
    dropout_samples: int
