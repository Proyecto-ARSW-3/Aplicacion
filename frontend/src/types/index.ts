export interface Overview {
  total_students: number
  dropout_count: number
  graduate_count: number
  enrolled_count: number
  dropout_rate: number
  at_risk_count: number
  at_risk_rate: number
  best_model: string
  best_model_accuracy: number
}

export interface ModelMetrics {
  name: string
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  auc_roc: number
  training_time_ms: number
}

export interface FeatureImportance {
  feature: string
  importance: number
  raw_importance: number
}

export interface SurvivalPoint {
  semester: number
  survival_probability: number
}

export interface AtRiskBySemester {
  semester: number
  at_risk: number
  events: number
  censored: number
}

export interface CoxEntry {
  covariate: string
  exp_coef: number
  p_value: number
  hr_lower: number
  hr_upper: number
}

export interface SurvivalData {
  kaplan_meier_points: SurvivalPoint[]
  at_risk_by_semester: AtRiskBySemester[]
  median_survival: number
  cox_summary: CoxEntry[]
}

export interface RiskStudent {
  id: number
  age_at_enrollment: number
  gender: string
  scholarship_holder: boolean
  debtor: boolean
  tuition_fees_up_to_date: boolean
  sem1_approved: number
  sem1_grade: number
  sem2_approved: number
  sem2_grade: number
  dropout_probability: number
  risk_level: 'alto' | 'medio' | 'bajo'
  predicted_class: string
  actual_class: string
}

export interface StudentsResponse {
  students: RiskStudent[]
  total: number
  page: number
  page_size: number
}

export interface ThresholdStats {
  threshold: number
  at_risk_count: number
  sensitivity: number
  specificity: number
  precision: number
}

export interface RiskBucket {
  range: string
  count: number
}

export interface RocPoint {
  threshold: number
  sensitivity: number
  one_minus_specificity: number
  youden_index: number
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'complete' | 'error'
  message: string
  models_trained?: string[]
  total_samples?: number
  dropout_samples?: number
}
