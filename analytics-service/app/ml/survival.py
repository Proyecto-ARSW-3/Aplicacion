import math
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from dataclasses import dataclass, field


def _safe_float(v, default: float = 0.0) -> float:
    """Return a JSON-safe float (never inf or nan)."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


@dataclass
class SurvivalResults:
    kaplan_meier_points: list[dict]
    cox_summary: list[dict]
    median_survival: float
    at_risk_by_semester: list[dict]


def _estimate_duration_and_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate 'time to dropout' from cross-sectional data.
    Students who dropped out are assigned duration based on their academic performance.
    Graduates get the maximum duration (10 semesters = typical graduation).
    Enrolled students are censored at semester 6 (ongoing).
    """
    result = df.copy()

    # Base duration heuristic: use number of approved units as a proxy for time
    approved_1 = result.get("Curricular units 1st sem (approved)", pd.Series(0, index=result.index))
    approved_2 = result.get("Curricular units 2nd sem (approved)", pd.Series(0, index=result.index))

    if isinstance(approved_1, str):
        approved_1 = 0
    if isinstance(approved_2, str):
        approved_2 = 0

    total_approved = approved_1.fillna(0) + approved_2.fillna(0)

    # Map target
    target_map = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
    if "Target" in result.columns:
        target_label = result["Target"].map(target_map)
    else:
        return pd.DataFrame()

    # Assign duration (semesters):
    # - Graduates: between 8-10 semesters (assume 10 for simplicity)
    # - Dropouts: estimate based on performance (1-6 semesters)
    # - Enrolled: censored at semester 4 (still ongoing)
    durations = []
    events = []  # 1 = dropout occurred, 0 = censored

    for idx in result.index:
        label = target_label.loc[idx]
        approved = total_approved.loc[idx]

        if label == "Dropout":
            # Estimate dropout time: fewer approved units = earlier dropout
            if approved <= 0:
                dur = 1
            elif approved <= 4:
                dur = 2
            elif approved <= 8:
                dur = 3
            elif approved <= 12:
                dur = 4
            else:
                dur = 5
            durations.append(dur)
            events.append(1)
        elif label == "Graduate":
            durations.append(10)
            events.append(0)  # Censored (completed successfully)
        else:  # Enrolled
            durations.append(4)
            events.append(0)  # Censored (still studying)

    result["duration"] = durations
    result["event"] = events
    return result


def compute_kaplan_meier(df: pd.DataFrame) -> SurvivalResults:
    """Compute Kaplan-Meier survival curve from dataset."""
    df_surv = _estimate_duration_and_event(df)

    if df_surv.empty:
        return SurvivalResults([], [], 0.0, [])

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df_surv["duration"],
        event_observed=df_surv["event"],
        label="Cohorte Total",
    )

    # Extract survival function values
    sf = kmf.survival_function_
    timeline = sf.index.tolist()
    survival_probs = sf["Cohorte Total"].tolist()

    # Compute at-risk table
    at_risk_table = kmf.event_table

    kaplan_points = []
    at_risk_by_sem = []

    for i, t in enumerate(timeline):
        prob = survival_probs[i]
        kaplan_points.append({
            "semester": int(t),
            "survival_probability": round(float(prob), 4),
        })

    for t in at_risk_table.index:
        row = at_risk_table.loc[t]
        at_risk_by_sem.append({
            "semester": int(t),
            "at_risk": int(row.get("at_risk", 0)),
            "events": int(row.get("observed", 0)),
            "censored": int(row.get("censored", 0)),
        })

    # median_survival_time_ is inf when >50% of the cohort is still censored — not JSON serialisable
    median_survival = _safe_float(kmf.median_survival_time_, default=0.0)

    # Cox PH model
    cox_summary = _compute_cox(df_surv)

    return SurvivalResults(
        kaplan_meier_points=kaplan_points,
        cox_summary=cox_summary,
        median_survival=median_survival,
        at_risk_by_semester=at_risk_by_sem,
    )


def _compute_cox(df_surv: pd.DataFrame) -> list[dict]:
    """Fit Cox PH model and return covariate summary."""
    cox_features = [
        "Age at enrollment",
        "Scholarship holder",
        "Debtor",
        "Tuition fees up to date",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
        "GDP",
    ]

    available = [c for c in cox_features if c in df_surv.columns]
    if not available:
        return []

    cox_df = df_surv[available + ["duration", "event"]].copy()
    cox_df = cox_df.fillna(cox_df.median(numeric_only=True))
    cox_df = cox_df.dropna()

    # Avoid multicollinearity: use a subset
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(cox_df, duration_col="duration", event_col="event")
        summary = cph.summary.reset_index()
        result = []
        for _, row in summary.iterrows():
            result.append({
                "covariate": str(row["covariate"]),
                "exp_coef": round(_safe_float(row["exp(coef)"], 1.0), 4),
                "p_value": round(_safe_float(row["p"], 1.0), 4),
                "hr_lower": round(_safe_float(row.get("exp(coef) lower 95%", 0), 0.0), 4),
                "hr_upper": round(_safe_float(row.get("exp(coef) upper 95%", 0), 0.0), 4),
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]
