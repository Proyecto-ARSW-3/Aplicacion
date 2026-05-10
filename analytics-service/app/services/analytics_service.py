import math
from app.ml.model_manager import manager


def get_overview() -> dict:
    if not manager.is_trained:
        return {}

    df = manager.df_all_proba
    total = len(df)
    if total == 0:
        return {}

    dropout = int((df["actual_class"] == "Desertor").sum())
    graduate = int((df["actual_class"] == "Graduado").sum())
    enrolled = int((df["actual_class"] == "Activo").sum())
    at_risk = int((df["dropout_proba"] >= manager.threshold).sum())

    best = manager.best_model

    return {
        "total_students": total,
        "dropout_count": dropout,
        "graduate_count": graduate,
        "enrolled_count": enrolled,
        "dropout_rate": round(dropout / total, 4),
        "at_risk_count": at_risk,
        "at_risk_rate": round(at_risk / total, 4),
        "best_model": best.name if best else "",
        "best_model_accuracy": round(_safe(best.accuracy), 4) if best else 0.0,
    }


def get_survival_data() -> dict:
    if not manager.survival_results:
        return {}

    sr = manager.survival_results

    # Extra safety: ensure median_survival is JSON serialisable
    median = _safe(sr.median_survival, 0.0)

    # Sanitise kaplan_meier_points
    km_points = [
        {
            "semester": int(p["semester"]),
            "survival_probability": _safe(p["survival_probability"], 0.0),
        }
        for p in sr.kaplan_meier_points
    ]

    # Sanitise cox_summary
    cox = []
    for row in sr.cox_summary:
        if "error" in row:
            continue
        cox.append({
            "covariate": str(row.get("covariate", "")),
            "exp_coef": _safe(row.get("exp_coef", 1.0), 1.0),
            "p_value": _safe(row.get("p_value", 1.0), 1.0),
            "hr_lower": _safe(row.get("hr_lower", 0.0), 0.0),
            "hr_upper": _safe(row.get("hr_upper", 0.0), 0.0),
        })

    return {
        "kaplan_meier_points": km_points,
        "at_risk_by_semester": sr.at_risk_by_semester,
        "median_survival": median,
        "cox_summary": cox,
    }


def get_risk_distribution() -> list[dict]:
    if manager.df_all_proba is None:
        return []

    # ASCII-safe range labels (avoid en-dash encoding issues on some systems)
    buckets = [
        ("0.0-0.1", 0.0, 0.1),
        ("0.1-0.2", 0.1, 0.2),
        ("0.2-0.3", 0.2, 0.3),
        ("0.3-0.4", 0.3, 0.4),
        ("0.4-0.5", 0.4, 0.5),
        ("0.5-0.6", 0.5, 0.6),
        ("0.6-0.7", 0.6, 0.7),
        ("0.7-0.8", 0.7, 0.8),
        ("0.8-0.9", 0.8, 0.9),
        ("0.9-1.0", 0.9, 1.01),
    ]

    df = manager.df_all_proba
    result = []
    for label, low, high in buckets:
        mask = (df["dropout_proba"] >= low) & (df["dropout_proba"] < high)
        result.append({"range": label, "count": int(mask.sum())})
    return result


def _safe(v, default: float = 0.0) -> float:
    """Return a JSON-serialisable float (never inf/nan)."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default
