from app.ml.model_manager import manager


def get_overview() -> dict:
    if not manager.is_trained:
        return {}

    df = manager.df_all_proba
    total = len(df)
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
        "dropout_rate": round(dropout / total, 4) if total > 0 else 0.0,
        "at_risk_count": at_risk,
        "at_risk_rate": round(at_risk / total, 4) if total > 0 else 0.0,
        "best_model": best.name if best else "",
        "best_model_accuracy": round(best.accuracy, 4) if best else 0.0,
    }


def get_survival_data() -> dict:
    if not manager.survival_results:
        return {}

    sr = manager.survival_results
    return {
        "kaplan_meier_points": sr.kaplan_meier_points,
        "at_risk_by_semester": sr.at_risk_by_semester,
        "median_survival": sr.median_survival,
        "cox_summary": sr.cox_summary,
    }


def get_risk_distribution() -> list[dict]:
    if manager.df_all_proba is None:
        return []

    buckets = [
        ("0.0–0.1", 0.0, 0.1),
        ("0.1–0.2", 0.1, 0.2),
        ("0.2–0.3", 0.2, 0.3),
        ("0.3–0.4", 0.3, 0.4),
        ("0.4–0.5", 0.4, 0.5),
        ("0.5–0.6", 0.5, 0.6),
        ("0.6–0.7", 0.6, 0.7),
        ("0.7–0.8", 0.7, 0.8),
        ("0.8–0.9", 0.8, 0.9),
        ("0.9–1.0", 0.9, 1.01),
    ]

    df = manager.df_all_proba
    result = []
    for label, low, high in buckets:
        mask = (df["dropout_proba"] >= low) & (df["dropout_proba"] < high)
        result.append({"range": label, "count": int(mask.sum())})
    return result
