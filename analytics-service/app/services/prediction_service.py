from app.ml.model_manager import manager, _risk_label


def get_students_list(threshold: float | None, page: int, page_size: int, risk_filter: str | None):
    df = manager.get_students_at_risk(threshold)
    if df.empty:
        return {"students": [], "total": 0, "page": page, "page_size": page_size}

    if risk_filter and risk_filter != "all":
        df = df[df["risk_level"] == risk_filter]

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    col_gender = "Gender"
    col_scholarship = "Scholarship holder"
    col_debtor = "Debtor"
    col_tuition = "Tuition fees up to date"
    col_age = "Age at enrollment"
    col_sem1_approved = "Curricular units 1st sem (approved)"
    col_sem1_grade = "Curricular units 1st sem (grade)"
    col_sem2_approved = "Curricular units 2nd sem (approved)"
    col_sem2_grade = "Curricular units 2nd sem (grade)"

    students = []
    for i, (idx, row) in enumerate(page_df.iterrows()):
        students.append({
            "id": int(idx),
            "age_at_enrollment": int(row.get(col_age, 0)),
            "gender": "Hombre" if row.get(col_gender, 1) == 1 else "Mujer",
            "scholarship_holder": bool(row.get(col_scholarship, 0)),
            "debtor": bool(row.get(col_debtor, 0)),
            "tuition_fees_up_to_date": bool(row.get(col_tuition, 1)),
            "sem1_approved": int(row.get(col_sem1_approved, 0)),
            "sem1_grade": round(float(row.get(col_sem1_grade, 0)), 2),
            "sem2_approved": int(row.get(col_sem2_approved, 0)),
            "sem2_grade": round(float(row.get(col_sem2_grade, 0)), 2),
            "dropout_probability": round(float(row["dropout_proba"]), 4),
            "risk_level": str(row["risk_level"]),
            "predicted_class": "Desertor" if row["dropout_proba"] >= (threshold or manager.threshold) else "No Desertor",
            "actual_class": str(row.get("actual_class", "")),
        })

    return {"students": students, "total": total, "page": page, "page_size": page_size}


def get_model_comparison() -> list[dict]:
    return [
        {
            "name": m.name,
            "accuracy": round(m.accuracy * 100, 1),
            "precision": round(m.precision * 100, 1),
            "recall": round(m.recall * 100, 1),
            "f1_score": round(m.f1 * 100, 1),
            "auc_roc": round(m.auc_roc * 100, 1),
            "training_time_ms": m.training_time_ms,
        }
        for m in manager.models
    ]


def get_feature_importance(model_name: str | None) -> list[dict]:
    model = manager.best_model
    if model_name:
        found = next((m for m in manager.models if m.name == model_name), None)
        if found:
            model = found

    if not model:
        return []

    fi = model.feature_importances
    total = sum(fi.values()) or 1.0
    result = sorted(
        [
            {
                "feature": manager.preprocessor.get_display_name(k),
                "importance": round(v / total, 4),
                "raw_importance": round(v, 6),
            }
            for k, v in fi.items()
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return result[:15]  # Top 15
