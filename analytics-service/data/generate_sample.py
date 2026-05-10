"""
Generates a synthetic dataset matching the structure of the Kaggle
'Higher Education Predictors of Student Retention' dataset.

Run this script to create a sample dataset for testing if you have not
yet downloaded the real one from Kaggle.

Usage:
    python data/generate_sample.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 4424


def generate():
    data = {}

    data["Marital status"] = RNG.integers(1, 7, N)
    data["Application mode"] = RNG.integers(1, 57, N)
    data["Application order"] = RNG.integers(0, 9, N)
    data["Course"] = RNG.integers(33, 9991, N)
    data["Daytime/evening attendance\t"] = RNG.integers(0, 2, N)
    data["Previous qualification"] = RNG.integers(1, 43, N)
    data["Previous qualification (grade)"] = RNG.uniform(95, 190, N)
    data["Nacionality"] = RNG.integers(1, 21, N)
    data["Mother's qualification"] = RNG.integers(1, 44, N)
    data["Father's qualification"] = RNG.integers(1, 44, N)
    data["Mother's occupation"] = RNG.integers(0, 46, N)
    data["Father's occupation"] = RNG.integers(0, 46, N)
    data["Admission grade"] = RNG.uniform(95, 190, N)

    data["Displaced"] = RNG.integers(0, 2, N)
    data["Educational special needs"] = RNG.integers(0, 2, N)
    debtor = RNG.integers(0, 2, N)
    data["Debtor"] = debtor
    tuition = RNG.integers(0, 2, N)
    data["Tuition fees up to date"] = tuition
    data["Gender"] = RNG.integers(0, 2, N)
    scholarship = RNG.integers(0, 2, N)
    data["Scholarship holder"] = scholarship
    data["Age at enrollment"] = RNG.integers(17, 55, N)
    data["International"] = RNG.integers(0, 2, N)

    # Academic performance — correlated with outcome
    grade_1 = RNG.normal(10, 4, N).clip(0, 20)
    grade_2 = RNG.normal(10, 4, N).clip(0, 20)
    approved_1 = RNG.integers(0, 10, N)
    approved_2 = RNG.integers(0, 10, N)

    data["Curricular units 1st sem (credited)"] = RNG.integers(0, 20, N)
    data["Curricular units 1st sem (enrolled)"] = RNG.integers(4, 10, N)
    data["Curricular units 1st sem (evaluations)"] = RNG.integers(4, 15, N)
    data["Curricular units 1st sem (approved)"] = approved_1
    data["Curricular units 1st sem (grade)"] = grade_1
    data["Curricular units 1st sem (without evaluations)"] = RNG.integers(0, 3, N)

    data["Curricular units 2nd sem (credited)"] = RNG.integers(0, 20, N)
    data["Curricular units 2nd sem (enrolled)"] = RNG.integers(4, 10, N)
    data["Curricular units 2nd sem (evaluations)"] = RNG.integers(4, 15, N)
    data["Curricular units 2nd sem (approved)"] = approved_2
    data["Curricular units 2nd sem (grade)"] = grade_2
    data["Curricular units 2nd sem (without evaluations)"] = RNG.integers(0, 3, N)

    data["Unemployment rate"] = RNG.uniform(7, 17, N)
    data["Inflation rate"] = RNG.uniform(-1, 4, N)
    data["GDP"] = RNG.uniform(-4, 4, N)

    # Build outcome based on correlated factors
    dropout_score = (
        (grade_1 < 8).astype(float) * 0.3
        + (grade_2 < 8).astype(float) * 0.3
        + (approved_1 < 3).astype(float) * 0.2
        + (debtor == 1).astype(float) * 0.1
        + (tuition == 0).astype(float) * 0.1
        - (scholarship == 1).astype(float) * 0.1
    )
    dropout_score += RNG.normal(0, 0.1, N)

    targets = []
    for score in dropout_score:
        if score > 0.55:
            targets.append("Dropout")
        elif score > 0.25:
            targets.append("Enrolled")
        else:
            targets.append("Graduate")
    data["Target"] = targets

    df = pd.DataFrame(data)
    out_path = Path(__file__).parent / "dataset.csv"
    df.to_csv(out_path, sep=";", index=False)
    print(f"Sample dataset written to {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    generate()
