"""
Run this file once to generate sample CSV datasets for testing.
Run it with:  python generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs("sample_data", exist_ok=True)

# Titanic-like dataset
n = 891
titanic = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Survived":    np.random.choice([0, 1], size=n, p=[0.62, 0.38]),
    "Pclass":      np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55]),
    "Name":        [f"Passenger_{i}" for i in range(1, n + 1)],
    "Sex":         np.random.choice(["male", "female"], size=n, p=[0.65, 0.35]),
    "Age":         np.where(
                       np.random.rand(n) < 0.2, np.nan,
                       np.clip(np.random.normal(29.7, 14.5, n), 0.42, 80)
                   ),
    "SibSp":       np.random.choice(range(0, 6), size=n, p=[0.68, 0.23, 0.03, 0.02, 0.02, 0.02]),
    "Parch":       np.random.choice(range(0, 5), size=n, p=[0.76, 0.13, 0.08, 0.02, 0.01]),
    "Fare":        np.clip(np.random.exponential(32, n), 0, 512),
    "Embarked":    np.random.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09]),
})
titanic.to_csv("sample_data/titanic.csv", index=False)
print("Done! sample_data/titanic.csv created.")
