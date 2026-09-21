from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .config import PROCESSED_DIR, MODELS_DIR, METRICS_DIR, RANDOM_STATE

CLASS_FEATURES = [
    "UserContinent",
    "UserRegion",
    "UserCountry",
    "UserCityName",
    "VisitYear",
    "VisitMonth",
    "Attraction",
    "AttractionTypeLabel",
    "AttractionCityName",
    "AttractionCountry",
]
REG_FEATURES = CLASS_FEATURES + ["VisitModeName"]
NUMERIC_FEATURES = ["VisitYear", "VisitMonth"]


def onehot_preprocessor(features: list[str]) -> ColumnTransformer:
    cats = [c for c in features if c not in NUMERIC_FEATURES]
    nums = [c for c in features if c in NUMERIC_FEATURES]
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                ]),
                cats,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                nums,
            ),
        ]
    )


def ordinal_preprocessor(features: list[str]) -> ColumnTransformer:
    cats = [c for c in features if c not in NUMERIC_FEATURES]
    nums = [c for c in features if c in NUMERIC_FEATURES]
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                cats,
            ),
            ("num", SimpleImputer(strategy="median"), nums),
        ],
        verbose_feature_names_out=False,
    )


def _split(X, y, stratify=None):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify
    )
    strat_temp = y_temp if stratify is not None else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_classification(df: pd.DataFrame) -> dict:
    X = df[CLASS_FEATURES].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["VisitModeName"].astype(str))
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, stratify=y)

    models = {
        "Logistic Regression": Pipeline([
            ("prep", onehot_preprocessor(CLASS_FEATURES)),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", n_jobs=2, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("prep", ordinal_preprocessor(CLASS_FEATURES)),
            ("model", RandomForestClassifier(n_estimators=140, max_depth=18, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=2, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("prep", ordinal_preprocessor(CLASS_FEATURES)),
            ("model", XGBClassifier(n_estimators=160, max_depth=7, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", n_jobs=2, random_state=RANDOM_STATE)),
        ]),
    }

    rows = []
    fitted = {}
    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        rows.append({
            "task": "classification",
            "model": name,
            "validation_accuracy": accuracy_score(y_val, val_pred),
            "validation_precision_weighted": precision_score(y_val, val_pred, average="weighted", zero_division=0),
            "validation_recall_weighted": recall_score(y_val, val_pred, average="weighted", zero_division=0),
            "validation_f1_weighted": f1_score(y_val, val_pred, average="weighted", zero_division=0),
            "validation_f1_macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
            "fit_seconds": round(time.time() - t0, 3),
        })
        fitted[name] = pipe

    comp = pd.DataFrame(rows).sort_values("validation_f1_weighted", ascending=False)
    best_name = comp.iloc[0]["model"]
    best_model = fitted[best_name]
    test_pred = best_model.predict(X_test)
    test_proba = best_model.predict_proba(X_test)

    test_metrics = {
        "best_model": best_name,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision_weighted": float(precision_score(y_test, test_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, test_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "classes": label_encoder.classes_.tolist(),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    report = classification_report(y_test, test_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_test, test_pred), index=label_encoder.classes_, columns=label_encoder.classes_)

    comp.to_csv(METRICS_DIR / "classification_model_comparison.csv", index=False)
    cm.to_csv(METRICS_DIR / "classification_confusion_matrix.csv")
    with open(METRICS_DIR / "classification_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(METRICS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    joblib.dump(best_model, MODELS_DIR / "best_classification_model.joblib", compress=3)
    joblib.dump(label_encoder, MODELS_DIR / "visit_mode_label_encoder.joblib", compress=3)

    # Compact prediction audit sample.
    audit = X_test.head(250).copy()
    audit["actual_visit_mode"] = label_encoder.inverse_transform(y_test[:250])
    audit["predicted_visit_mode"] = label_encoder.inverse_transform(test_pred[:250])
    audit["confidence"] = test_proba[:250].max(axis=1)
    audit.to_csv(METRICS_DIR / "classification_prediction_sample.csv", index=False)
    return {"comparison": comp, "test": test_metrics}


def train_regression(df: pd.DataFrame) -> dict:
    X = df[REG_FEATURES].copy()
    y = df["Rating"].astype(float).to_numpy()
    # Stratify by integer rating only for balanced splitting; the regression target remains continuous numeric.
    X_train, X_val, X_test, y_train, y_val, y_test = _split(X, y, stratify=y.astype(int))

    models = {
        "Ridge Regression": Pipeline([
            ("prep", onehot_preprocessor(REG_FEATURES)),
            ("model", Ridge(alpha=2.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep", ordinal_preprocessor(REG_FEATURES)),
            ("model", RandomForestRegressor(n_estimators=140, max_depth=18, min_samples_leaf=3, n_jobs=2, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("prep", ordinal_preprocessor(REG_FEATURES)),
            ("model", XGBRegressor(n_estimators=180, max_depth=7, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", n_jobs=2, random_state=RANDOM_STATE)),
        ]),
    }

    rows = []
    fitted = {}
    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        val_pred = np.clip(pipe.predict(X_val), 1, 5)
        rmse = math.sqrt(mean_squared_error(y_val, val_pred))
        rows.append({
            "task": "regression",
            "model": name,
            "validation_rmse": rmse,
            "validation_mse": mean_squared_error(y_val, val_pred),
            "validation_mae": mean_absolute_error(y_val, val_pred),
            "validation_r2": r2_score(y_val, val_pred),
            "fit_seconds": round(time.time() - t0, 3),
        })
        fitted[name] = pipe

    comp = pd.DataFrame(rows).sort_values("validation_rmse", ascending=True)
    best_name = comp.iloc[0]["model"]
    best_model = fitted[best_name]
    test_pred = np.clip(best_model.predict(X_test), 1, 5)
    test_metrics = {
        "best_model": best_name,
        "rmse": float(math.sqrt(mean_squared_error(y_test, test_pred))),
        "mse": float(mean_squared_error(y_test, test_pred)),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "r2": float(r2_score(y_test, test_pred)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }

    comp.to_csv(METRICS_DIR / "regression_model_comparison.csv", index=False)
    with open(METRICS_DIR / "regression_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    joblib.dump(best_model, MODELS_DIR / "best_regression_model.joblib", compress=3)

    audit = X_test.head(250).copy()
    audit["actual_rating"] = y_test[:250]
    audit["predicted_rating"] = test_pred[:250]
    audit["absolute_error"] = np.abs(audit["actual_rating"] - audit["predicted_rating"])
    audit.to_csv(METRICS_DIR / "regression_prediction_sample.csv", index=False)
    return {"comparison": comp, "test": test_metrics}


def _feature_importance_from_pipeline(model_path: Path, features: list[str], out_name: str):
    pipe = joblib.load(model_path)
    estimator = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = np.array(features)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coefs), axis=0) if coefs.ndim > 1 else np.abs(coefs)
    else:
        return
    n = min(len(names), len(values))
    fi = pd.DataFrame({"feature": names[:n], "importance": values[:n]}).sort_values("importance", ascending=False)
    fi.head(100).to_csv(METRICS_DIR / out_name, index=False)


def train_all() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "tourism_merged_clean.csv")
    classification = train_classification(df)
    regression = train_regression(df)
    _feature_importance_from_pipeline(MODELS_DIR / "best_classification_model.joblib", CLASS_FEATURES, "classification_feature_importance.csv")
    _feature_importance_from_pipeline(MODELS_DIR / "best_regression_model.joblib", REG_FEATURES, "regression_feature_importance.csv")

    schema = {
        "classification_features": CLASS_FEATURES,
        "regression_features": REG_FEATURES,
        "classification_target": "VisitModeName",
        "regression_target": "Rating",
        "split": "70% train / 15% validation / 15% test, random_state=42; target-stratified",
        "leakage_note": "Rating is deliberately excluded from VisitMode classification; UserId is excluded from both predictive models to avoid memorizing user identity.",
    }
    with open(MODELS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    summary = {
        "classification_best": classification["test"],
        "regression_best": regression["test"],
    }
    with open(METRICS_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train_all()
