import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, NUMERIC_FEATURES,
                     CATEGORICAL_FEATURES, TARGET)


def load_analytic_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "nhanes_2017_2018_mental_health.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.build_dataset` first.")
    return pd.read_csv(path)


def build_pipeline(model, numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])


def main():
    df = load_analytic_data()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        raise ValueError("No predictor columns found. Check downloaded NHANES components.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Both target classes are required for binary classification.")
    X, y = df[feature_cols].copy(), df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator, numeric_cols, categorical_cols)
        pipeline.fit(X_train, y_train)
        pred, prob = pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        joblib.dump({"pipeline": pipeline, "features": feature_cols, "target": TARGET},
                    MODELS_DIR / f"{name}.joblib")
        print(f"{name}: {results[name]}")
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {REPORTS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
