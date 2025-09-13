import matplotlib
matplotlib.use("Agg")
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

def infer_feature_types(df: pd.DataFrame, target: str, drop_cols=None):
    drop_cols = drop_cols or []
    usable = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    usable = usable.drop(columns=[target], errors="ignore")

    numeric_cols = usable.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in usable.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    # Typical Telco cleaning
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Strip whitespace from object cols
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    # Map target if needed
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"])
    return df

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return pre

def get_models(random_state=42):
    models = {
        "logreg": LogisticRegression(max_iter=200, solver="lbfgs"),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=2, random_state=random_state
        ),
        "xgb": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=random_state, n_jobs=4
        ),
    }
    return models

def plot_and_save_roc(model, X_test, y_test, out_path):
    disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_and_save_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"],
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--data", required=True, help="Path to Telco CSV, e.g., data/Telco-Customer-Churn.csv")
    parser.add_argument("--target", default="Churn", help="Target column name (default: Churn)")
    parser.add_argument("--drop_cols", nargs="*", default=["customerID"], help="Columns to drop before training")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df = clean_telco(df)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in data. Available: {list(df.columns)[:10]} ...")

    # Drop rows with missing target
    df = df.dropna(subset=[args.target])

    X = df.drop(columns=[c for c in args.drop_cols if c in df.columns] + [args.target], errors="ignore")
    y = df[args.target].astype(int)

    num_cols, cat_cols = infer_feature_types(df, args.target, drop_cols=args.drop_cols)
    pre = build_preprocessor(num_cols, cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    models = get_models(random_state=args.random_state)

    best_name = None
    best_model = None
    best_auc = -np.inf

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    print("=== Cross‑validating models (ROC AUC) ===")
    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", clf)])
        aucs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"{name:>6}: mean AUC={aucs.mean():.4f} (+/- {aucs.std():.4f})")
        if aucs.mean() > best_auc:
            best_auc = aucs.mean()
            best_name = name
            best_model = pipe

    # Fit best model on full train
    best_model.fit(X_train, y_train)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, y_prob)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n=== Best Model on Test Set ===")
    print(f"Model: {best_name}")
    print(f"ROC AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")

    # Save artifacts
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")
    print(f"Saved: {models_dir / 'best_model.joblib'}")

    artifacts_dir = Path("artifacts"); artifacts_dir.mkdir(exist_ok=True)
    feature_info = {"numeric": num_cols, "categorical": cat_cols}
    with open(artifacts_dir / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    print(f"Saved: {artifacts_dir / 'feature_info.json'}")

    # Save plots
    reports_dir = Path("reports"); reports_dir.mkdir(exist_ok=True)
    plot_and_save_roc(best_model, X_test, y_test, reports_dir / "roc_curve.png")
    plot_and_save_confusion(y_test, y_pred, reports_dir / "confusion_matrix.png")
    print(f"Saved: {reports_dir / 'roc_curve.png'} & {reports_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
