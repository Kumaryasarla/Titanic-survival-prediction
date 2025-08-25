
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

CATEGORICAL_COLS = ["Sex", "Embarked", "Title"]
DROP_COLS = ["Name", "Ticket", "Cabin"]

def _safe_mode(series: pd.Series):
    try:
        return series.mode(dropna=True).iloc[0]
    except Exception:
        return series.dropna().iloc[0] if series.dropna().shape[0] else None

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        mode_val = _safe_mode(df["Embarked"])
        df["Embarked"] = df["Embarked"].fillna(mode_val if mode_val is not None else "S")
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Title extraction
    if "Name" in df.columns:
        titles = df["Name"].astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
        titles = titles.replace(
            ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
            "Rare"
        ).replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
        df["Title"] = titles
    else:
        if "Title" not in df.columns:
            df["Title"] = "Unknown"

    # Drop high-cardinality text columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")

    # Encode categoricals
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = df[c].astype(str)
            df[c] = le.fit_transform(df[c])
    return df

def prepare_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    df_proc = _add_engineered_features(df)
    if "Survived" not in df.columns:
        raise ValueError("Training data must include 'Survived' column.")
    y = df["Survived"].astype(int).values
    X = df_proc.drop(columns=["Survived"], errors="ignore")
    if "PassengerId" in X.columns:
        X = X.drop(columns=["PassengerId"])
    return X, y

def prepare_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    df_proc = _add_engineered_features(df)
    ids = df_proc["PassengerId"].values if "PassengerId" in df_proc.columns else np.arange(len(df_proc))
    X = df_proc.drop(columns=["PassengerId"], errors="ignore")
    return X, ids

def train_and_validate(train_df: pd.DataFrame, random_state: int = 42) -> Dict[str, Any]:
    X, y = prepare_train(train_df)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    model = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2,
        random_state=random_state, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    # 5-fold CV on full train set for robustness
    cv_acc = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_mean = float(np.mean(cv_acc))
    cv_std = float(np.std(cv_acc))

    report_text = classification_report(y_val, y_pred)

    # Feature importances (if available)
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ROC curve points
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)

    return {
        "model": model,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(auc),
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_text,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist()
            }
        },
        "feature_importances": feat_imp
    }

def predict_submission(model, test_df: pd.DataFrame) -> pd.DataFrame:
    X_test, ids = prepare_test(test_df)
    preds = model.predict(X_test)
    return pd.DataFrame({"PassengerId": ids, "Survived": preds.astype(int)})
