
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from model_utils import train_and_validate, predict_submission

st.set_page_config(page_title="Titanic Survival — Y.Kumar", page_icon="🚢", layout="wide")
st.title("🚢 Titanic Survival Prediction — Y. Kumar")

st.markdown("""
This app lets clients clearly see data insights, training metrics, model diagnostics, and download predictions — with no ambiguity.
""")

# Sidebar: data options
st.sidebar.header("Data")
use_bundled_train = st.sidebar.checkbox("Use bundled train.csv", value=True)
use_bundled_test = st.sidebar.checkbox("Use bundled test.csv", value=True)

uploaded_train = None
uploaded_test = None
if not use_bundled_train:
    uploaded_train = st.sidebar.file_uploader("Upload training CSV (must include 'Survived')", type=["csv"])
if not use_bundled_test:
    uploaded_test = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

# Load data
if use_bundled_train:
    train_df = pd.read_csv("data/train.csv")
else:
    if uploaded_train is None:
        st.stop()
    train_df = pd.read_csv(uploaded_train)

if use_bundled_test:
    test_df = pd.read_csv("data/test.csv")
else:
    if uploaded_test is None:
        st.stop()
    test_df = pd.read_csv(uploaded_test)

tabs = st.tabs(["📊 EDA", "🧠 Train & Metrics", "📈 Diagnostics", "📥 Predict & Download"])

# ---------------- EDA ----------------
with tabs[0]:
    st.subheader("Quick Glance")
    st.write("Top rows of training data:")
    st.dataframe(train_df.head())

    numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]

    col1, col2, col3 = st.columns(3)

    # Age distribution
    if "Age" in train_df.columns:
        with col1:
            fig = plt.figure()
            train_df["Age"].dropna().plot(kind="hist", bins=30)
            plt.title("Age Distribution")
            plt.xlabel("Age")
            plt.ylabel("Count")
            st.pyplot(fig)

    # Fare distribution
    if "Fare" in train_df.columns:
        with col2:
            fig = plt.figure()
            train_df["Fare"].dropna().plot(kind="hist", bins=30)
            plt.title("Fare Distribution")
            plt.xlabel("Fare")
            plt.ylabel("Count")
            st.pyplot(fig)

    # Survival by Sex
    if "Survived" in train_df.columns and "Sex" in train_df.columns:
        with col3:
            fig = plt.figure()
            train_df.groupby("Sex")["Survived"].mean().plot(kind="bar")
            plt.title("Survival Rate by Sex")
            plt.xlabel("Sex")
            plt.ylabel("Mean Survived")
            st.pyplot(fig)

    # Survival by Pclass
    if "Survived" in train_df.columns and "Pclass" in train_df.columns:
        fig = plt.figure()
        train_df.groupby("Pclass")["Survived"].mean().plot(kind="bar")
        plt.title("Survival Rate by Pclass")
        plt.xlabel("Pclass")
        plt.ylabel("Mean Survived")
        st.pyplot(fig)

# ---------------- Train & Metrics ----------------
with tabs[1]:
    st.subheader("Train Model")
    if st.button("Train RandomForest on current training data"):
        try:
            result = train_and_validate(train_df)
            st.session_state["result"] = result
            st.success("Training completed successfully.")
            m = result["metrics"]
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Accuracy", f"{m['accuracy']:.3f}")
            colB.metric("Precision", f"{m['precision']:.3f}")
            colC.metric("Recall", f"{m['recall']:.3f}")
            colD.metric("F1-score", f"{m['f1']:.3f}")
            st.caption(f"ROC AUC: {m['roc_auc']:.3f} • CV Accuracy (5-fold): {m['cv_mean_accuracy']:.3f} ± {m['cv_std_accuracy']:.3f}")
            st.code(m["classification_report"])
        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------------- Diagnostics ----------------
with tabs[2]:
    st.subheader("Model Diagnostics")
    if "result" not in st.session_state:
        st.info("Train the model first.")
    else:
        m = st.session_state["result"]["metrics"]

        # Confusion Matrix
        st.markdown("**Confusion Matrix**")
        cm = np.array(m["confusion_matrix"])
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xticks([0,1], ["Pred 0","Pred 1"])
        plt.yticks([0,1], ["True 0","True 1"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

        # ROC curve
        st.markdown("**ROC Curve**")
        roc = m["roc_curve"]
        fpr, tpr = np.array(roc["fpr"]), np.array(roc["tpr"])
        fig = plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        st.pyplot(fig)

        # Feature Importances
        fi = st.session_state["result"]["feature_importances"]
        if fi is not None and len(fi) > 0:
            st.markdown("**Top Features**")
            st.dataframe(fi.head(15))
            fig = plt.figure()
            fi.head(15).set_index("feature")["importance"].plot(kind="barh")
            plt.title("Feature Importances (Top 15)")
            st.pyplot(fig)

# ---------------- Predict & Download ----------------
with tabs[3]:
    st.subheader("Generate Predictions")
    if "result" not in st.session_state:
        st.info("Train the model first.")
    else:
        try:
            submission = predict_submission(st.session_state["result"]["model"], test_df)
            st.dataframe(submission.head())
            csv_bytes = submission.to_csv(index=False).encode("utf-8")
            st.download_button("Download submission.csv", data=csv_bytes, file_name="submission.csv", mime="text/csv")
            st.success("Prediction file ready.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
