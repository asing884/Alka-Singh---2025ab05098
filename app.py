
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing — ML Models", layout="wide")

st.title("📊 Bank Marketing — Model Inference & Evaluation")
st.write(
    "Upload **test data** (CSV) and choose a trained model. "
    "If your CSV includes a **`y`** column (0/1 or 'no'/'yes'), the app will compute evaluation metrics."
)

# === IMPORTANT: Required input columns (duration and y are excluded) ===
REQUIRED_COLUMNS = [
    "age","job","marital","education","default","housing","loan",
    "contact","month","day_of_week","campaign","pdays","previous",
    "poutcome","emp.var.rate","cons.price.idx","cons.conf.idx",
    "euribor3m","nr.employed"
]

# Small CSV template for users to download
def get_template_df():
    # One empty row with headers only
    return pd.DataFrame(columns=REQUIRED_COLUMNS)

with st.expander("📥 Download CSV template (headers only)"):
    tmpl_csv = get_template_df().to_csv(index=False)
    st.download_button(
        label="Download template.csv",
        data=tmpl_csv,
        file_name="template_bank_marketing.csv",
        mime="text/csv"
    )
    st.caption("The uploaded CSV must have **exactly these headers** (order doesn’t matter).")

# === File uploader ===
uploaded = st.file_uploader(
    "Upload CSV (test split only). Columns must match the template headers.",
    type=["csv"]
)

# === Model selector (maps to pickles under model/) ===
MODEL_FILES = {
    "Logistic Regression":       "logistic_regression.pkl",
    "Decision Tree":             "decision_tree.pkl",
    "KNN":                       "knn.pkl",
    "Naive Bayes":               "naive_bayes.pkl",
    "Random Forest":             "random_forest.pkl",
    "XGBoost":                   "xgboost.pkl",
}

model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
run_btn = st.button("Run")

def normalize_target(y_series):
    # Accept 0/1 or 'no'/'yes' (case-insensitive)
    mapping = {"no": 0, "yes": 1, "0": 0, "1": 1}
    def _map_one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in mapping:
            return mapping[s]
        try:
            return int(float(s))
        except Exception:
            return np.nan
    return y_series.map(_map_one).astype("float")

def align_columns(df):
    # Keep only required; add any missing as NaN (pipeline will handle via OHE/transform)
    cols = df.columns.tolist()
    to_keep = [c for c in cols if c in REQUIRED_COLUMNS]
    out = df[to_keep].copy()
    for c in REQUIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    # Reorder to a stable order (not strictly required, but nice)
    out = out[REQUIRED_COLUMNS]
    return out

def load_model(pkl_name):
    p = Path("model") / pkl_name
    if not p.exists():
        st.error(f"Model file not found: {p}")
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

if run_btn:
    if uploaded is None:
        st.warning("Please upload a CSV first.")
        st.stop()

    # Read CSV
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    # Separate target if present
    y_true = None
    y_col_present = "y" in raw_df.columns
    if y_col_present:
        y_true = normalize_target(raw_df["y"])
        raw_df = raw_df.drop(columns=["y"])

    # Align columns to what the pipeline expects (minus y and duration)
    X = align_columns(raw_df)

    # Load model
    pipe = load_model(MODEL_FILES[model_name])
    if pipe is None:
        st.stop()

    # Predict
    try:
        y_pred = pipe.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Try probabilities for AUC if available
    if hasattr(pipe, "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
    else:
        y_proba = None

    st.subheader("🔎 Results")
    st.write(f"**Rows predicted:** {len(X)}")

    # If true target present, compute metrics
    if y_col_present:
        if y_true.isna().any():
            st.warning("Some values in the `y` column could not be interpreted as 0/1 or no/yes. They were treated as NaN and dropped for scoring.")
        # Drop rows where y is NaN to compute fair scores
        mask = ~y_true.isna()
        if mask.sum() == 0:
            st.error("No valid ground-truth labels found after cleaning `y`. Cannot compute metrics.")
        else:
            yt = y_true[mask].astype(int).values
            yp = np.asarray(y_pred)[mask]
            if y_proba is not None:
                ypb = np.asarray(y_proba)[mask]
            else:
                ypb = None

            # Metrics
            acc = accuracy_score(yt, yp)
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            mcc = matthews_corrcoef(yt, yp)
            auc = roc_auc_score(yt, ypb) if ypb is not None else float("nan")

            # Show metrics nicely
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("AUC", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
            c3.metric("Precision", f"{prec:.4f}")
            c4.metric("Recall", f"{rec:.4f}")
            c5.metric("F1", f"{f1:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")

            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(yt, yp)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification report
            st.markdown("#### Classification Report")
            st.code(classification_report(yt, yp, target_names=["no", "yes"]))
    else:
        st.info("No `y` column found in the uploaded CSV. Displaying predictions only.")
        # Show head of predictions and (if available) probabilities
        out = pd.DataFrame({"prediction": y_pred})
        if y_proba is not None:
            out["prob_yes"] = y_proba
        st.dataframe(out.head(20))
        st.download_button(
            "Download predictions (CSV)",
            out.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

st.caption("Note: Pipelines were trained with the `duration` column removed and expect the headers shown in the template.")
