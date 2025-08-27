# app_streamlit.py
# -------------------------------------------------------------
# Professional Sentiment Analysis Prototype (Streamlit) — Session-State Safe
# - Stores the loaded model in st.session_state so it persists across reruns/tabs
# - Single-text prediction with confidence
# - Batch predictions from CSV (columns: text or review/content)
# - Optional display of metrics if you provide outputs/summary.json and metrics_summary.csv
# - Clear UI, error handling, and rubric-aligned documentation blocks
# -------------------------------------------------------------

import json
from pathlib import Path
import os
import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Helpers
# ------------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def simple_clean(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = t.translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(Path(path))


def infer(model, text: str):
    # clean -> let the *pipeline* handle vectorization
    cleaned = simple_clean(text)
    X = [cleaned]  # a list of one string works well with sklearn Pipeline

    # prediction from the full pipeline
    pred = model.predict(X)[0]

    # optional confidence using the *pipeline* interface
    conf = None
    try:
        if hasattr(model, "decision_function"):
            dfun = model.decision_function(X)
            conf = float(1 / (1 + np.exp(-np.max(dfun)))) if np.ndim(dfun) else float(1 / (1 + np.exp(-dfun)))
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = float(np.max(proba))
    except Exception:
        pass

    return str(pred), conf



def normalize_text_column(df: pd.DataFrame) -> pd.Series:
    """Return a Series containing text from common column names."""
    for c in df.columns:
        if c.lower() in {"text", "review", "content", "sentence", "comment"}:
            return df[c].astype(str)
    raise ValueError("CSV must contain a text-like column (text/review/content/sentence/comment).")


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Sentiment Analysis Pro", page_icon="📊", layout="wide")

st.title("📊 Sentiment Analysis — Professional Prototype")
st.caption("Load your trained model, analyze text sentiment, and export batch predictions.")

with st.sidebar:
    st.header("⚙️ Model Settings")
    default_model = "models/best_model_LinearSVM.joblib"
    model_path = st.text_input("Path to saved .joblib model", value=default_model)
    load_button = st.button("Load / Reload Model")

    st.markdown("---")
    st.subheader("📄 Optional: Show Evaluation Artifacts")
    metrics_csv_path = st.text_input("metrics_summary.csv (optional)", value="outputs/metrics_summary.csv")
    summary_json_path = st.text_input("summary.json (optional)", value="outputs/summary.json")

# Persist and auto-load model via session_state
if load_button:
    try:
        st.session_state["model"] = load_model(model_path)
        st.success(f"Model loaded: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Optional: auto-load on first run if file exists and model not yet loaded
if "model" not in st.session_state and Path("models/best_model_LinearSVM.joblib").exists():
    try:
        st.session_state["model"] = load_model("models/best_model_LinearSVM.joblib")
        st.info("Auto-loaded models/best_model_LinearSVM.joblib")
    except Exception:
        pass

# -------------- Main Tabs --------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Single Prediction", "📦 Batch Predictions", "📈 Evaluation Summary", "📘 Help & Methodology"]
)

# --- Single Prediction ---
with tab1:
    st.subheader("🔍 Analyze a Single Text")
    text = st.text_area("Enter text to analyze", "I absolutely loved this product! It works great.", height=150)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        predict_btn = st.button("Predict Sentiment", type="primary")
    with col_b:
        clean_preview = st.checkbox("Show preprocessing preview")

    if clean_preview:
        st.code(simple_clean(text), language="text")

    if predict_btn:
        if "model" not in st.session_state:
            st.warning("Please load a model from the sidebar first.")
        else:
            label, conf = infer(st.session_state["model"], text)
            st.success(f"Prediction: **{label}**")
            if conf is not None:
                st.write(f"Confidence (approx): **{conf:.3f}**")

# --- Batch Predictions ---
with tab2:
    st.subheader("📦 Batch Predictions from CSV")
    st.caption("Upload a CSV with a column named text/review/content/sentence/comment.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
            texts = normalize_text_column(df_in)
            st.write("Preview:")
            st.dataframe(df_in.head(10))

            if "model" not in st.session_state:
                st.warning("Please load a model from the sidebar first.")
            else:
                preds, confs = [], []
                for t in texts.tolist():
                    yhat, conf = infer(st.session_state["model"], t)
                    preds.append(yhat)
                    confs.append(conf)
                out = df_in.copy()
                out["prediction"] = preds
                out["confidence"] = confs
                st.success("Batch predictions complete.")
                st.dataframe(out.head(20))

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions CSV",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# --- Evaluation Summary ---
with tab3:
    st.subheader("📈 Evaluation Summary (Optional Artifacts)")
    cols = st.columns(2)
    with cols[0]:
        if Path(metrics_csv_path).exists():
            try:
                dfm = pd.read_csv(metrics_csv_path)
                st.write("**Metrics Summary (from training run)**")
                st.dataframe(dfm)
            except Exception as e:
                st.error(f"Could not read metrics CSV: {e}")
        else:
            st.info("Upload or point to outputs/metrics_summary.csv to display.")
    with cols[1]:
        if Path(summary_json_path).exists():
            try:
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                st.write("**Best Model Summary (from training run)**")
                st.json(js)
            except Exception as e:
                st.error(f"Could not read summary JSON: {e}")
        else:
            st.info("Upload or point to outputs/summary.json to display.")

    st.markdown("---")
    st.write("**Confusion Matrices (if present in outputs/):**")
    cm_paths = []
    out_dir = Path("outputs")
    if out_dir.exists():
        for fn in os.listdir(out_dir):
            if fn.lower().startswith("confusion_matrix_") and fn.lower().endswith((".png", ".jpg", ".jpeg")):
                cm_paths.append(out_dir / fn)
    if cm_paths:
        for p in cm_paths:
            st.image(str(p), caption=p.name)
    else:
        st.info("Place confusion_matrix_*.png files in outputs/ to display them here.")

# --- Help & Methodology ---
with tab4:
    st.subheader("📘 How to Use & Marking Rubric Alignment")
    st.markdown(
        """
        **Steps to run**
        1. Train your models in Jupyter (we provided a full notebook). The best model is saved as a `.joblib` in the `models/` folder.
        2. Start this app from your terminal:  
           `streamlit run app_streamlit.py`
        3. In the sidebar, set the **model path** (e.g., `models/best_model_LinearSVM.joblib`) and click **Load / Reload Model**.
        4. Use **Single Prediction** to analyze a sentence, or **Batch Predictions** to upload a CSV.
        5. (Optional) Provide `outputs/metrics_summary.csv` and `outputs/summary.json` to display training metrics.

        **CSV for batch predictions** must include a text-like column named one of:  
        `text`, `review`, `content`, `sentence`, or `comment`.

        ---
        **How this app maps to the marking rubric**
        - **Functionality & Correctness:** Real model loaded, deterministic predictions, batch export, optional metrics display.
        - **Use of AI Techniques:** Classical ML with TF–IDF + LinearSVM/Naive Bayes/KNN; consistent with documentation.
        - **Evaluation & Analysis:** Shows metrics summary, best model JSON, and (if present) confusion matrices.
        - **Usability & Interface:** Clear tabs, sidebar configuration, explicit error handling, download artifact.
        - **Reproducibility:** Uses the same saved model used in the report; preprocessing shown/consistent.
        - **Professionalism:** Clean copy, labels, and inline help; suitable for demo to lecturer.
        """
    )

    with st.expander("Troubleshooting"):
        st.write("- **Model failed to load**: verify the path and that it was saved with joblib as a Pipeline containing TF-IDF + classifier.")
        st.write("- **Weird predictions**: confirm your training and app use identical preprocessing (this app uses the pipeline's TF-IDF).")
        st.write("- **Batch CSV errors**: ensure your CSV has a `text`-like column. Try renaming your column to `text`.")
        st.write("- **No metrics shown**: ensure outputs/metrics_summary.csv and outputs/summary.json exist and paths are correct.")
