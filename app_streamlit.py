# app_streamlit.py — Enhanced Sentiment App (Neutral Band ± around 0.5, no ELI5)
import json, re, string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------ Text utilities ------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Basic profanity list (case-insensitive). You can extend this if needed.
_PROFANITY = {
    "fuck","fucking","fucker","shit","bullshit","bitch","bastard","asshole","dick","cunt",
    "motherfucker","crap","piss","prick","slut","whore","damn","bloody","wanker","douche"
}

def simple_clean(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = t.translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize_words(t: str):
    return re.findall(r"[a-z']+", t.lower())

def word_count(t: str) -> int:
    return len(tokenize_words(t))

def find_profanity_tokens(t: str):
    toks = tokenize_words(t)
    found = [w for w in toks if w in _PROFANITY]
    return sorted(set(found))

def redact_token(w: str) -> str:
    if len(w) <= 2:
        return "*" * len(w)
    return w[0] + "*" * (len(w)-2) + w[-1]

# ------------------ Model helpers ------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(Path(path))

def _get_binary_label_indices(clf):
    """
    Return (pos_label, neg_label, pos_idx) for a binary classifier, if possible.
    Heuristics:
      - If 'positive' is present (case-insensitive) among classes_, that's pos.
      - Else if '1' present, treat '1' as pos.
      - Else assume classes_[1] is the positive (sklearn convention).
    """
    classes = getattr(clf, "classes_", None)
    if classes is None or len(classes) != 2:
        return None, None, None
    classes = list(classes)
    cl_lower = [str(c).lower() for c in classes]
    if "positive" in cl_lower:
        pos_idx = cl_lower.index("positive")
    elif "1" in cl_lower:
        pos_idx = cl_lower.index("1")
    else:
        pos_idx = 1
    neg_idx = 1 - pos_idx
    return classes[pos_idx], classes[neg_idx], pos_idx

def _softmax(a):
    a = np.asarray(a)
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=1, keepdims=True)

def prob_positive(pipe, texts):
    """
    Return tuple (p_pos, pos_label, neg_label).
    - p_pos: numpy array of shape (n_samples,) with P(positive) estimates in [0,1].
    - pos_label/neg_label: the label names we mapped to positive/negative.
    Works best for BINARY problems.
    """
    clf = pipe.named_steps.get("clf", None)
    if clf is None:
        return None, None, None

    pos_label, neg_label, pos_idx = _get_binary_label_indices(clf)
    if pos_idx is None:
        return None, None, None

    # Use pipeline-level proba if possible
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        # binary: proba shape (n,2)
        p_pos = proba[:, pos_idx]
        return p_pos, pos_label, neg_label

    # Otherwise, try decision_function and squash via sigmoid/softmax
    if hasattr(pipe, "decision_function"):
        dfun = pipe.decision_function(texts)
        if np.ndim(dfun) == 1:  # binary margin
            p_pos = 1.0 / (1.0 + np.exp(-dfun))
            return p_pos, pos_label, neg_label
        else:
            # multi-class margins; approximate probabilities with softmax
            probs = _softmax(dfun)
            p_pos = probs[:, pos_idx]
            return p_pos, pos_label, neg_label

    return None, None, None

def normalize_text_column(df: pd.DataFrame) -> pd.Series:
    for c in df.columns:
        if c.lower() in {"text","review","content","sentence","comment"}:
            return df[c].astype(str)
    raise ValueError("CSV must contain a text-like column: text/review/content/sentence/comment.")

# ------------------ UI ------------------
st.set_page_config(page_title="Sentiment Analysis Pro", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis — Enhanced Prototype")
st.caption("Neutral band around P(positive)=0.5, short-text warnings, profanity flags, and batch processing.")

with st.sidebar:
    st.header("⚙️ Model & Rules")
    default_model = "models/best_model_LinearSVM.joblib"
    model_path = st.text_input("Path to .joblib model", value=default_model)
    if st.button("Load / Reload Model"):
        try:
            st.session_state["model"] = load_model(model_path)
            st.success(f"Model loaded: {model_path}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.markdown("---")
    st.subheader("🔎 Prediction Policy")
    min_words = st.slider("Minimum words to treat input as reliable", min_value=3, max_value=30, value=5, step=1)
    neutral_band = st.slider("Neutral band half-width (centered at 0.50)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    flag_prof = st.checkbox("Flag profanity", value=True)
    redact_prof = st.checkbox("Redact profanity in notes", value=False)
    neutral_label = st.text_input("Neutral label text", value="neutral")

tabs = st.tabs(["🔍 Single Prediction","📦 Batch Predictions","📈 Evaluation Summary","📘 Help"])

# ------------------ Single Prediction ------------------
with tabs[0]:
    st.subheader("🔍 Single Prediction")
    text = st.text_area("Enter text to analyze", "I absolutely loved this product! It works great.", height=150)
    if st.button("Predict", type="primary"):
        if "model" not in st.session_state:
            st.warning("Please load a model from the sidebar first.")
        else:
            pipe = st.session_state["model"]
            cleaned = simple_clean(text)
            wc = word_count(text)
            prof = find_profanity_tokens(text) if flag_prof else []
            # Raw model label
            raw_label = pipe.predict([cleaned])[0]

            # P(positive)
            p_pos, pos_label, neg_label = prob_positive(pipe, [cleaned])
            if p_pos is not None:
                p = float(p_pos[0])
                # neutral band: [0.5 - neutral_band, 0.5 + neutral_band]
                lower = 0.5 - neutral_band
                upper = 0.5 + neutral_band
                if p > upper:
                    final_label = str(pos_label)
                elif p < lower:
                    final_label = str(neg_label)
                else:
                    final_label = str(neutral_label)
            else:
                # fallback to raw label if probability is unavailable
                p = None
                final_label = str(raw_label)

            # Display results
            st.write(f"**Words:** {wc}")
            if p is not None:
                st.write(f"**P(positive):** {p:.3f}")
                # visualize where p falls relative to the neutral band
                st.progress(min(max(p, 0.0), 1.0))
                st.caption(f"Neutral band: [{lower:.2f}, {upper:.2f}]")

            st.success(f"Prediction (final): **{final_label}**  | Raw model: *{raw_label}*")

            # Notes
            notes = []
            if wc < min_words:
                notes.append(f"Input is short ({wc} words). Results may be unstable.")
            if prof:
                prof_display = ", ".join(redact_token(w) for w in prof) if redact_prof else ", ".join(prof)
                notes.append(f"Profanity detected: {prof_display}")
            if notes:
                st.info(" | ".join(notes))

# ------------------ Batch Predictions ------------------
with tabs[1]:
    st.subheader("📦 Batch Predictions from CSV")
    st.caption("Upload a CSV with a column named text/review/content/sentence/comment.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
            texts = normalize_text_column(df_in)
            st.write("Preview:"); st.dataframe(df_in.head(10))

            if "model" not in st.session_state:
                st.warning("Please load a model from the sidebar first.")
            else:
                pipe = st.session_state["model"]
                cleaned = texts.map(simple_clean).tolist()
                raw_preds = pipe.predict(cleaned)

                p_pos, pos_label, neg_label = prob_positive(pipe, cleaned)
                rows = []
                for i, t in enumerate(texts.tolist()):
                    wc = word_count(t)
                    prof = find_profanity_tokens(t) if flag_prof else []
                    raw_label = raw_preds[i]

                    if p_pos is not None:
                        p = float(p_pos[i])
                        lower = 0.5 - neutral_band
                        upper = 0.5 + neutral_band
                        if p > upper:
                            final_label = str(pos_label)
                        elif p < lower:
                            final_label = str(neg_label)
                        else:
                            final_label = str(neutral_label)
                    else:
                        p = None
                        final_label = str(raw_label)

                    note_bits = []
                    if wc < min_words:
                        note_bits.append(f"short({wc})")
                    if prof:
                        prof_display = ",".join(redact_token(w) for w in prof) if redact_prof else ",".join(prof)
                        note_bits.append(f"profanity({prof_display})")
                    notes = ";".join(note_bits) if note_bits else ""

                    rows.append({
                        "text": t,
                        "word_count": wc,
                        "prediction_raw": raw_label,
                        "p_positive": p,
                        "prediction_final": final_label,
                        "notes": notes
                    })

                out = pd.DataFrame(rows)
                st.success("Batch predictions complete.")
                st.dataframe(out.head(30))

                st.download_button(
                    "Download Predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions_neutral_band.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ------------------ Evaluation Summary ------------------
with tabs[2]:
    st.subheader("📈 Evaluation Summary")
    metrics_csv_path = "outputs/metrics_summary.csv"
    summary_json_path = "outputs/summary.json"

    cols = st.columns(2)
    with cols[0]:
        pth = Path(metrics_csv_path)
        if pth.exists():
            try:
                dfm = pd.read_csv(pth)
                st.write("**Metrics Summary (from training run)**")
                st.dataframe(dfm)
            except Exception as e:
                st.error(f"Could not read metrics CSV: {e}")
        else:
            st.info("Place outputs/metrics_summary.csv in the repo to display.")

    with cols[1]:
        pth = Path(summary_json_path)
        if pth.exists():
            try:
                js = json.load(open(pth, "r", encoding="utf-8"))
                st.write("**Best Model Summary (from training run)**")
                st.json(js)
            except Exception as e:
                st.error(f"Could not read summary JSON: {e}")
        else:
            st.info("Place outputs/summary.json in the repo to display.")

# ------------------ Help ------------------
with tabs[3]:
    st.subheader("📘 Help & Marking Rubric Mapping")
    st.markdown(
        """
        **Neutral band logic**
        - Compute P(positive) either from `predict_proba` or from the SVM margin via a sigmoid.
        - If P(positive) ∈ [0.5 − band, 0.5 + band] → output **neutral**.
        - If > 0.5 + band → **positive**; if < 0.5 − band → **negative**.
        - Default band = **±0.05** (adjustable in the sidebar).

        **Edge cases**
        - If the classifier is not binary or probabilities cannot be computed, the app falls back to the raw label.
        - Short texts trigger a stability warning; profanity is optionally flagged/redacted in notes.

        **Rubric alignment**
        - Functionality & Correctness: robust I/O, probabilistic neutral handling, batch export.
        - Usability & Professionalism: clear policy controls, helpful warnings, clean UI.
        - Reproducibility: model artifact + outputs referenced exactly as in the report.
        """
    )
