
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)

st.set_page_config(page_title="Song Popularity Predictor", layout="wide")

# -----------------------------
# 1) Features used to train the models (your exact list & order)
# -----------------------------
FEATURE_COLS = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo',
    'track_genre_encoded','valence_energy','energy_danceability'
]

# -----------------------------
# 2) Models to load (adjust names if needed)
# -----------------------------
MODEL_FILES = {
    "RandomForest": "models/best_rf.pkl",
    "XGBoost":      "models/best_xgb.pkl",
    "CatBoost":     "models/cat.pkl",
    "Ensemble":     "models/ensemble.pkl",   # optional
}

METRICS_CSV = "model_metrics_summary.csv"    # optional (df_res.to_csv(...))

# -----------------------------
# 3) Helpers
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load {name} from {path}: {e}")
    return models

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_CSV):
        try:
            return pd.read_csv(METRICS_CSV)
        except Exception as e:
            st.warning(f"Could not read {METRICS_CSV}: {e}")
    # fallback demo table (replace with your CSV when ready)
    return pd.DataFrame([
        {"Model":"RandomForest","Accuracy Test":0.982,"Balanced Acc Test":0.793,"Weighted F1 Test":0.981,"ROC AUC Test":0.950},
        {"Model":"XGBoost","Accuracy Test":0.972,"Balanced Acc Test":0.808,"Weighted F1 Test":0.974,"ROC AUC Test":0.928},
        {"Model":"CatBoost","Accuracy Test":0.936,"Balanced Acc Test":0.826,"Weighted F1 Test":0.951,"ROC AUC Test":0.926},
        {"Model":"Ensemble","Accuracy Test":0.977,"Balanced Acc Test":0.629,"Weighted F1 Test":0.973,"ROC AUC Test":0.931},
    ])

def build_input_row():
    st.sidebar.header("Inputs (what-if)")
    dance  = st.sidebar.slider("danceability",     0.0, 1.0, 0.60, 0.01)
    energy = st.sidebar.slider("energy",           0.0, 1.0, 0.60, 0.01)
    loud   = st.sidebar.slider("loudness (dB)",   -60.0, 5.0, -7.0, 0.1)
    speech = st.sidebar.slider("speechiness",      0.0, 1.0, 0.05, 0.01)
    acous  = st.sidebar.slider("acousticness",     0.0, 1.0, 0.20, 0.01)
    instr  = st.sidebar.slider("instrumentalness", 0.0, 1.0, 0.00, 0.01)
    live   = st.sidebar.slider("liveness",         0.0, 1.0, 0.10, 0.01)
    val    = st.sidebar.slider("valence",          0.0, 1.0, 0.50, 0.01)
    tempo  = st.sidebar.slider("tempo (BPM)",     40.0, 220.0, 120.0, 1.0)
    genre  = st.sidebar.number_input("track_genre_encoded", 0, 1000, 0)

    df = pd.DataFrame([{
        'danceability': dance,
        'energy': energy,
        'loudness': loud,
        'speechiness': speech,
        'acousticness': acous,
        'instrumentalness': instr,
        'liveness': live,
        'valence': val,
        'tempo': tempo,
        'track_genre_encoded': genre
    }])

    # interaction features used in training
    df['valence_energy']      = df['valence'] * df['energy']
    df['energy_danceability'] = df['energy']  * df['danceability']

    # ensure exact order/columns
    df = df.reindex(columns=FEATURE_COLS, fill_value=0.0)
    return df

def predict_with(model, x_df, threshold=0.5):
    proba = float(model.predict_proba(x_df)[:, 1])
    pred  = int(proba >= threshold)
    return proba, pred

# -----------------------------
# 4) Layout
# -----------------------------
st.title("🎵 Song Popularity Predictor (Demo)")

tab_pred, tab_models, tab_explain = st.tabs(["Predict", "Models", "Explain"])

models = load_models()
metrics_df = load_metrics()

with tab_pred:
    colL, colR = st.columns([1,1])

    with colL:
        st.subheader("Choose a model")
        if not models:
            st.error("No models found in /models. Save your trained models as .pkl and reload the page.")
        model_name = st.selectbox("Model", list(models.keys()) if models else ["(no models)"])
        threshold  = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

        x_row = build_input_row()

        if st.button("Predict") and model_name in models:
            try:
                model = models[model_name]
                proba, pred = predict_with(model, x_row, threshold)
                st.metric("Probability of being popular (class 1)", f"{proba:.1%}")
                st.write(f"**Predicted class** (threshold {threshold:.2f}): **{pred}**")
                st.progress(min(max(proba, 0.0), 1.0))
                st.caption("0 = not popular | 1 = popular")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with colR:
        st.subheader("Model comparison (Test)")
        if not metrics_df.empty:
            cols = ["Model","Accuracy Test","Balanced Acc Test","Weighted F1 Test","ROC AUC Test"]
            show = [c for c in cols if c in metrics_df.columns]
            st.dataframe(metrics_df[show].set_index("Model") if "Model" in show else metrics_df)

with tab_models:
    st.subheader("Full metrics (Train & Test)")
    st.dataframe(metrics_df.set_index("Model") if "Model" in metrics_df.columns else metrics_df)

    st.markdown("---")
    st.subheader("Bar chart — Accuracy & AUC (Test)")
    if set(["Accuracy Test","ROC AUC Test"]).issubset(metrics_df.columns):
        st.bar_chart(metrics_df.set_index("Model")[["Accuracy Test","ROC AUC Test"]])

    if "Balanced Acc Test" in metrics_df.columns:
        st.markdown("### Bar chart — Balanced Accuracy (Test)")
        st.bar_chart(metrics_df.set_index("Model")[["Balanced Acc Test"]])

    if "Weighted F1 Test" in metrics_df.columns:
        st.markdown("### Bar chart — Weighted F1 (Test)")
        st.bar_chart(metrics_df.set_index("Model")[["Weighted F1 Test"]])

with tab_explain:
    st.subheader("Feature importance (selected model)")
    if models:
        model_for_exp = st.selectbox("Model to explain", list(models.keys()), key="exp_model")
        mdl = models[model_for_exp]
        if hasattr(mdl, "feature_importances_"):
            try:
                imp = pd.Series(mdl.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
                st.bar_chart(imp)
                st.dataframe(imp.rename("importance"))
            except Exception as e:
                st.warning(f"Could not display importances: {e}")
        else:
            st.info("This model does not expose feature_importances_. Try RandomForest/XGBoost/CatBoost.")

