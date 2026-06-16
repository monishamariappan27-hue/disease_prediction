"""
Disease Prediction Dashboard
Run with: streamlit run app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Disease Prediction Dashboard", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
MODELS = BASE / "models"
ASSETS = BASE / "assets"

# -------------------------------------------------------------------
# Load artifacts (cached so the app doesn't reload on every interaction)
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS / "logistic_regression_model.pkl")
    scaler = joblib.load(MODELS / "scaler.pkl")
    label_encoder = joblib.load(MODELS / "label_encoder.pkl")
    feature_names = joblib.load(MODELS / "feature_names.pkl")
    with open(MODELS / "metrics.json") as f:
        metrics = json.load(f)
    with open(MODELS / "feature_stats.json") as f:
        feature_stats = json.load(f)
    return model, scaler, label_encoder, feature_names, metrics, feature_stats

required_files = [
    "logistic_regression_model.pkl", "scaler.pkl", "label_encoder.pkl",
    "feature_names.pkl", "metrics.json", "feature_stats.json",
]
missing = [f for f in required_files if not (MODELS / f).exists()]
if missing:
    st.error(
        f"Missing model file(s) in `{MODELS}`: {', '.join(missing)}.\n\n"
        "The `models/` folder must sit in the **same directory** as `app.py`. "
        "Either re-extract the full project zip so `models/` and `assets/` are "
        "siblings of `app.py`, or run `python train_pipeline.py` from this folder "
        "to regenerate them."
    )
    st.stop()

try:
    model, scaler, label_encoder, feature_names, metrics, feature_stats = load_artifacts()
except Exception as e:
    st.error(
        f"Failed to load model files from `{MODELS}`: **{type(e).__name__}: {e}**\n\n"
        "This usually means a package version mismatch (most often **NumPy**) between "
        "the environment that created these `.pkl` files and the one running this app. "
        "Fix by either:\n\n"
        "1. Installing the exact pinned versions: `pip install -r requirements.txt`, or\n"
        "2. Regenerating the model files with your current packages: run "
        "`python train_pipeline.py` from this folder, then restart the app."
    )
    st.stop()

# Group features for a friendlier input layout
FEATURE_GROUPS = {
    "Blood Count": ["Hemoglobin", "Platelets", "White Blood Cells", "Red Blood Cells",
                     "Hematocrit", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
                     "Mean Corpuscular Hemoglobin Concentration"],
    "Metabolic Panel": ["Glucose", "Insulin", "HbA1c", "BMI"],
    "Lipid Panel": ["Cholesterol", "Triglycerides", "LDL Cholesterol", "HDL Cholesterol"],
    "Organ Function": ["ALT", "AST", "Creatinine", "Troponin", "C-reactive Protein"],
    "Vitals": ["Systolic Blood Pressure", "Diastolic Blood Pressure", "Heart Rate"],
}

DISEASE_COLORS = {
    "Healthy": "#2ecc71",
    "Diabetes": "#e67e22",
    "Anemia": "#e74c3c",
    "Thalasse": "#9b59b6",
    "Thromboc": "#3498db",
}

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("🩺 Disease Prediction")
st.sidebar.markdown(
    "Multinomial **Logistic Regression** model trained on a balanced blood-sample "
    "dataset to classify patients into 5 categories: Healthy, Diabetes, Anemia, "
    "Thalassemia, Thrombocytopenia."
)
st.sidebar.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
st.sidebar.metric("Training samples", metrics["n_train"])
st.sidebar.metric("Test samples", metrics["n_test"])
st.sidebar.caption(
    "Note: feature values are normalized 0–1 (matching how the source dataset "
    "was prepared), not raw lab units."
)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_predict, tab_data, tab_model = st.tabs(["🔮 Predict", "📊 Data Insights", "📈 Model Performance"])

# =====================================================================
# TAB 1: PREDICT
# =====================================================================
with tab_predict:
    st.header("Enter Patient Blood Panel")
    st.caption("Adjust the sliders to match a patient's normalized lab values, then click Predict.")

    inputs = {}
    cols = st.columns(len(FEATURE_GROUPS))
    for col, (group_name, feats) in zip(cols, FEATURE_GROUPS.items()):
        with col:
            st.subheader(group_name)
            for feat in feats:
                stats = feature_stats[feat]
                inputs[feat] = st.slider(
                    feat,
                    min_value=float(round(stats["min"], 3)),
                    max_value=float(round(stats["max"], 3)),
                    value=float(round(stats["mean"], 3)),
                    step=0.01,
                    key=f"slider_{feat}",
                )

    st.divider()
    predict_clicked = st.button("🔍 Predict Disease", type="primary", width="content")

    if predict_clicked:
        X = np.array([[inputs[f] for f in feature_names]])
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        result_col, chart_col = st.columns([1, 2])

        with result_col:
            color = DISEASE_COLORS.get(pred_label, "#34495e")
            st.markdown(
                f"""
                <div style="padding: 24px; border-radius: 12px; background-color: {color}22;
                            border: 2px solid {color}; text-align: center;">
                    <div style="font-size: 14px; color: #555;">Predicted Condition</div>
                    <div style="font-size: 32px; font-weight: 700; color: {color};">{pred_label}</div>
                    <div style="font-size: 14px; color: #555; margin-top: 8px;">
                        Confidence: {pred_proba[pred_idx]:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chart_col:
            classes = label_encoder.classes_
            fig = go.Figure(
                go.Bar(
                    x=pred_proba,
                    y=classes,
                    orientation="h",
                    marker_color=[DISEASE_COLORS.get(c, "#34495e") for c in classes],
                    text=[f"{p:.1%}" for p in pred_proba],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability",
                xaxis_range=[0, 1],
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, width='stretch')

        st.info(
            "This prediction is generated by a statistical model for demonstration/educational "
            "purposes and is not a medical diagnosis. Always consult a qualified healthcare "
            "professional for clinical decisions.",
            icon="ℹ️",
        )

# =====================================================================
# TAB 2: DATA INSIGHTS
# =====================================================================
with tab_data:
    st.header("Dataset Overview")
    n_total = metrics["n_train"] + metrics["n_test"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", n_total)
    c2.metric("Features", len(feature_names))
    c3.metric("Disease Classes", len(metrics["classes"]))

    st.subheader("Class Distribution")
    st.image(str(ASSETS / "class_distribution.png"), width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Feature Correlation")
        st.image(str(ASSETS / "correlation_heatmap.png"), width='stretch')
    with col_b:
        st.subheader("Key Feature Distributions by Disease")
        st.image(str(ASSETS / "feature_distributions.png"), width='stretch')

# =====================================================================
# TAB 3: MODEL PERFORMANCE
# =====================================================================
with tab_model:
    st.header("Model Performance")
    st.metric("Overall Test Accuracy", f"{metrics['accuracy']:.2%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confusion Matrix")
        st.image(str(ASSETS / "confusion_matrix.png"), width='stretch')
    with col_b:
        st.subheader("ROC Curves (One-vs-Rest)")
        st.image(str(ASSETS / "roc_curves.png"), width='stretch')

    st.subheader("Feature Importance")
    st.image(str(ASSETS / "feature_importance.png"), width='stretch')

    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    report_df = report_df.drop(index=[i for i in report_df.index if i not in metrics["classes"]], errors="ignore")
    st.dataframe(report_df.style.format("{:.3f}"), width='stretch')
