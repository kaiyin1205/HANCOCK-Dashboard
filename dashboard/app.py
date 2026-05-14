import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Head & Neck Cancer Recurrence Risk",
    page_icon="🏥",
    layout="wide"
)

# ── Load models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_dir    = Path(__file__).parent / "models"
    model        = joblib.load(model_dir / "model.pkl")
    preprocessor = joblib.load(model_dir / "preprocessor.pkl")
    selector     = joblib.load(model_dir / "selector.pkl")
    feature_names= joblib.load(model_dir / "feature_names.pkl")
    clip_bounds  = joblib.load(model_dir / "clip_bounds.pkl")
    return model, preprocessor, selector, feature_names, clip_bounds

model, preprocessor, selector, feature_names, clip_bounds = load_models()


# ── Calculate inflammatory biomarkers ─────────────────────
def calculate_biomarkers(neutrophils, lymphocytes, platelets, monocytes):
    nlr = neutrophils / lymphocytes if lymphocytes > 0 else 0
    plr = platelets   / lymphocytes if lymphocytes > 0 else 0
    lmr = lymphocytes / monocytes   if monocytes   > 0 else 0
    sii = (platelets * neutrophils) / lymphocytes if lymphocytes > 0 else 0
    return nlr, plr, lmr, sii


# ── Add interaction features ───────────────────────────────
def add_interaction_features(df):
    df   = df.copy()
    cols = list(df.columns)

    def find_col(keywords):
        for kw in keywords:
            matches = [c for c in cols if kw.lower() in str(c).lower()]
            if matches:
                return matches[0]
        return None

    nlr_col = find_col(["NLR"])
    plr_col = find_col(["PLR"])
    lmr_col = find_col(["LMR"])
    pn_col  = find_col(["pN_stage"])
    lym_col = find_col(["positive_lymph"])
    pni_col = find_col(["perineural"])

    if nlr_col and pn_col:
        df["interact_NLR_x_pNstage"] = df[nlr_col] * df[pn_col].fillna(0)
    if plr_col and lym_col:
        df["interact_PLR_x_lymph"]   = df[plr_col] * df[lym_col].fillna(0)
    if lmr_col and pni_col:
        df["interact_LMR_x_PNI"]     = df[lmr_col] * df[pni_col].fillna(0)
    if nlr_col and plr_col:
        df["interact_NLR_div_PLR"]   = df[nlr_col] / (df[plr_col].replace(0, np.nan)).fillna(df[plr_col].median())
    if nlr_col and lmr_col:
        df["interact_NLR_x_LMR"]     = df[nlr_col] * df[lmr_col]
    return df


# ── Predict recurrence risk ────────────────────────────────
def predict_risk(input_data):
    df = pd.DataFrame([input_data])
    # Load feature names from training data to fill missing columns
    model_dir = Path(__file__).parent / "models"
    train_cols = joblib.load(model_dir / "train_columns.pkl")
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_cols]

    for col, (q_low, q_high) in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=q_low, upper=q_high)

    df         = add_interaction_features(df)
    X_proc     = preprocessor.transform(df)
    X_selected = selector.transform(X_proc)
    proba      = model.predict_proba(X_selected)[0][1]
    return proba, X_selected


# ── Get SHAP values ────────────────────────────────────────
def get_shap_values(X_selected):
    rf_model  = model.estimators_[0]
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_selected)
    if isinstance(shap_vals, list):
        return shap_vals[1][0], explainer.expected_value[1]
    if len(np.array(shap_vals).shape) == 3:
        return shap_vals[0, :, 1], explainer.expected_value[1]
    return shap_vals[0], explainer.expected_value


# ── Risk level classification ──────────────────────────────
def risk_level(proba):
    if proba < 0.3:
        return "Low Risk",    "🟢", "green"
    elif proba < 0.6:
        return "Medium Risk", "🟡", "orange"
    else:
        return "High Risk",   "🔴", "red"


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("🏥 HANCOCK Risk System")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Select Mode",
    ["👤 Patient Mode", "👨‍⚕️ Physician Mode"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This system is designed for post-operative recurrence risk assessment "
    "in head and neck cancer patients. For reference only — does not replace "
    "professional medical diagnosis."
)


# ══════════════════════════════════════════════════════════
# PATIENT MODE
# ══════════════════════════════════════════════════════════
if mode == "👤 Patient Mode":
    st.title("🏥 Head & Neck Cancer Recurrence Risk Assessment")
    st.markdown(
        "Please enter the values from your most recent blood test report. "
        "The system will automatically assess your recurrence risk."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Basic Information")
        age     = st.number_input("Age", min_value=18, max_value=100, value=60)
        sex     = st.selectbox("Sex", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never", "Current", "Former"])

    with col2:
        st.subheader("🩸 Blood Test Values")
        st.caption("Please find the following values from your blood test report")
        neutrophils = st.number_input("Neutrophils (×10³/μL)",  min_value=0.0,  value=4.0,   step=0.1)
        lymphocytes = st.number_input("Lymphocytes (×10³/μL)",  min_value=0.1,  value=2.0,   step=0.1)
        platelets   = st.number_input("Platelets (×10³/μL)",    min_value=0.0,  value=200.0, step=1.0)
        monocytes   = st.number_input("Monocytes (×10³/μL)",    min_value=0.1,  value=0.5,   step=0.1)

    nlr, plr, lmr, sii = calculate_biomarkers(neutrophils, lymphocytes, platelets, monocytes)

    st.markdown("---")
    st.subheader("📊 Automatically Calculated Inflammatory Markers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NLR", f"{nlr:.2f}", help="Neutrophil-to-Lymphocyte Ratio, normal < 3")
    c2.metric("PLR", f"{plr:.2f}", help="Platelet-to-Lymphocyte Ratio, normal < 150")
    c3.metric("LMR", f"{lmr:.2f}", help="Lymphocyte-to-Monocyte Ratio, normal > 4")
    c4.metric("SII", f"{sii:.2f}", help="Systemic Immune-Inflammation Index, normal < 600")

    st.markdown("---")

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        input_data = {
            "sex"                               : 1 if sex == "Male" else 0,
            "primarily_metastasis"              : 0,
            "smoking_status"                    : {"Never": 0, "Current": 1, "Former": 2}[smoking],
            "age_at_initial_diagnosis"          : age,
            "perinodal_invasion"                : 0,
            "lymphovascular_invasion_L"         : 0,
            "vascular_invasion_V"               : 0,
            "perineural_invasion_Pn"            : 0,
            "carcinoma_in_situ"                 : 0,
            "primary_tumor_site"                : 0,
            "grading"                           : 2,
            "hpv_association_p16"               : 0,
            "resection_status"                  : 0,
            "resection_status_carcinoma_in_situ": 0,
            "histologic_type"                   : 0,
            "number_of_positive_lymph_nodes"    : 0,
            "infiltration_depth_in_mm"          : 5.0,
            "pT_stage"                          : 2,
            "pN_stage"                          : 0,
            "NLR"                               : nlr,
            "PLR"                               : plr,
            "LMR"                               : lmr,
        }

        with st.spinner("Analyzing..."):
            try:
                proba, X_selected  = predict_risk(input_data)
                level, icon, color = risk_level(proba)

                st.markdown("---")
                st.subheader("📈 Results")

                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown(f"""
                    <div style='text-align:center; padding:30px;
                                border-radius:15px; background-color:#f0f0f0;'>
                        <h1 style='font-size:60px'>{icon}</h1>
                        <h2 style='color:{color}'>{level}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col_r2:
                    if level == "Low Risk":
                        st.success("✅ You are currently at low risk. Please continue regular follow-up visits every 6 months.")
                    elif level == "Medium Risk":
                        st.warning("⚠️ You are currently at medium risk. Follow-up visits every 3 months are recommended. Please inform your physician.")
                    else:
                        st.error("🚨 You are currently at high risk. Please consult your physician as soon as possible to arrange more frequent follow-up examinations.")

                    st.markdown("**Markers requiring attention:**")
                    if nlr > 3:
                        st.markdown(f"- 🔴 NLR ({nlr:.2f}) is elevated — may indicate stronger inflammatory response")
                    if plr > 150:
                        st.markdown(f"- 🔴 PLR ({plr:.2f}) is elevated — may indicate weaker immune status")
                    if lmr < 4:
                        st.markdown(f"- 🔴 LMR ({lmr:.2f}) is low — may indicate reduced immune function")
                    if sii > 600:
                        st.markdown(f"- 🔴 SII ({sii:.2f}) is elevated — systemic inflammation index is high")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")


# ══════════════════════════════════════════════════════════
# PHYSICIAN MODE
# ══════════════════════════════════════════════════════════
else:
    st.title("👨‍⚕️ Clinical Decision Support System")
    st.markdown("Please enter the complete patient clinical data for a detailed risk analysis report.")
    st.markdown("---")

    with st.form("doctor_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Basic Information")
            age        = st.number_input("Age",          min_value=18, max_value=100, value=60)
            sex        = st.selectbox("Sex",             ["Male", "Female"])
            smoking    = st.selectbox("Smoking Status",  ["Never", "Current", "Former"])
            metastasis = st.selectbox("Distant Metastasis", ["No", "Yes"])

        with col2:
            st.subheader("Blood Test Values")
            neutrophils = st.number_input("Neutrophils (×10³/μL)", min_value=0.0,  value=4.0,   step=0.1)
            lymphocytes = st.number_input("Lymphocytes (×10³/μL)", min_value=0.1,  value=2.0,   step=0.1)
            platelets   = st.number_input("Platelets (×10³/μL)",   min_value=0.0,  value=200.0, step=1.0)
            monocytes   = st.number_input("Monocytes (×10³/μL)",   min_value=0.1,  value=0.5,   step=0.1)

        with col3:
            st.subheader("Pathological Data")
            pt_stage     = st.selectbox("pT Stage",   [1, 2, 3, 4])
            pn_stage     = st.selectbox("pN Stage",   [0, 1, 2, 3])
            n_lymph      = st.number_input("Number of Positive Lymph Nodes", min_value=0, value=0)
            infiltration = st.number_input("Infiltration Depth (mm)",        min_value=0.0, value=5.0)
            perinodal    = st.selectbox("Perinodal Invasion",       ["No", "Yes"])
            lymphovasc   = st.selectbox("Lymphovascular Invasion",  ["No", "Yes"])
            vascular     = st.selectbox("Vascular Invasion",        ["No", "Yes"])
            perineural   = st.selectbox("Perineural Invasion",      ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Analyze", type="primary", use_container_width=True)

    if submitted:
        nlr, plr, lmr, sii = calculate_biomarkers(neutrophils, lymphocytes, platelets, monocytes)

        input_data = {
            "sex"                               : 1 if sex == "Male" else 0,
            "primarily_metastasis"              : 1 if metastasis == "Yes" else 0,
            "smoking_status"                    : {"Never": 0, "Current": 1, "Former": 2}[smoking],
            "age_at_initial_diagnosis"          : age,
            "perinodal_invasion"                : 1 if perinodal == "Yes" else 0,
            "lymphovascular_invasion_L"         : 1 if lymphovasc == "Yes" else 0,
            "vascular_invasion_V"               : 1 if vascular == "Yes" else 0,
            "perineural_invasion_Pn"            : 1 if perineural == "Yes" else 0,
            "carcinoma_in_situ"                 : 0,
            "primary_tumor_site"                : 0,
            "grading"                           : 2,
            "hpv_association_p16"               : 0,
            "resection_status"                  : 0,
            "resection_status_carcinoma_in_situ": 0,
            "histologic_type"                   : 0,
            "number_of_positive_lymph_nodes"    : n_lymph,
            "infiltration_depth_in_mm"          : infiltration,
            "pT_stage"                          : pt_stage,
            "pN_stage"                          : pn_stage,
            "NLR"                               : nlr,
            "PLR"                               : plr,
            "LMR"                               : lmr,
        }

        with st.spinner("Analyzing, please wait..."):
            try:
                proba, X_selected  = predict_risk(input_data)
                level, icon, color = risk_level(proba)

                st.markdown("---")
                st.subheader("📊 Analysis Results")

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("NLR", f"{nlr:.2f}")
                col_m2.metric("PLR", f"{plr:.2f}")
                col_m3.metric("LMR", f"{lmr:.2f}")
                col_m4.metric("SII", f"{sii:.2f}")

                st.markdown("---")

                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown(f"""
                    <div style='text-align:center; padding:30px;
                                border-radius:15px; background-color:#f0f0f0;'>
                        <h1 style='font-size:60px'>{icon}</h1>
                        <h2 style='color:{color}'>{level}</h2>
                        <h3>Recurrence Probability: {proba*100:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with col_r2:
                    st.markdown("**SHAP Risk Attribution Analysis**")
                    try:
                        shap_vals, base_val = get_shap_values(X_selected)
                        explanation = shap.Explanation(
                            values       = shap_vals,
                            base_values  = base_val,
                            data         = X_selected[0],
                            feature_names= feature_names
                        )
                        fig, ax = plt.subplots(figsize=(8, 6))
                        shap.plots.waterfall(explanation, show=False)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"SHAP plot could not be displayed: {e}")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")