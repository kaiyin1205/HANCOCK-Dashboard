import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import base64
import io
import json
from PIL import Image
import fitz
import anthropic

sys.path.append(str(Path(__file__).parents[1]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Head & Neck Cancer Recurrence Risk",
    page_icon="🏥",
    layout="wide"
)

# ── AI suggestion for patient mode ────────────────────────
def get_ai_suggestion_patient(nlr, plr, lmr, sii, risk_level_str):
    """Generate AI suggestion for patient mode"""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    prompt = (
        "You are a medical AI assistant helping head and neck cancer patients "
        "understand their post-operative recurrence risk assessment results.\n\n"
        f"Patient's inflammatory markers:\n"
        f"- NLR (Neutrophil-to-Lymphocyte Ratio): {round(nlr, 2)} (normal < 3)\n"
        f"- PLR (Platelet-to-Lymphocyte Ratio): {round(plr, 2)} (normal < 150)\n"
        f"- LMR (Lymphocyte-to-Monocyte Ratio): {round(lmr, 2)} (normal > 4)\n"
        f"- SII (Systemic Immune-Inflammation Index): {round(sii, 2)} (normal < 600)\n\n"
        f"Risk assessment result: {risk_level_str}\n\n"
        "Please provide a brief, friendly, and easy-to-understand suggestion for the patient (3-4 sentences). Focus on:\n"
        "1. Whether they should visit their doctor soon or can wait for regular follow-up\n"
        "2. Simple lifestyle advice if applicable\n"
        "3. Reassurance and encouragement\n\n"
        "Important: Do NOT mention specific medical diagnoses or treatments. Keep it simple and non-alarming. Write in English."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ── AI suggestion for physician mode ──────────────────────
def get_ai_suggestion_physician(nlr, plr, lmr, sii, risk_level_str, proba,
                                pt_stage, pn_stage, n_lymph, metastasis,
                                perinodal, lymphovasc, vascular, perineural):
    """Generate AI suggestion for physician mode with literature references"""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    prompt = (
        "You are a clinical decision support AI for head and neck squamous cell carcinoma (HNSCC).\n\n"
        "Patient clinical data:\n"
        f"- Recurrence probability: {round(proba*100, 1)} percent\n"
        f"- Risk level: {risk_level_str}\n"
        f"- NLR: {round(nlr, 2)} (normal < 3)\n"
        f"- PLR: {round(plr, 2)} (normal < 150)\n"
        f"- LMR: {round(lmr, 2)} (normal > 4)\n"
        f"- SII: {round(sii, 2)} (normal < 600)\n"
        f"- pT Stage: {pt_stage}\n"
        f"- pN Stage: {pn_stage}\n"
        f"- Positive lymph nodes: {n_lymph}\n"
        f"- Distant metastasis: {metastasis}\n"
        f"- Perinodal invasion: {perinodal}\n"
        f"- Lymphovascular invasion: {lymphovasc}\n"
        f"- Vascular invasion: {vascular}\n"
        f"- Perineural invasion: {perineural}\n\n"
        "Please provide a clinical recommendation (4-5 sentences) including:\n"
        "1. Which abnormal values are driving the high/low risk and why\n"
        "2. Suggested clinical follow-up actions or diagnostic workup\n"
        "3. Reference to 1-2 relevant published studies supporting your recommendation (include author, journal, year)\n\n"
        "Write in a professional clinical tone in English."
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ── OCR: Extract blood values from image ──────────────────
def extract_blood_values_from_image(image_bytes, media_type="image/jpeg"):
    """Use Claude Vision to extract blood test values from health check report"""
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Please analyze this health check report image and extract the following blood test values:\n\n"
        "1. Neutrophils (in x10^3/uL or similar unit)\n"
        "2. Lymphocytes (in x10^3/uL or similar unit)\n"
        "3. Platelets (in x10^3/uL or similar unit)\n"
        "4. Monocytes (in x10^3/uL or similar unit)\n\n"
        "The report may be in Chinese or English. Please:\n"
        "- Look for these values in the complete blood count (CBC) section\n"
        "- Convert units if necessary (e.g., if in x10^9/L, divide by 1 to get x10^3/uL)\n"
        "- If a value is not found, return null for that field\n\n"
        "Return ONLY a JSON object in this exact format, nothing else:\n"
        "{\n"
        '  "neutrophils": <number or null>,\n'
        '  "lymphocytes": <number or null>,\n'
        '  "platelets": <number or null>,\n'
        '  "monocytes": <number or null>,\n'
        '  "notes": "<any important notes about the extraction>"\n'
        "}"
    )
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt}
            ],
        }],
    )
    response_text = message.content[0].text.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    return json.loads(response_text)


# ── PDF to image ───────────────────────────────────────────
def pdf_to_image_bytes(pdf_bytes):
    """Convert first page of PDF to image bytes"""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = pdf_document[0]
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    pdf_document.close()
    return img_bytes


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
    model_dir  = Path(__file__).parent / "models"
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


# ── Biomarker-based risk for patient mode ─────────────────
def biomarker_risk_level(nlr, plr, lmr, sii):
    score = 0

    # NLR: normal < 3.0 (literature threshold for HNSCC)
    if nlr > 5.0:   score += 2
    elif nlr > 3.0: score += 1

    # PLR: normal < 150 (literature threshold for HNSCC)
    if plr > 300:   score += 2
    elif plr > 150: score += 1

    # LMR: normal > 4.0 (literature threshold for HNSCC)
    if lmr < 2.0:   score += 2
    elif lmr < 4.0: score += 1

    # SII: normal < 600 (literature threshold for HNSCC)
    if sii > 1200:  score += 2
    elif sii > 600: score += 1

    if score >= 5:
        return "High Inflammation Risk",     "🔴", "red"
    elif score >= 2:
        return "Moderate Inflammation Risk", "🟡", "orange"
    else:
        return "Low Inflammation Risk",      "🟢", "green"


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

        # OCR upload section
        st.caption("Option 1: Upload your health check report for automatic extraction")
        uploaded_file = st.file_uploader(
            "Upload health check report (PDF, JPG, PNG)",
            type=["pdf", "jpg", "jpeg", "png"],
            key="patient_report"
        )

        if uploaded_file is not None:
            with st.spinner("Reading your health check report..."):
                try:
                    file_bytes = uploaded_file.read()
                    if uploaded_file.type == "application/pdf":
                        image_bytes = pdf_to_image_bytes(file_bytes)
                        media_type  = "image/jpeg"
                    else:
                        image_bytes = file_bytes
                        media_type  = uploaded_file.type

                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="Uploaded Report", use_container_width=True)

                    extracted = extract_blood_values_from_image(image_bytes, media_type)

                    if extracted:
                        st.success("Values extracted successfully! Please verify before analyzing.")
                        if extracted.get("notes"):
                            st.info(f"Note: {extracted['notes']}")

                        # Show extracted values for user confirmation
                        st.markdown("**Please confirm the extracted values:**")
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            confirmed_neutrophils = st.number_input(
                                "Neutrophils (x10^3/uL)",
                                min_value=0.0,
                                value=float(extracted.get("neutrophils") or 4.0),
                                step=0.1,
                                key="confirm_neutrophils"
                            )
                            confirmed_lymphocytes = st.number_input(
                                "Lymphocytes (x10^3/uL)",
                                min_value=0.1,
                                value=float(extracted.get("lymphocytes") or 2.0),
                                step=0.1,
                                key="confirm_lymphocytes"
                            )
                        with col_e2:
                            confirmed_platelets = st.number_input(
                                "Platelets (x10^3/uL)",
                                min_value=0.0,
                                value=float(extracted.get("platelets") or 200.0),
                                step=1.0,
                                key="confirm_platelets"
                            )
                            confirmed_monocytes = st.number_input(
                                "Monocytes (x10^3/uL)",
                                min_value=0.1,
                                value=float(extracted.get("monocytes") or 0.5),
                                step=0.1,
                                key="confirm_monocytes"
                            )

                        if st.button("Confirm and Use These Values", type="primary"):
                            st.session_state["neutrophils"] = confirmed_neutrophils
                            st.session_state["lymphocytes"] = confirmed_lymphocytes
                            st.session_state["platelets"]   = confirmed_platelets
                            st.session_state["monocytes"]   = confirmed_monocytes
                            st.rerun()

                except Exception as e:
                    st.error(f"Could not extract values: {e}")

        st.caption("Option 2: Enter values manually")
        neutrophils = st.number_input(
            "Neutrophils (x10^3/uL)",
            min_value=0.0,
            value=float(st.session_state.get("neutrophils", 4.0)),
            step=0.1
        )
        lymphocytes = st.number_input(
            "Lymphocytes (x10^3/uL)",
            min_value=0.1,
            value=float(st.session_state.get("lymphocytes", 2.0)),
            step=0.1
        )
        platelets = st.number_input(
            "Platelets (x10^3/uL)",
            min_value=0.0,
            value=float(st.session_state.get("platelets", 200.0)),
            step=1.0
        )
        monocytes = st.number_input(
            "Monocytes (x10^3/uL)",
            min_value=0.1,
            value=float(st.session_state.get("monocytes", 0.5)),
            step=0.1
        )

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
                level, icon, color = biomarker_risk_level(nlr, plr, lmr, sii)

                st.markdown("---")
                st.subheader("📈 Inflammatory Status Assessment")
                st.caption("Note: This assessment is based on inflammatory markers only. "
                           "It does not replace a full clinical evaluation by your physician.")

                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    st.markdown(
                        f"<div style='text-align:center; padding:30px; border-radius:15px; background-color:#f0f0f0;'>"
                        f"<h1 style='font-size:60px'>{icon}</h1>"
                        f"<h2 style='color:{color}'>{level}</h2>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                with col_r2:
                    if level == "Low Inflammation Risk":
                        st.success(
                            "Your inflammatory markers are within normal ranges. "
                            "Please continue regular follow-up visits every 6 months "
                            "as scheduled by your physician."
                        )
                    elif level == "Moderate Inflammation Risk":
                        st.warning(
                            "Some of your inflammatory markers are above normal ranges, "
                            "suggesting moderate immune stress. We recommend informing "
                            "your physician at your next visit for a comprehensive evaluation."
                        )
                    else:
                        st.error(
                            "Multiple inflammatory markers are significantly elevated, "
                            "suggesting high immune stress. We recommend scheduling a "
                            "visit with your physician soon for a comprehensive evaluation, "
                            "including pathological assessment."
                        )

                    st.markdown("**Markers requiring attention:**")
                    if nlr > 3:
                        st.markdown(f"- NLR ({nlr:.2f}) is elevated — may indicate stronger inflammatory response")
                    if plr > 150:
                        st.markdown(f"- PLR ({plr:.2f}) is elevated — may indicate weaker immune status")
                    if lmr < 4:
                        st.markdown(f"- LMR ({lmr:.2f}) is low — may indicate reduced immune function")
                    if sii > 600:
                        st.markdown(f"- SII ({sii:.2f}) is elevated — systemic inflammation index is high")

                    st.markdown("---")
                    st.markdown("**🤖 AI Suggestion**")
                    with st.spinner("Generating AI suggestion..."):
                        try:
                            ai_suggestion = get_ai_suggestion_patient(nlr, plr, lmr, sii, level)
                            st.info(ai_suggestion)
                        except Exception as e:
                            st.warning(f"AI suggestion unavailable: {e}")

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
            neutrophils = st.number_input("Neutrophils (x10^3/uL)", min_value=0.0,  value=4.0,   step=0.1)
            lymphocytes = st.number_input("Lymphocytes (x10^3/uL)", min_value=0.1,  value=2.0,   step=0.1)
            platelets   = st.number_input("Platelets (x10^3/uL)",   min_value=0.0,  value=200.0, step=1.0)
            monocytes   = st.number_input("Monocytes (x10^3/uL)",   min_value=0.1,  value=0.5,   step=0.1)

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
                    st.markdown(
                        f"<div style='text-align:center; padding:30px; border-radius:15px; background-color:#f0f0f0;'>"
                        f"<h1 style='font-size:60px'>{icon}</h1>"
                        f"<h2 style='color:{color}'>{level}</h2>"
                        f"<h3>Recurrence Probability: {proba*100:.1f}%</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

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

                    st.markdown("---")
                    st.markdown("**🤖 AI Clinical Recommendation**")
                    with st.spinner("Generating clinical recommendation..."):
                        try:
                            ai_rec = get_ai_suggestion_physician(
                                nlr, plr, lmr, sii, level, proba,
                                pt_stage, pn_stage, n_lymph, metastasis,
                                perinodal, lymphovasc, vascular, perineural
                            )
                            st.info(ai_rec)
                        except Exception as e:
                            st.warning(f"AI recommendation unavailable: {e}")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")