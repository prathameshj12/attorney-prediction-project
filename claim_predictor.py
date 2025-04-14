import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Page Config ===
st.set_page_config(page_title="Attorney Predictor", layout="wide")

# === Custom CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .title-card {
        background-color: #d4f8d4;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    div.stButton > button {
        width: 100%;
        padding: 0.6rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# === Title ===
st.markdown("""
<div class="title-card">
    <h2>Motor Insurance - Attorney Predictor</h2>
    <p>AI-powered prediction tool to assist legal evaluation in insurance claims</p>
</div>
""", unsafe_allow_html=True)

# === Input Form ===
with st.form("claim_form"):
    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.markdown("### 👨‍💼 Claimant Information")
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=25)

        st.markdown("### 💥🚗 Accident Details")
        seatbelt = st.selectbox("Seatbelt Worn?", ["Yes", "No"])
        severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe"])
        driving_record = st.selectbox("Driving Record", ["Clean", "Minor Offenses", "Major Offenses"])
        loss = st.number_input("Estimated Loss", min_value=0)

    with col2:
        st.markdown("### 📄 Claim & Policy")
        insured = st.selectbox("Claimant Insured?", ["Yes", "No"])
        claim_amount = st.number_input("Claim Amount Requested", min_value=0, value=0)
        approved = st.selectbox("Claim Approved?", ["Yes", "No"])
        settlement = st.number_input("Settlement Amount", min_value=0, value=0)
        policy_type = st.selectbox("Policy Type", ["Comprehensive", "Third-Party"])

        # === Buttons ===
        st.markdown("###")
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            predict_btn = st.form_submit_button("🔍 Predict Attorney Need")
        with btn_col2:
            clear_btn = st.form_submit_button("🧹 Clear")

    with col3:
        st.markdown("### 🗂️ Prediction Summary")
        prediction_area = st.empty()

# === Preprocessing ===
def preprocess_input(df):
    df["Policy_Type"] = df["Policy_Type"].map({"Comprehensive": 0, "Third-Party": 1})
    df["Driving_Record"] = df["Driving_Record"].map({"Clean": 0, "Minor Offenses": 1, "Major Offenses": 2})
    df["Accident_Severity"] = df["Accident_Severity"].map({"Minor": 1, "Moderate": 2, "Severe": 3})
    df["SEATBELT"] = df["SEATBELT"].map({"Yes": 1, "No": 0})
    df["CLMSEX"] = 1 if df["CLMSEX"][0] == "Male" else 0
    df["CLMINSUR"] = 1 if df["CLMINSUR"][0] == "Yes" else 0
    df["Claim_Approval_Status"] = 1 if df["Claim_Approval_Status"][0] == "Yes" else 0

    df["Claim_Diff"] = df["Claim_Amount_Requested"] - df["Settlement_Amount"]
    df["Claim_Diff_Perc"] = df["Claim_Diff"] / df["Claim_Amount_Requested"].replace(0, 1)
    df["Underpaid_Claim_Flag"] = (df["Settlement_Amount"] < 0.7 * df["Claim_Amount_Requested"]).astype(int)
    df["Is_High_Settlement"] = (df["Settlement_Amount"] > 20000).astype(int)
    df["Settlement_vs_Claim"] = df["Settlement_Amount"] / df["Claim_Amount_Requested"].replace(0, 1)
    df["Is_Young_Claimant"] = (df["CLMAGE"] < 25).astype(int)
    df["Is_ThirdParty_and_Denied"] = ((df["Policy_Type"] == 1) & (df["Claim_Approval_Status"] == 0)).astype(int)
    df["Is_High_Loss"] = (df["LOSS"] > 20000).astype(int)

    final_cols = [
        'CLMSEX', 'Claim_Diff', 'Claim_Diff_Perc', 'Underpaid_Claim_Flag',
        'Is_High_Settlement', 'Settlement_vs_Claim', 'SEATBELT',
        'Is_Young_Claimant', 'Is_ThirdParty_and_Denied', 'Is_High_Loss'
    ]
    return df[final_cols]

# === Prediction Logic ===
if predict_btn:
    try:
        input_data = {
            "CLMSEX": sex,
            "CLMINSUR": insured,
            "SEATBELT": seatbelt,
            "CLMAGE": age,
            "LOSS": loss,
            "Claim_Amount_Requested": claim_amount,
            "Claim_Approval_Status": approved,
            "Settlement_Amount": settlement,
            "Policy_Type": policy_type,
            "Driving_Record": driving_record,
            "Accident_Severity": severity
        }

        df_input = pd.DataFrame([input_data])
        processed_input = preprocess_input(df_input)
        model = joblib.load("best_model.pkl")
        X_array = processed_input.to_numpy()

        prediction = model.predict(X_array)[0]
        confidence = model.predict_proba(X_array)[0][prediction]

        with col3:
            if prediction == 1:
                st.success("✅ This claimant **may require legal representation**.")
            else:
                st.info("ℹ️ This claimant **may not require an attorney.**")

            # Confidence bar
            st.markdown(f"**🔵 Confidence:** {confidence:.2%}")
            st.progress(confidence)

            # Key Conditions
            st.markdown("**🧩 Key Conditions:**")
            if processed_input["Underpaid_Claim_Flag"].values[0] == 1:
                st.markdown("- 💲 Claim may have been underpaid.")
            if processed_input["Is_ThirdParty_and_Denied"].values[0] == 1:
                st.markdown("- 📄 Third-party claim possibly denied.")
            if processed_input["Is_High_Loss"].values[0] == 1:
                st.markdown("- 🚨 High estimated loss noted.")

            # Insight
            if prediction == 1 and confidence > 0.6:
                st.markdown("**💡 Insight:** Strong indicators suggest legal counsel may be needed.")
            elif prediction == 1:
                st.markdown("**💡 Insight:** Possibility of legal support exists, but with lower certainty.")
            else:
                st.markdown("**💡 Insight:** Legal representation is likely unnecessary based on claim data.")

    except Exception as e:
        with col3:
            st.error(f"Prediction failed: {e}")

# === Clear Button Action ===
if clear_btn:
    st.rerun()
