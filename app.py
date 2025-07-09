import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# === Load model and preprocessing tools ===
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# === Streamlit Config ===
st.set_page_config(page_title="Loan Approval App", layout="wide")

# === Sidebar Navigation ===
page = st.sidebar.radio("📌 Navigation", [
    "🏠 About App",
    "📝 Single Prediction",
    "📂 Batch Prediction",
    "🗨️ App Review"
])

# === 1. About App ===
if page == "🏠 About App":
    st.title("📄 Loan Approval Prediction App")
    st.markdown("""
    Welcome!  
    This application helps predict whether a customer's loan will be **approved or rejected**, based on key financial and personal details.

    ### 🔍 Features:
    - Predict a single customer's loan status.
    - Upload a CSV file for batch prediction.
    - Share your review of this app to help improve it.

    ---
    Developed with ❤️ by Kurnia Dewy Isnaini
    """)

# === 2. Single Prediction ===
elif page == "📝 Single Prediction":
    st.title("📝 Predict One Customer")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["male", "female"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Associate"])
        home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])
        default = st.selectbox("Previous Loan Default", ["Yes", "No"])

    with col2:
        income = st.number_input("Annual Income", 1000, 1000000, 50000, step=1000)
        loan_amt = st.number_input("Loan Amount", 500, 50000, 10000, step=500)
        interest = st.slider("Loan Interest Rate (%)", 5.0, 30.0, 15.0)
        percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION"])

    if st.button("🔍 Predict"):
        input_data = pd.DataFrame([{
            'person_age': age,
            'person_gender': gender,
            'person_education': education,
            'person_income': income,
            'person_home_ownership': home,
            'loan_amnt': loan_amt,
            'loan_intent': intent,
            'loan_int_rate': interest,
            'loan_percent_income': percent_income,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': default
        }])

        for col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

        feature_names = model.get_booster().feature_names
        input_scaled = scaler.transform(input_data[feature_names])
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("🎯 Prediction Result")
        if pred == 1:
            st.success("✅ Loan is likely to be Approved.")
        else:
            st.error("❌ Loan is likely to be Rejected.")
        st.metric("Probability of Approval", f"{prob * 100:.2f}%")

# === 3. Batch Prediction ===
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV file with customer data to get multiple loan predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])

        feature_names = model.get_booster().feature_names
        df_scaled = scaler.transform(df[feature_names])
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        df['prediction'] = preds
        df['approval_prob'] = probs

        st.markdown("### 🎯 Results Preview")
        filter_val = st.slider("Show top results with approval probability above", 0.0, 1.0, 0.5)
        filtered_df = df[df['approval_prob'] >= filter_val]
        st.dataframe(filtered_df.head(10))

        st.download_button("⬇️ Download Full Results", data=df.to_csv(index=False), file_name="loan_predictions.csv")

# === 4. App Review Page ===
elif page == "🗨️ App Review":
    st.title("🗨️ We'd Love Your Review")

    name = st.text_input("Your Name (Optional)")
    rating = st.slider("How helpful is this app?", 1, 5, 4)
    comments = st.text_area("Tell us what you liked or suggest improvements:")

    if st.button("📨 Submit Review"):
        st.success("✅ Thank you for your review!")
        st.markdown(f"""
        **Name**: {name if name else "Anonymous"}  
        **Rating**: {rating} / 5  
        **Comment**: {comments}
        """)
