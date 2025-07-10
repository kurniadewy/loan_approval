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
st.set_page_config(page_title="Loan Approval Application", layout="wide")


# === Sidebar Navigation ===
st.sidebar.image("loan.png", width=180)
st.sidebar.title("Loan Approval App")
page = st.sidebar.radio("📌 Navigation", 
                        ["🏠 About App", 
                         "📖 How to Use", 
                         "📝 Single Prediction", 
                         "📂 Batch Prediction", 
                         "💬 App Review"
                        ])

# === 1. About App ===
if page == "🏠 About App":
    st.markdown("<h2 style='color:#4B8BBE;'>📄 Loan Approval Prediction App</h2>", unsafe_allow_html=True)
    st.markdown("""
Welcome to the **Loan Approval Prediction App**! 🎯  
This tool helps estimate whether a customer's loan application is likely to be **approved** or **rejected** based on their personal and financial data.

---

### 🔍 What Can You Do?

- 📝 **Single Prediction** — Predict loan status for one customer  
- 📂 **Batch Prediction** — Upload CSV for multiple customers  
- 📖 **How to Use** — Learn what each field means and how to fill it  
- 💬 **App Review** — Share your feedback with us  

---

💡 This app is made for **everyone**, including non-technical users.

👩‍💻 *Developed with ❤️ by Kurnia Dewy*
""")

# === 2. How to Use ===
elif page == "📖 How to Use":
    st.markdown("<h2 style='color:#4B8BBE;'>📖 How to Use This App</h2>", unsafe_allow_html=True)

    st.markdown("### 📝 Single Prediction")
    st.markdown("""
Fill in the form with the following details:

| Field | Description |
|-------|-------------|
| **🎂 Age** | Age of applicant (18–100 years) |
| **👤 Gender** | Choose *male* or *female* |
| **🎓 Education** | Highest education level: High School, Bachelor, etc |
| **🏠 Home Ownership** | Options: `RENT`, `MORTGAGE`, or `OWN` |
| **❌ Previous Loan Default** | Has user failed to pay a loan before? Yes/No |
| **💵 Annual Income** | Total yearly income in IDR (e.g., 50000000) |
| **💰 Loan Amount** | Requested loan amount (e.g., 10000000) |
| **📈 Interest Rate (%)** | Loan's annual interest rate (e.g., 15%) |
| **📊 Loan % of Income** | Ratio: Loan ÷ Income. *e.g., 10000000 ÷ 50000000 = 0.2* |
| **📉 Credit Score** | Number from 300 (bad) to 850 (excellent) |
| **🎯 Loan Purpose** | Choose from: `EDUCATION`, `VENTURE`, `MEDICAL`, etc |

Click **🔍 Predict** to see the result.

---

### 📂 Batch Prediction
Upload a CSV file with these exact columns:

- person_age, person_gender, person_education, person_income  
- person_home_ownership, previous_loan_defaults_on_file  
- loan_amnt, loan_int_rate, loan_percent_income  
- credit_score, loan_intent

You’ll get a downloadable result with predictions.

---

### 💬 App Review
Share how helpful the app is or give ideas for improvement!
""")

# === 3. Single Prediction ===
elif page == "📝 Single Prediction":
    st.markdown("<h2 style='color:#4B8BBE;'>📝 Predict One Customer</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("#### 💼 Customer Details")
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Age", 18, 100, 30)
            gender = st.selectbox("👤 Gender", ["male", "female"])
            education = st.selectbox("🎓 Education", ["High School", "Bachelor", "Master", "Associate"])
            home = st.selectbox("🏠 Home Ownership", ["RENT", "MORTGAGE", "OWN"])
            default = st.selectbox("❌ Previous Loan Default", ["Yes", "No"])

        with col2:
            income = st.number_input("💵 Annual Income ($)", 1000, 1000000000, 50000000, step=1000000)
            loan_amt = st.number_input("💰 Loan Amount ($)", 1000, 100000000, 10000000, step=1000000)
            interest = st.slider("📈 Interest Rate (%)", 5.0, 30.0, 15.0)
            percent_income = st.slider("📊 Loan % of Income", 0.0, 1.0, round(loan_amt/income, 2) if income != 0 else 0.2)
            credit_score = st.slider("📉 Credit Score", 300, 850, 650)
            intent = st.selectbox("🎯 Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION"])

        st.markdown("""
        <div style="background-color:#e0f7fa;padding:15px;border-radius:10px;margin-top:10px">
        💡 <b>Tips:</b> If you are confused, try filling it in with example data: income 50 million, loan 10 million, then <b>Loan % Income</b> is <b>0.2</b>.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Predict"):
            input_data = pd.DataFrame([{
                'person_age': age,
                'person_gender': gender,
                'person_education': education,
                'person_income': income,
                'person_home_ownership': home,
                'loan_amnt': loan_amt,
                'loan_int_rate': interest,
                'loan_percent_income': percent_income,
                'credit_score': credit_score,
                'loan_intent': intent,
                'previous_loan_defaults_on_file': default
            }])

            for col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])

            feature_names = model.get_booster().feature_names
            input_scaled = scaler.transform(input_data[feature_names])
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            st.markdown("### 🎯 Prediction Result")
            if pred == 1:
                st.success("✅ The loan is likely to be **Approved**")
            else:
                st.error("❌ The loan is likely to be **Rejected**")
            st.metric("📊 Approval Probability", f"{prob * 100:.2f} %")

# === 4. Batch Prediction ===
elif page == "📂 Batch Prediction":
    st.markdown("<h2 style='color:#4B8BBE;'>📂 Batch Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        for col in label_encoders:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col])
        feature_names = model.get_booster().feature_names
        df_scaled = scaler.transform(df[feature_names])
        df['prediction'] = model.predict(df_scaled)
        df['approval_prob'] = model.predict_proba(df_scaled)[:, 1]

        st.markdown("### 🔎 Preview Results")
        filter_val = st.slider("📊 Filter by Approval Probability >", 0.0, 1.0, 0.5)
        st.dataframe(df[df['approval_prob'] > filter_val].head(10))
        st.download_button("⬇️ Download Results", data=df.to_csv(index=False), file_name="loan_predictions.csv")

# === 5. App Review Page ===
elif page == "💬 App Review":
    st.markdown("<h2 style='color:#4B8BBE;'>💬 We'd Love Your Review</h2>", unsafe_allow_html=True)
    name = st.text_input("👤 Your Name (Optional)")
    rating = st.slider("🌟 Rate this app", 1, 5, 4)
    comments = st.text_area("🗨️ Comments or Suggestions:")
    if st.button("📨 Submit Review"):
        st.success("✅ Thank you for your feedback!")
        st.markdown(f"**Name**: {name if name else 'Anonymous'}  \n**Rating**: {rating}/5  \n**Comment**: {comments}")
