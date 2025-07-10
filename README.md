
# 📄 Loan Approval Prediction App

A simple, user-friendly web application to predict whether a customer's **loan application** is likely to be **approved or rejected**, based on their financial and personal information.

---

## 📌 Project Overview

This project was built to help individuals or institutions assess the approval likelihood of loan applicants quickly and intuitively — using a machine learning model trained on historical loan data.

Whether you're a credit analyst, business owner, or a non-technical user, this app helps you make better decisions with **AI-powered prediction**.

---

## 🧠 Model & Dataset

- ✅ **Model Used**: XGBoost Classifier
- ⚙️ **Preprocessing**:
  - Label Encoding for categorical variables
  - Feature Scaling using StandardScaler
- 📊 **Key Features Used**:
  - Age, Gender, Education
  - Annual Income
  - Loan Amount, Interest Rate, Loan % of Income
  - Credit Score
  - Home Ownership, Previous Defaults, Loan Intent

---

## 🚀 Features

- 📝 **Single Prediction**: Fill out a form for one applicant and get instant prediction and approval probability.
- 📂 **Batch Prediction**: Upload a CSV file for multiple applicants and download the prediction results.
- 📖 **How to Use**: Step-by-step guide for each input.
- 💬 **App Review**: Submit suggestions and feedback directly within the app.

---

## 📖 How to Use

### 📝 Single Prediction

Fill out the form with:

| Field                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Age**                  | Applicant's age (18–100)                                                    |
| **Gender**               | `male` or `female`                                                          |
| **Education**            | Highest education level (e.g., Bachelor, Master)                            |
| **Home Ownership**       | `RENT`, `MORTGAGE`, or `OWN`                                                |
| **Previous Loan Default**| Whether the applicant has defaulted before (`Yes` or `No`)                  |
| **Annual Income**        | Total annual income (e.g., `50000`)                                         |
| **Loan Amount**          | Amount of loan requested (e.g., `10000`)                                    |
| **Loan Interest Rate**   | Annual interest rate in % (e.g., `15.0`)                                     |
| **Loan % of Income**     | Ratio of loan to income (e.g., `0.2` if 10,000/50,000)                       |
| **Credit Score**         | Score from `300–850`                                                        |
| **Loan Purpose**         | Purpose of loan: `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, etc.        |

After filling the form, click **🔍 Predict** to see the result.

---

### 📂 Batch Prediction

Prepare a CSV with the following columns:

```
person_age,person_gender,person_education,person_income,person_home_ownership,previous_loan_defaults_on_file,loan_amnt,loan_int_rate,loan_percent_income,credit_score,loan_intent
```

Upload it in the **Batch Prediction** page and download your results.

---

## 💻 Run Locally (Optional)

You can also run this app locally:

1. Clone the repo:
   ```bash
   git clone https://github.com/kurniadewy/loan-approval-app.git
   cd loan-approval-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

> ⚠️ Make sure `xgb_model.json`, `scaler.pkl`, and `label_encoders.pkl` are in the same directory.

---

## 📷 Screenshot / Demo

![App Preview](https://github.com/kurniadewy/loan_approval/blob/main/app.png?raw=true)
> 💡 [Click here to try the app live](https://app-loan-approval.streamlit.app/)

---

## 📩 Feedback / Contact

We'd love to hear your feedback!  
You can submit it via the **App Review** page or reach out:

- 📧 Email: [kurniadewyisnaini@gmail.com](mailto:kurniadewyisnaini@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/kurniadewy](https://www.linkedin.com/in/kurniadewy)

---

👩‍💻 *Built with ❤️ by kurniadewy – Powered by Streamlit & XGBoost*
