<div align="center">

# 📉 Customer Churn Prediction

### End-to-End Machine Learning Project with Streamlit Deployment

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

*Predict which telecom customers are likely to leave, so the business can act before they do.*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Key Insights](#-key-insights)
- [Model Performance](#-model-performance)
- [Streamlit App](#-streamlit-app)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Acknowledgments](#-acknowledgments)

---

##  Overview

This project builds a complete machine learning pipeline to predict customer churn for a telecom company. It covers data cleaning, exploratory data analysis, preprocessing, training and comparing three models, and deploying the best one as an interactive **Streamlit** web app where a user can enter customer details and get an instant churn prediction.

---

##  Problem Statement

Customer churn is a major issue for subscription-based businesses, and losing customers directly reduces revenue. Acquiring a new customer is usually more expensive than retaining an existing one.

**Goal:** build a model that identifies customers at high risk of leaving, so the company can take proactive retention measures such as targeted offers, better support, or contract upgrades.

---

##  Dataset

| Property | Details |
|----------|---------|
| **Source** | IBM Sample Data Sets: Telco Customer Churn |
| **Records** | 7,032 customers (after cleaning) |
| **Features** | 19 (after dropping `customerID`) |
| **Target** | `Churn` (Yes/No, encoded as 1/0) |

The features cover customer demographics, account information (tenure, contract, payment method, charges), and subscribed services (internet, security, tech support, streaming, etc.).

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Deployment** | Streamlit |
| **Model Persistence** | Joblib |

---

##  Project Workflow

**1. Data Cleaning**
- Dropped the `customerID` column (identifier, no predictive value)
- Converted `TotalCharges` to numeric
- Handled missing values (11 rows dropped)
- Encoded the target `Churn` as 0 and 1

**2. Exploratory Data Analysis**
- Analyzed the churn distribution
- Studied how key features relate to churn

**3. Preprocessing**
- One-Hot Encoding for categorical variables
- 80/20 train-test split with stratification to preserve class ratio
- Feature scaling with `StandardScaler`

**4. Model Building and Evaluation**
- Logistic Regression
- Random Forest
- XGBoost
- Compared using Accuracy, Precision, Recall, F1-Score and ROC-AUC

**5. Deployment**
- Saved the model, scaler and training columns with Joblib
- Built an interactive Streamlit web app

---

##  Exploratory Data Analysis

### Churn Distribution
![Churn Distribution](<img width="549" height="393" alt="churn_distribution png" src="https://github.com/user-attachments/assets/6c4a985a-f0b4-49fb-a448-12d312da85cd" />
)

### Contract Type vs Churn
![Contract vs Churn](<img width="704" height="470" alt="contract_vs_churn png" src="https://github.com/user-attachments/assets/2f2a8318-7b3d-462a-8cae-c5ce8a3faf44" />
)

### Internet Service vs Churn
![Internet Service vs Churn](<img width="704" height="470" alt="internet_service_vs_churn png" src="https://github.com/user-attachments/assets/26eebe97-0457-440e-bedc-5b3728820ccd" />
)

### Tenure vs Churn
![Tenure vs Churn](<img width="686" height="470" alt="tenure_vs_churn png" src="https://github.com/user-attachments/assets/96cbfec7-5a76-4a62-8a9d-e7102a3064c9" />
)

### Monthly Charges vs Churn
![Monthly Charges vs Churn](<img width="695" height="470" alt="monthly_charges_vs_churn png" src="https://github.com/user-attachments/assets/04761fe1-b3f1-4f99-9b0d-3cda3cc267d8" />
)

<details>
<summary><b>More visualizations</b></summary>

### Online Security vs Churn
![Online Security vs Churn](<img width="704" height="470" alt="online_security_vs_churn png" src="https://github.com/user-attachments/assets/20b0aed9-e099-4788-9d46-101605485cbe" />
)

### Tech Support vs Churn
![Tech Support vs Churn](<img width="704" height="470" alt="tech_support_vs_churn png" src="https://github.com/user-attachments/assets/2371b6a2-8912-49fd-8299-7d27e7851357" />
)

### Payment Method vs Churn
![Payment Method vs Churn](<img width="859" height="593" alt="payment_method_vs_churn png" src="https://github.com/user-attachments/assets/df3919fb-a012-49c2-9b86-28f35a08353d" />
)

</details>

---

##  Key Insights

- **Contract type matters most:** customers on **month-to-month** contracts have the highest churn rate.
- **Internet service:** **Fiber optic** users churn more than DSL users.
- **Tenure:** customers with **low tenure** are far more likely to leave, so the early months are the critical retention window.
- **Support services:** customers without **Online Security** and **Tech Support** show higher churn.
- **Pricing:** newer customers with **higher monthly charges** are at the greatest risk.

### Business Recommendations
- Offer incentives to move month-to-month customers onto longer contracts.
- Run onboarding and retention campaigns during the first few months.
- Bundle or promote security and tech support add-ons.
- Review pricing and service quality for fiber optic customers.

---

##  Model Performance

Metrics for the churn class are reported on the held-out 20% test set.

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
|-------|:--------:|:-----------------:|:--------------:|:----------------:|:-------:|
| **Logistic Regression** | **0.80** | **0.65** | **0.57** | **0.61** | **0.836** |
| Random Forest | 0.79 | 0.62 | 0.51 | 0.56 | 0.816 |
| XGBoost | 0.77 | 0.58 | 0.53 | 0.55 | 0.811 |

** Best model: Logistic Regression**, which scored highest on every metric and was selected for deployment.

> **Note:** The dataset is imbalanced (far fewer churners than non-churners), which limits recall on the churn class. Addressing this with SMOTE and threshold tuning is planned under [Future Improvements](#-future-improvements).

---

##  Streamlit App

The app lets a user enter customer details (contract, tenure, internet service, charges, add-on services, etc.) and returns:

- the predicted outcome (**will churn / will not churn**)
- the churn probability

<!-- Add a screenshot of your app here:
![Streamlit App](images/app_screenshot.png)
-->

 **Live demo:** *Coming soon (planned deployment on Streamlit Cloud)*

---

##  Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── training_columns.pkl
│
├── app/
│   └── app.py
│
├── images/
│
├── README.md
└── requirements.txt
```

---

##  How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit app**
```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

To reproduce the analysis and training, open `notebooks/Customer_Churn_Prediction.ipynb` in Jupyter.

---

##  Future Improvements

- [ ] Handle class imbalance using **SMOTE**
- [ ] **Hyperparameter tuning** (GridSearchCV / RandomizedSearchCV)
- [ ] Optimize the decision threshold to improve churn recall
- [ ] Add **SHAP** for model explainability
- [ ] Deploy on **Streamlit Cloud**

---

##  Author

**Madhu Shankar Kumar**
AI & Data Science Student

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/madhushankarkr-cmd)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/madhu-shankar-kumar-74563537b/)

---

##  Acknowledgments

- Dataset: [IBM Sample Data Sets, Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

<div align="center">

 If you found this project useful, consider giving it a star!

</div>
