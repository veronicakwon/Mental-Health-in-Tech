# Mental Health in Tech: Predicting Treatment-Seeking Behavior

## 📌 Project Overview
This project explores the factors that influence whether employees in the tech industry seek mental health treatment. Using survey data, we performed Exploratory Data Analysis (EDA) and trained machine learning models to identify key predictors of treatment-seeking behavior. The goal is to provide data-driven insights into how personal background and workplace dynamics interact with mental health choices.

## 📊 Dataset
The dataset used in this analysis consists of roughly 1,250 survey responses detailing demographics, workplace mental health benefits, and personal histories. 
* *Note: The data is primarily based on a 2014 industry survey, which carries limitations regarding geographic distribution and evolving modern workplace culture.*

## 🚀 How to Run the Project
You can view or run the interactive Jupyter Notebooks directly via Google Colab by clicking the badges below:

* **Step 1: Exploratory Data Analysis** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](INSERT_YOUR_COLAB_LINK_FOR_EDA_HERE)
* **Step 2: Predictive Modeling & Evaluation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](INSERT_YOUR_COLAB_LINK_FOR_MODELING_HERE)

Alternatively, clone this repository and run `01_eda.ipynb` followed by `02_modeling.ipynb` locally.

---

## 🔍 Key Findings

### 1. Exploratory Data Analysis (EDA)
* **Family History Matters:** Employees with a known family history of mental illness showed a significantly stronger association with seeking treatment.
* **Corporate Support vs. Reality:** While the presence of workplace wellness programs and mental health benefits showed a loose association during initial data plotting, they were overridden by personal indicators when modeled.

### 2. Model Performance
We trained a **Logistic Regression** model and a **Random Forest Classifier** to predict who seeks treatment. 

| Model | Accuracy | False Positives | False Negatives |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | ~83% | 27 | 16 |
| **Random Forest** | ~82% | 34 | 12 |

* **Trade-off Analysis:** While Logistic Regression achieved slightly higher overall accuracy, the Random Forest model was better at minimizing False Negatives (missing individuals who actually sought treatment). Depending on the real-world application (e.g., ensuring vulnerable employees receive proactive outreach), a lower false-negative rate might be preferred.

### 3. Summary of EDA
* **family_history:** To observe whether family_history has an association with seeking treatment or not, 

### 3. Model findings + interpretations
* **The 'Work Interfere' Catalyst:** In both models, the `work_interfere` variables (Often, Sometimes, Rarely) were the single strongest predictors of seeking treatment. This strongly suggests that the primary trigger for individuals to seek professional mental health support is when their symptoms begin to visibly impact their daily work execution.
* **Algorithm Nuances:** The Random Forest heavily weighted `Age` and `family_history`. Because Random Forests excel at capturing non-linear relationships, it captured complex demographic nuances that the linear Logistic Regression model partially missed. 
* **The Corporate Takeaway:** Workplace policy features were surprisingly weak predictors in both models, indicating that individual health status, personal history, and immediate work interference drive treatment decisions much more than the existence of corporate wellness programs.

---

## ⚠️ Limitations & Next Steps
* **Data Constraints:** The dataset is self-reported, relatively small (~1,250 rows), and heavily skewed toward specific countries and company types from 2014. Correlation found in this study does not imply causation.
* **Future Work:** Next steps include hyperparameter tuning using cross-validation, trying more advanced ensemble models like Gradient Boosting (XGBoost), or developing a simple Streamlit web application to map out predictions interactively.
