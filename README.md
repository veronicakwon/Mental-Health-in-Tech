# Mental Health in Tech: Predicting Treatment-Seeking Behavior

## 📌 Project Overview
This project explores the factors that influence whether employees in the tech industry seek mental health treatment. Using survey data, we performed Exploratory Data Analysis (EDA) and trained machine learning models to identify key predictors of treatment-seeking behavior. The goal is to provide data-driven insights into how personal background and workplace dynamics interact with mental health choices.

## 📊 Dataset
The dataset used in this analysis consists of roughly 1,250 survey responses detailing demographics, workplace mental health benefits, and personal histories. 
* *Note: The data is primarily based on a 2014 industry survey, which carries limitations regarding geographic distribution and evolving modern workplace culture.*

## 🚀 How to Run the Project
Follow these steps to run the Python scripts on your local machine.

### Prerequisites
Ensure you have Python installed and any required libraries. If you came from a Jupyter Notebook, you may need to install dependencies first:
```bash
pip install pandas matplotlib  # Add any other libraries your project uses
```

### Execution Steps

1. **Open your Terminal / Command Prompt**
   * **Windows:** Press `Win`, type `cmd`, and hit Enter.
   * **Mac/Linux:** Press `Cmd + Space`, type `Terminal`, and hit Enter.

2. **Navigate to the project folder**
   Use the `cd` command followed by your project folder path:
   ```bash
   cd /path/to/your/project/folder
   ```

3. **Run the scripts**
   Execute the files using Python. Run them in the specific order your project requires:

   * **To run the first file:**
     ```bash
     # Windows
     python 01_eda-Copy1.py
     
     # Mac / Linux
     python3 01_eda-Copy1.py
     ```

   * **To run the second file:**
     ```bash
     # Windows
     python 02_modeling-Copy1.py
     
     # Mac / Linux
     python3 02_modeling-Copy1.py
     ```


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
* **Family history of mental illness and seeking treatment** I created a categorical bar plot to count the number respondees who sought treatment and did not and compared these results between those who reported a family history of mental illness and does who did not. Within the group of respondees that did not report a family history of mental illness, more workers did not seek treatment. Meanwhile, the group that did report a family history of mental illness had more workers seek treatment.

* **Workplace policies and seeking treatment** I made three bar graphs that displayed the number of responses to each of the following questions: does your workplace provide benefits/care options/wellness program(s)? Then, I split each of these respondees into two groups: those who sought treatment, and those who did not. Those who answered yes to having all three policies were more likely to seek treatment.

<img width="612" height="458" alt="Screenshot 2026-06-16 at 10 05 15 PM" src="https://github.com/user-attachments/assets/a5fb6c9a-283e-4ee9-96f2-772b95f9638a" />

<img width="1335" height="364" alt="Screenshot 2026-06-16 at 10 05 47 PM" src="https://github.com/user-attachments/assets/aabe81fd-96a2-4e83-885d-138a91d98d04" />

### 3. Model findings + interpretations
* **The 'Work Interfere' Catalyst:** In both models, the `work_interfere` variables (Often, Sometimes, Rarely) were the single strongest predictors of seeking treatment. This strongly suggests that the primary trigger for individuals to seek professional mental health support is when their symptoms begin to visibly impact their daily work execution.
* **Algorithm Nuances:** The Random Forest heavily weighted `Age` and `family_history`. Because Random Forests excel at capturing non-linear relationships, it captured complex demographic nuances that the linear Logistic Regression model partially missed. 
* **The Corporate Takeaway:** Workplace policy features were surprisingly weak predictors in both models, indicating that individual health status, personal history, and immediate work interference drive treatment decisions much more than the existence of corporate wellness programs.

---

## ⚠️ Limitations & Next Steps
* **Data Constraints:** The dataset is self-reported, relatively small (~1,250 rows), and heavily skewed toward specific countries and company types from 2014. Correlation found in this study does not imply causation.
* **Future Work:** Next steps include hyperparameter tuning using cross-validation, trying more advanced ensemble models like Gradient Boosting (XGBoost), or developing a simple Streamlit web application to map out predictions interactively.
