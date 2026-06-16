#!/usr/bin/env python
# coding: utf-8

# # Mental Health in Tech: Predicting Treatment-Seeking
# 
# **This is notebook 2 of 2.** We'll use the cleaned dataset from `01_eda.ipynb` (`mental_health_cleaned.csv`) to build a model that predicts whether someone sought mental health treatment, based on workplace and personal factors.
# 
# **Goal:** practice the standard supervised learning workflow — encode features, split data, train models, evaluate them, and interpret what they learned.
# 
# ## How to use this notebook
# Same format as before: explanation → `# TODO` → collapsed hint. Try writing the code yourself first.
# 
# ## Setup
# Upload `mental_health_cleaned.csv` (the file you saved at the end of notebook 1) to your Colab session / project folder.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('mental_health_cleaned.csv')
df.head()


# ## 1. Define features and target
# 
# Our **target** (the thing we're predicting) is `treatment`. Everything else (minus a couple columns we'll exclude) becomes our **features**.
# 
# Columns to exclude from features:
# - `treatment` (it's the target, not a feature)
# - `Gender` (we'll use `Gender_clean` instead, which you created in notebook 1)
# - `Country` (has too many categories — one-hot encoding this would create dozens of columns from very few examples each; a reasonable simplification for a first model is to drop it, though you could revisit this later)
# 
# ### TODO
# Create two variables:
# - `X` = `df` with `treatment`, `Gender`, and `Country` dropped
# - `y` = `df['treatment']`
# 
# Then print `X.columns.tolist()` and `X.dtypes` to see what we're working with.

# In[3]:


X = df.drop(columns = ['treatment', 'Gender', 'Country'])
y = df['treatment']
print(X.columns.tolist())
print(X.dtypes)


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# X = df.drop(columns=['treatment', 'Gender', 'Country'])
# y = df['treatment']
# 
# print(X.columns.tolist())
# print(X.dtypes)
# ```
# 
# You should see `Age` as numeric and everything else as `object` (text/categorical). Models need numbers, so the categorical columns need encoding — that's next.
# </details>

# ## 2. Encode categorical features: one-hot encoding
# 
# Models can't work with text like `'Yes'`/`'No'` directly — they need numbers. **One-hot encoding** turns a categorical column with `k` categories into `k` (or `k-1`) binary 0/1 columns, one per category.
# 
# Example: a `remote_work` column with values `Yes`/`No` becomes a single column `remote_work_Yes` with 1s and 0s (using `drop_first=True` to avoid redundancy — if `remote_work_Yes` is 0, we know it was "No").
# 
# **New concept — `pd.get_dummies()`:** does one-hot encoding for you across all categorical columns at once.
# 
# ```python
# X_encoded = pd.get_dummies(X, drop_first=True)
# ```
# 
# ### TODO
# 1. Apply `pd.get_dummies()` to `X` with `drop_first=True`, save as `X_encoded`
# 2. Print `X_encoded.shape` — notice how many columns you now have compared to before
# 3. Look at `X_encoded.head()` to see the binary columns

# In[4]:


X_encoded = pd.get_dummies(X, drop_first=True)
print(X_encoded.shape)
X_encoded.head()


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# X_encoded = pd.get_dummies(X, drop_first=True)
# print(X_encoded.shape)
# X_encoded.head()
# ```
# 
# You went from ~20 columns to likely 40-50+ — this is normal and expected with one-hot encoding of many categorical columns.
# </details>

# ## 3. Encode the target variable
# 
# `y` is currently `'Yes'`/`'No'` text. Most sklearn classifiers can actually handle string labels directly, but it's standard practice to encode the target as 0/1 — it makes metrics like ROC-AUC and coefficient interpretation more straightforward.
# 
# **New concept — `LabelEncoder`:** converts categories to integers (0, 1, 2, ...). For a binary Yes/No column, it'll map them to 0 and 1 (alphabetically: 'No' → 0, 'Yes' → 1).
# 
# ```python
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# ```
# 
# ### TODO
# Encode `y` using `LabelEncoder`. Then check `le.classes_` to confirm which value maps to 0 and which to 1.

# In[5]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(le.classes_)


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# 
# print(le.classes_)  # e.g. array(['No', 'Yes']) -> No=0, Yes=1
# ```
# </details>

# ## 4. Train/test split
# 
# **Why split the data?** If we train and evaluate on the same data, the model could just "memorize" it and look perfect, but fail on new data (this is called **overfitting**). Splitting into a training set (to fit the model) and a test set (to evaluate it, untouched during training) gives a more honest estimate of how the model performs on data it hasn't seen.
# 
# **New concept — `train_test_split()`:**
# ```python
# X_train, X_test, y_train, y_test = train_test_split(
#     X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
# ```
# - `test_size=0.2` → 20% of data held out for testing
# - `random_state=42` → makes the split reproducible (you'll get the same split every time you run it)
# - `stratify=y_encoded` → keeps the same proportion of Yes/No in both train and test sets (important when classes aren't perfectly balanced)
# 
# ### TODO
# Run the split above. Then print the shapes of `X_train`, `X_test`, `y_train`, `y_test` to confirm the sizes make sense.

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# X_train, X_test, y_train, y_test = train_test_split(
#     X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
# )
# 
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# ```
# </details>

# ## 5. Model 1: Logistic Regression
# 
# Despite the name, logistic regression is a **classification** algorithm. It learns a weight (coefficient) for each feature and combines them to estimate the probability of class 1 (`treatment = Yes`). It's a good first model because it's fast, and its coefficients are directly interpretable.
# 
# **New concept — the fit/predict pattern:** almost every sklearn model follows this pattern:
# ```python
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)        # learn from training data
# y_pred = model.predict(X_test)     # predict on unseen data
# ```
# (`max_iter=1000` just gives the optimizer more iterations to converge — you may see a convergence warning without it.)
# 
# ### TODO
# 1. Create a `LogisticRegression(max_iter=1000)` model
# 2. Fit it on `X_train, y_train`
# 3. Predict on `X_test`, save as `y_pred_lr`

# In[7]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# log_reg = LogisticRegression(max_iter=1000)
# log_reg.fit(X_train, y_train)
# y_pred_lr = log_reg.predict(X_test)
# ```
# </details>

# ## 6. Evaluate Logistic Regression
# 
# **New concepts:**
# - **Accuracy** = % of predictions that were correct. Simple, but can be misleading if classes are imbalanced.
# - **Confusion matrix** = a 2x2 table showing true positives, true negatives, false positives, false negatives. `ConfusionMatrixDisplay` can plot it directly.
# - **Classification report** = precision, recall, and F1-score per class.
#   - **Precision**: of everyone the model predicted "Yes", what fraction actually sought treatment?
#   - **Recall**: of everyone who actually sought treatment, what fraction did the model catch?
# 
# ### TODO
# 1. Print `accuracy_score(y_test, y_pred_lr)`
# 2. Plot a confusion matrix using `ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)`
# 3. Print `classification_report(y_test, y_pred_lr)`

# In[8]:


print(accuracy_score(y_test, y_pred_lr))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=le.classes_)
print(classification_report(y_test, y_pred_lr))


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# print("Accuracy:", accuracy_score(y_test, y_pred_lr))
# 
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=le.classes_)
# plt.title('Logistic Regression - Confusion Matrix')
# plt.show()
# 
# print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
# ```
# </details>

# ## 7. Model 2: Random Forest
# 
# A **Random Forest** builds many decision trees on random subsets of the data/features and averages their predictions. Unlike logistic regression, it can capture non-linear relationships and interactions between features without you specifying them.
# 
# ### TODO
# Repeat the fit/predict/evaluate pattern from sections 5-6, but with:
# ```python
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# ```
# Save predictions as `y_pred_rf`, and again print accuracy, confusion matrix, and classification report.

# In[9]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=le.classes_)
print(classification_report(y_test, y_pred_rf))


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# 
# print("Accuracy:", accuracy_score(y_test, y_pred_rf))
# 
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=le.classes_)
# plt.title('Random Forest - Confusion Matrix')
# plt.show()
# 
# print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
# ```
# </details>

# ## 8. Compare the two models
# 
# ### TODO
# Write 2-3 sentences comparing logistic regression and random forest on this dataset:
# - Which had higher accuracy?
# - Were precision/recall noticeably different between the two classes for either model?
# - Given the small difference (if any), which would you prefer, and why? (Hint: think about interpretability vs. performance — there's rarely one "correct" answer)

# - Logistic regression model had higher accuracy.
# - Yes, precision/recall were different between the two classes, but are balanced in terms of which were greater.
# - While Logistic Regression achieved slightly higher overall accuracy, the Random Forest model was better at minimizing False Negatives (missing individuals who actually sought treatment). Depending on the real-world application (e.g., ensuring vulnerable employees receive proactive outreach), a lower false-negative rate might be preferred.

# ## 9. Interpretation — what did the models learn?
# 
# This is the most interview-relevant section: *why* does the model predict what it predicts?
# 
# **Random Forest feature importances:** `rf.feature_importances_` gives a score per feature (higher = more influential in the model's decisions overall, not direction-specific).
# 
# ```python
# importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
# top_features = importances.sort_values(ascending=False).head(10)
# 
# plt.figure(figsize=(8, 6))
# sns.barplot(x=top_features.values, y=top_features.index)
# plt.title('Top 10 Feature Importances (Random Forest)')
# plt.xlabel('Importance')
# plt.show()
# ```
# 
# **Logistic Regression coefficients:** `log_reg.coef_[0]` gives a coefficient per feature. Positive = pushes prediction toward `treatment = Yes`; negative = pushes toward `No`. Larger magnitude = stronger effect.
# 
# ```python
# coefs = pd.Series(log_reg.coef_[0], index=X_encoded.columns)
# top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).head(10).index)
# 
# plt.figure(figsize=(8, 6))
# sns.barplot(x=top_coefs.values, y=top_coefs.index, palette='coolwarm')
# plt.title('Top 10 Logistic Regression Coefficients (by magnitude)')
# plt.xlabel('Coefficient (sign = direction)')
# plt.show()
# ```
# 
# ### TODO
# 1. Plot the Random Forest feature importances
# 2. Plot the Logistic Regression coefficients
# 3. Write 3-4 sentences: do the two models agree on what's important? Do these results match what you found in the EDA notebook? Any surprises?

# In[10]:


importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.show()

coefs = pd.Series(model.coef_[0], index=X_encoded.columns)
top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).head(10).index)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_coefs.values, y=top_coefs.index, palette='coolwarm')
plt.title('Top 10 Logistic Regression Coefficients (by magnitude)')
plt.xlabel('Coefficient (sign = direction)')
plt.show()


# - The two models somewhat agree on which factors are important, especially the work_interfere variables (no, rarely, sometimes, often).
# - However, age and family history are deemed as much more important with the random forest over the logistic regression model.
# - This is interesting, because in the EDA notebook, family history mattered a lot more, which the logistic regression model did not reflect. Additionally, in the EDA notebook, workplace policies showed association with seeking treatment, but in both models this is not reflected. 

# ## 10. Summary and next steps
# 
# Write a final summary covering:
# - Your overall question and approach
# - Key EDA findings (from notebook 1)
# - Model performance (accuracy, and precision/recall for the class that matters most — think about which type of error, false positive or false negative, would matter more in a real application)
# - Top predictive features and what they suggest
# - **Limitations**: small dataset (~1,250 responses, mostly from a 2014 survey, likely skewed toward certain countries/company types), self-reported data, association is not the same as causation
# - **Next steps**: e.g., try other models (SVM, gradient boosting), try `Country` with more careful grouping, hyperparameter tuning with cross-validation, or build a simple Streamlit app where someone inputs their situation and sees the model's prediction

# Overall question and approach:
# - What factors appeared to have an association with seeking treatment?
# 
# Key EDA findings:
# - Based on the barplots, having a family history of mental illness and various work benefits had strong association with seeking treatment.
# 
# Model performances:
# - 27 False positives (Log Regression)
# - 16 False negatives (Log Regression)
# - 34 False positives (Random Forest)
# - 12 False negatives (Random Forest)
# 
# Top Predictive Features & Interpretation:
# In both models, the work_interfere variables (Often, Sometimes, Rarely) were the strongest predictors of seeking treatment. This suggests that the primary catalyst for individuals seeking mental health support is when their symptoms begin visibly impacting their work execution.
# 
# Additionally, the Random Forest heavily weighted Age and family_history, indicating that personal background and demographic nuances play a massive role—complex interactions that the linear Logistic Regression model partially missed. Workplace policy features were surprisingly weak predictors in both models, suggesting individual health status and history drive treatment decisions much more than corporate wellness programs.

# ## Portfolio checklist
# 
# 1. Clean up both notebooks (remove hints/TODOs, Restart Kernel & Run All on each, in order)
# 2. Write a top-level `README.md` with: project question, dataset link, summary of EDA + model findings, your best 2-3 charts as images, and instructions to run both notebooks
# 3. Push `01_eda.ipynb`, `02_modeling.ipynb`, `mental_health_cleaned.csv`, and `README.md` to a GitHub repo
# 4. Add "Open in Colab" badges to your README for both notebooks
