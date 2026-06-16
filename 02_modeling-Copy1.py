#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


X = df.drop(columns = ['treatment', 'Gender', 'Country'])
y = df['treatment']
print(X.columns.tolist())
print(X.dtypes)


# In[4]:


X_encoded = pd.get_dummies(X, drop_first=True)
print(X_encoded.shape)
X_encoded.head()


# In[5]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(le.classes_)


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


# In[7]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_lr = model.predict(X_test)


# In[8]:


print(accuracy_score(y_test, y_pred_lr))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, display_labels=le.classes_)
print(classification_report(y_test, y_pred_lr))


# In[9]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, display_labels=le.classes_)
print(classification_report(y_test, y_pred_rf))


# - Logistic regression model had higher accuracy.
# - Yes, precision/recall were different between the two classes, but are balanced in terms of which were greater.
# - While Logistic Regression achieved slightly higher overall accuracy, the Random Forest model was better at minimizing False Negatives (missing individuals who actually sought treatment). Depending on the real-world application (e.g., ensuring vulnerable employees receive proactive outreach), a lower false-negative rate might be preferred.

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
# 
# Limitations:
# - This is a small data set with only ~1,250 responses and from 2014
# - Skewed toward certain countries/company types
# - Self reported data
# 
# Next steps:
# - Try other models (SVM, gradient boosting),
# - Try Country with more careful grouping
# - Build a simple Streamlit app where someone inputs their situation and sees the model's prediction
