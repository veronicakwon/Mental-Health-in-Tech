#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


DATA_PATH = "survey.csv"

df = pd.read_csv(DATA_PATH)
df.head()


# In[3]:


df.shape
df.columns
df.info()
df.isnull().sum()


# In[4]:


df = df.drop(columns=['Timestamp', 'comments', 'state'])


# In[5]:


df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
df['Age'].describe()


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# print(df['Age'].describe())
# 
# df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
# 
# df['Age'].describe()
# ```
# 
# Note: filtering rows changes the DataFrame's index (it'll have gaps). You can reset it with `df = df.reset_index(drop=True)` — not strictly required, but tidy.
# </details>

# In[6]:


df['Gender'] = df['Gender'].str.lower().str.strip()
def clean_gender(g):
    """Creates a whole new column Gender_clean with genders only 'male', 'female', and other.

    """
    male_list = ['m', 'male', 'male-ish', 'maile', 'something kinda male?', 'cis male', 'mal', 'male (cis)', 'make', 'guy (-ish) ^_^', 'man', 'msle', 'mail', 'malr', 'cis man', 'ostensibly male, unsure what that really means']
    female_list = ['female', 'trans-female', 'cis female', 'f', 'woman', 'femake', 'cis-female/femme', 'trans woman', 'female (trans)', 'female (cis)', 'femail']
    if g in male_list:
        return 'male'
    elif g in female_list:
        return 'female'
    else:
        return 'other'

df['Gender'] = df['Gender'].apply(clean_gender)
df['Gender']



# In[7]:


df = df.fillna('No')
df = df.fillna('Not applicable')
df.isnull().sum()


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# df['self_employed'] = df['self_employed'].fillna('No')
# df['work_interfere'] = df['work_interfere'].fillna('Not applicable')
# 
# df.isnull().sum()
# ```
# </details>

# In[8]:


print(df['treatment'].value_counts(normalize=True))

sns.countplot(data=df, x='treatment')
plt.title('Sought Treatment?')
plt.show()


# In[9]:


# TODO: countplot and crosstab for family_history vs treatment
sns.countplot(data=df, x = 'family_history', hue = 'treatment')
pd.crosstab(df['family_history'], df['treatment'], normalize='index')
plt.title('family_history vs treatment')


# Yes; more people who responded "Yes" to family history of mental illness also sought treatment, while those who do not have a family member that has experienced poor mental health were less likely to seek treatment. 

# In[10]:


sns.countplot(data=df, x = 'work_interfere', hue = 'treatment', order=['Not applicable', 'Never', 'Rarely', 'Sometimes', 'Often'])
plt.title('Work Interference vs Treatment')
plt.show()


# Somewhat - when work starts to interfere, this is when more tech workers are likely to seek treatment. 

# In[11]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cols = ['benefits', 'care_options', 'wellness_program']

for ax, col in zip(axes, cols):
    sns.countplot(data=df, x=col, hue='treatment', ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


# _Your observations here..._
# Yes it appears so. When it is certain that the workplace has these policies, there is a dramatic increase of people who say yes to seeking treatment. 

# In[12]:


top_countries_list = ['United States', 'United Kingdom', 'Canada', 'Germany', 'Netherlands', 'Ireland']
fig, axes = plt.subplots(1, 6, figsize=(18, 5))

for ax, col in zip(axes, top_countries_list):
    df_country = df[df['Country'] == col]
    sns.countplot(data=df_country, x='treatment', hue='treatment', ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


# EDA suggested:
# - Family history and workplace policies are strongly associated with treatment, so we'd expect these to be important features.

# In[13]:


df.to_csv('mental_health_cleaned.csv', index=False)

