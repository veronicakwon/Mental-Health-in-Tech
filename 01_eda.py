#!/usr/bin/env python
# coding: utf-8

# # Mental Health in Tech: Exploratory Data Analysis
# 
# **Project question:** What workplace and personal factors are associated with tech employees seeking treatment for a mental health condition?
# 
# **Dataset:** [OSMI Mental Health in Tech Survey (2014)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) — `survey.csv`, ~1,250 responses.
# 
# This is **notebook 1 of 2**. Here we'll clean the data and explore it. At the end, we'll save a cleaned version that notebook 2 (`02_modeling.ipynb`) will use to build a predictive model.
# 
# ## How to use this notebook
# Each section has: a short explanation → a `# TODO` code cell for you to write → a collapsed hint you can expand if stuck. Try writing the code yourself first.
# 
# ## Setup
# 1. Download `survey.csv` from the Kaggle link above
# 2. Upload it to your Colab session (or Google Drive) / same folder if running locally
# 3. Update `DATA_PATH` below if needed

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


# ## 1. First look at the data
# 
# ### TODO
# Check `df.shape`, `df.columns`, `df.info()`, and `df.isnull().sum()`. This dataset has more missing values and messier columns than a typical "clean" Kaggle dataset — that's normal for real survey data, and part of why this is good practice.

# In[3]:


df.shape
df.columns
df.info()
df.isnull().sum()


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# print(df.shape)
# print(df.columns.tolist())
# df.info()
# df.isnull().sum()
# ```
# 
# You should see that `state`, `self_employed`, `work_interfere`, and `comments` have a lot of missing values. We'll handle each differently below.
# </details>

# ## 2. Drop columns we won't use
# 
# - `Timestamp`: when the survey was filled out — not useful for our question
# - `comments`: free-text, mostly missing — out of scope for this project (could be a future NLP project!)
# - `state`: only filled in for US respondents, lots of missing values, and we already have `Country`
# 
# **New concept — `.drop(columns=[...])`:** removes columns from a DataFrame. By default this returns a *new* DataFrame, so either assign it back to `df` or pass `inplace=True`.
# 
# ### TODO
# Drop `Timestamp`, `comments`, and `state` from `df`.

# In[4]:


df = df.drop(columns=['Timestamp', 'comments', 'state'])


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# df = df.drop(columns=['Timestamp', 'comments', 'state'])
# ```
# </details>

# ## 3. Clean the `Age` column
# 
# Real survey data often has junk values. Run `df['Age'].describe()` and `sorted(df['Age'].unique())` — you'll likely spot some impossible ages (negative numbers, or numbers in the millions/billions).
# 
# **New concept — boolean filtering:** `df[(df['Age'] >= 18) & (df['Age'] <= 100)]` keeps only rows where the condition is True. This is one of the most common patterns in pandas.
# 
# ### TODO
# 1. Inspect `Age` for weird values
# 2. Filter `df` to keep only rows where `Age` is between 18 and 100 (inclusive)
# 3. Confirm with `df['Age'].describe()` that it looks reasonable now

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

# ## 4. Clean the `Gender` column
# 
# This is the messiest column — run `df['Gender'].unique()` and you'll see dozens of variations: `'Male'`, `'male'`, `'M'`, `'m'`, `'Cis Male'`, `'Woman'`, `'f'`, `'Female '`, `'something kinda male?'`, etc.
# 
# **New concept — `.str.lower().str.strip()`:** lowercases and removes leading/trailing whitespace, which immediately fixes a lot of the duplication (`'Male'` vs `'male'` vs `'Male '`).
# 
# **New concept — `.apply()` with a custom function:** for the remaining variety, you can write a function that maps many strings to a few categories, then apply it to every value in the column.
# 
# ### TODO
# 1. Print `df['Gender'].unique()` to see the variety
# 2. Lowercase and strip whitespace
# 3. Write a function `clean_gender(g)` that maps values into `'male'`, `'female'`, or `'other'`, then apply it to create a new column `Gender_clean`
# 
# There's no single "correct" mapping here — use your judgment, and **document your choices** in a markdown cell. This kind of judgment call is exactly the kind of thing to mention in an interview.

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



# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# print(df['Gender'].unique())
# 
# df['Gender'] = df['Gender'].str.lower().str.strip()
# 
# male_terms = ['male', 'm', 'cis male', 'man', 'malr', 'cis man', 'male (cis)']
# female_terms = ['female', 'f', 'cis female', 'woman', 'femake', 'cis-female/femme', 'female (cis)']
# 
# def clean_gender(g):
#     if g in male_terms:
#         return 'male'
#     elif g in female_terms:
#         return 'female'
#     else:
#         return 'other'
# 
# df['Gender_clean'] = df['Gender'].apply(clean_gender)
# df['Gender_clean'].value_counts()
# ```
# 
# Your exact lists might differ slightly — that's fine. Just check `value_counts()` afterward to make sure nothing obviously wrong got grouped together.
# </details>

# ## 5. Handle remaining missing values
# 
# Two columns still have missing values worth handling deliberately:
# 
# - **`self_employed`**: missing values likely mean the person didn't answer — since most respondents are *not* self-employed, filling with `'No'` is a reasonable default
# - **`work_interfere`**: this question ("if you have a mental health condition, does it interfere with work?") was likely skipped by people who don't have a mental health condition — filling with `'Not applicable'` preserves that information rather than treating it as "unknown"
# 
# ### TODO
# Use `.fillna()` to fill `self_employed` with `'No'` and `work_interfere` with `'Not applicable'`. Then confirm with `df.isnull().sum()` that there are no more missing values (besides any you intentionally left).

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

# ## 6. Question 1 — What's the baseline rate of treatment-seeking?
# 
# Before looking at *what's related to* `treatment`, look at the variable itself. This becomes your baseline: if 70% of people sought treatment overall, then a feature that's associated with 71% isn't very interesting, but one associated with 95% is.
# 
# **New concept — `sns.countplot()`:** shows counts of a categorical variable as bars. `df['treatment'].value_counts(normalize=True)` gives you proportions instead of raw counts.
# 
# ### TODO
# 1. Print `df['treatment'].value_counts(normalize=True)`
# 2. Plot a countplot of `treatment`

# In[8]:


print(df['treatment'].value_counts(normalize=True))

sns.countplot(data=df, x='treatment')
plt.title('Sought Treatment?')
plt.show()


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# print(df['treatment'].value_counts(normalize=True))
# 
# sns.countplot(data=df, x='treatment')
# plt.title('Sought Treatment?')
# plt.show()
# ```
# 
# If this is close to 50/50, that's actually convenient for the modeling notebook — a roughly balanced target is easier to evaluate.
# </details>

# ## 7. Question 2 — Does family history relate to treatment-seeking?
# 
# **New concept — grouped countplot with `hue`:** passing `hue='treatment'` to `sns.countplot()` splits each `family_history` bar into two (treatment = Yes/No), letting you compare proportions.
# 
# **New concept — `pd.crosstab()`:** builds a table of counts (or percentages, with `normalize='index'`) for two categorical variables — useful to see the exact numbers behind the plot.
# 
# ### TODO
# 1. `sns.countplot(data=df, x='family_history', hue='treatment')`
# 2. `pd.crosstab(df['family_history'], df['treatment'], normalize='index')`
# 3. Write a sentence: does having a family history of mental illness seem associated with seeking treatment?

# In[9]:


# TODO: countplot and crosstab for family_history vs treatment
sns.countplot(data=df, x = 'family_history', hue = 'treatment')
pd.crosstab(df['family_history'], df['treatment'], normalize='index')


# Yes, it appears having a family history of mental illness is associated with the lack of seeking treatment. 

# ## 8. Question 3 — Does `work_interfere` relate to treatment-seeking?
# 
# `work_interfere` has values like `'Never'`, `'Rarely'`, `'Sometimes'`, `'Often'`, `'Not applicable'` — it's an *ordinal* variable (has a natural order), which makes it a good candidate for a chart where order matters.
# 
# **New concept — controlling category order:** pass `order=['Not applicable', 'Never', 'Rarely', 'Sometimes', 'Often']` to `sns.countplot()` so the bars appear in a logical sequence instead of alphabetical/random order.
# 
# ### TODO
# 1. Create a countplot of `work_interfere` with `hue='treatment'`, using the order above
# 2. Write a sentence interpreting the pattern — is there a trend as interference increases?

# In[10]:


sns.countplot(data=df, x = 'work_interfere', hue = 'treatment', order=['Not applicable', 'Never', 'Rarely', 'Sometimes', 'Often'])
plt.title('Work Interference vs Treatment')
plt.show()


# No, there is no trend as interference increases. 

# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# order = ['Not applicable', 'Never', 'Rarely', 'Sometimes', 'Often']
# 
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='work_interfere', hue='treatment', order=order)
# plt.title('Work Interference vs Treatment')
# plt.show()
# ```
# </details>

# ## 9. Question 4 — Do workplace policies relate to treatment-seeking?
# 
# Three columns describe workplace mental health support: `benefits`, `care_options`, and `wellness_program` (each typically Yes/No/Don't know).
# 
# **New concept — looping to make multiple subplots:** rather than copy-pasting plotting code three times, use `plt.subplots()` to create a grid of plots and loop over the columns.
# 
# ```python
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# cols = ['benefits', 'care_options', 'wellness_program']
# 
# for ax, col in zip(axes, cols):
#     sns.countplot(data=df, x=col, hue='treatment', ax=ax)
#     ax.set_title(col)
# 
# plt.tight_layout()
# plt.show()
# ```
# 
# ### TODO
# Run the code above (or adapt it), then write 1-2 sentences: do workplace policies seem related to treatment-seeking? Any surprises?

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

# ## 10. Question 5 — Your own question
# 
# Pick one (or come up with your own):
# 
# - Does company size (`no_employees`) relate to whether benefits/wellness programs are offered, or to treatment-seeking?
# - Are remote workers (`remote_work`) more or less likely to seek treatment?
# - Do people who fear negative consequences for discussing mental health (`mental_health_consequence`) differ in treatment-seeking from those who don't?
# - How does `Age` differ between those who sought treatment and those who didn't? (Hint: `sns.boxplot(data=df, x='treatment', y='Age')` or `sns.histplot(..., hue='treatment')`)
# - What are the top 5-10 countries by number of respondents, and does treatment rate vary across them? (Careful: some countries have very few respondents, so percentages can be misleading — consider filtering to countries with at least, say, 20 respondents)
# 
# ### TODO
# State your question, why it's interesting, then explore it with at least one visualization.

# **My question:** What are the top 5-10 countries by number of respondents, and does treatment rate vary across them?
# 
# **Why it's interesting:** If we can find some associations within a country, we could do more research on mental health in other fields other than tech within that specific country. 

# In[12]:


top_countries = df_countries[df_countries >= 25]

top_countries_list = ['United States', 'United Kingdom', 'Canada', 'Germany', 'Netherlands', 'Ireland']
fig, axes = plt.subplots(1, 6, figsize=(18, 5))

for ax, col in zip(axes, top_countries_list):
    df_country = df[df['Country'] == col]
    sns.countplot(data=df_country, x='treatment', hue='treatment', ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


# ## 11. Summary of EDA findings
# 
# Write 3-5 bullet points summarizing what you found. This will feed directly into your README and into the framing of the modeling notebook (e.g., "EDA suggested family history and work interference are strongly associated with treatment, so we'd expect these to be important features").

# EDA suggested:
# - Family history and workplace policies are strongly associated with treatment, so we'd expect these to be important features.

# ## 12. Save the cleaned dataset
# 
# Notebook 2 (`02_modeling.ipynb`) will load this cleaned file instead of the raw `survey.csv`, so it doesn't have to repeat all the cleaning steps.
# 
# **New concept — `.to_csv()`:** writes a DataFrame to a CSV file. `index=False` avoids writing the row numbers as an extra column.
# 
# ### TODO
# Save `df` to `mental_health_cleaned.csv`. If you're in Colab, this saves to the session's temporary storage — download it afterward (or save to Drive) so you can upload it when you start notebook 2.

# In[ ]:


df.to_csv('mental_health_cleaned.csv', index=False)


# <details>
# <summary>💡 Hint / Solution</summary>
# 
# ```python
# df.to_csv('mental_health_cleaned.csv', index=False)
# ```
# </details>

# ## Next: `02_modeling.ipynb`
# 
# Once this notebook runs cleanly top-to-bottom and you've written your summary, move on to the modeling notebook, where we'll use `mental_health_cleaned.csv` to predict `treatment` with scikit-learn.
