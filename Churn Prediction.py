#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Telco Customer Churn - Logistic Regression


# 1) Import core libraries for data handling, plotting, and ML

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# 2) Load the dataset (update the path if your CSV is elsewhere)

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# 3) Basic cleaning: drop ID, convert charges, remove duplicates/nulls

df = df.drop(columns=["customerID"])


# In[8]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# In[9]:


df = df.drop_duplicates()


# In[10]:


df = df.dropna()


# In[11]:


# 4) Encode Yes/No and gender columns to numeric for the model

yes_no_cols = ["gender", "Partner","Dependents","PhoneService","PaperlessBilling","Churn"]

for col in yes_no_cols:
    df[col] = df[col].map({"Yes":1, "No":0, "Male":1, "Female":0})


# In[12]:


# 5) One-hot encode remaining categorical columns

df = pd.get_dummies(df, drop_first=True)


# In[13]:


# 6) Split data into train/test sets

from sklearn.model_selection import train_test_split

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[14]:


# 7) Scale features so Logistic Regression trains better

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


# 8) Train the Logistic Regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[16]:


# 9) Evaluate on the test set and show key metrics

y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["No Churn","Churn"]))

# 10) Plot a confusion matrix to visualize predictions

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()


# In[17]:


# 11) Predict churn probability for one sample (first test row)

sample = X_test.iloc[[0]]          # take one row
sample_scaled = scaler.transform(sample)
prob_churn = model.predict_proba(sample_scaled)[0][1]
print("Churn probability:", prob_churn)
print("Predicted class:", model.predict(sample_scaled)[0])


# In[18]:


# 12) Quick exploratory plots (class balance and relationships)

sns.countplot(x='Churn', data=df)


# In[19]:


sns.countplot(x='gender', hue='Churn', data=df)


# In[27]:


sns.pairplot(df[['tenure','MonthlyCharges','TotalCharges','Churn']])


# In[28]:


# Correlation heatmap for numeric features

plt.figure(figsize=(8,5))
numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Churn']

sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Simple Features)")
plt.show()


# In[24]:


# More distribution views

sns.histplot(df, x='tenure', hue='Churn', kde=True)


# In[25]:


sns.boxplot(x='Churn', y='MonthlyCharges', data=df)


# In[ ]:




