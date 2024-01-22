#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


# In[3]:


df = pd.read_csv('C:/Users/poste/Documents/data science/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()


# In[4]:


df.shape


# In[5]:


# Checking the presence of null values
df.isnull().sum()


# In[6]:


# Checking for the presence of duplicate values
df.duplicated().sum()


# In[7]:


# Overview of the entire dataset

df.info()


# In[8]:


# Let's drop the customerID variable because it's not really a valuable information for further analysis

df = df.drop('customerID',axis=1)


# In[9]:


df.head() # customerID variable is removed


# In[10]:


fig = px.histogram(df, x="PaymentMethod",color='PaymentMethod')
fig.show()


# In[11]:


# Customer churn vs gender

df_plot = df.copy()
df_plot['gender'] = df_plot['gender'].replace({1:"Male",2:"Female"})
df_plot['Churn'] = df_plot['Churn'].replace({0:"No",1:"Yes"})
sns.set(style="whitegrid")
sns.set_style("white")
sns.despine()
palette = [ '#a3a3ec','#ff9f9f']
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=df_plot, x='gender', hue='Churn', palette=palette)
plt.xlabel("Gender", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Customer churn Count by Gender", fontsize=16, fontweight='bold')

total_counts = len(df)

for p in ax.patches:
    count = int(p.get_height())
    percentage = f"{100 * count / total_counts:.2f}%"
    ax.annotate(f'{count} ({percentage})', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black')

plt.show()


# In[12]:


fig = px.histogram(df, x="MonthlyCharges", histnorm='probability density')
fig.show()


# In[13]:


fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn")
fig.update_traces(quartilemethod="exclusive") 
fig.show()


# In[14]:


from scipy.stats import shapiro
stat, p_value = shapiro(df['MonthlyCharges'])
print("Shapiro-Wilk Test Statistic:", stat)
print("P-value:", p_value)
alpha = 0.05
if p_value < alpha:
    print("The MonthlyCharges variable does not follow a normal distribution (reject the null hypothesis of normality).")
else:
    print("The MonthlyCharges variable follows a normal distribution (fail to reject the null hypothesis of normality).")


# In[15]:


import statsmodels.api as sm
sm.qqplot(df['MonthlyCharges'], line='s')
plt.title('Q-Q Plot of MonthlyCharges')
plt.show()


# In[16]:


from scipy.stats import mannwhitneyu
monthly_charges_churn_yes = df[df['Churn'] == 'Yes']['MonthlyCharges']
monthly_charges_churn_no = df[df['Churn'] == 'No']['MonthlyCharges']
u_stat, p_value_mw = mannwhitneyu(monthly_charges_churn_yes, monthly_charges_churn_no)
print("Mann-Whitney U Statistic:", u_stat)
print("P-value:", p_value_mw)
alpha = 0.05
if p_value_mw < alpha:
    print("There is a significant difference in the distributions of MonthlyCharges between churned and non-churned customers.")
else:
    print("There is no significant difference in the distributions of MonthlyCharges between churned and non-churned customers.")


# In[17]:


monthly_charges_churn_yes = df[df['Churn'] == 'Yes']['MonthlyCharges']
monthly_charges_churn_no = df[df['Churn'] == 'No']['MonthlyCharges']
u_stat, p_value_mw = mannwhitneyu(monthly_charges_churn_yes, monthly_charges_churn_no, alternative='greater')
print("Mann-Whitney U Statistic:", u_stat)
print("P-value:", p_value_mw)
alpha = 0.05
if p_value_mw < alpha:
    print("The median MonthlyCharges for churned customers is significantly greater than non-churned customers.")
else:
    print("There is no significant evidence that the median MonthlyCharges for churned customers is greater than non-churned customers.")


# In[18]:


df2 = df.copy()


# In[19]:


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded = df_encoded.astype(int)
df_encoded.head()
df_encoded['Churn'] = df2['Churn']


# In[20]:


df_encoded = df_encoded.drop('Churn_Yes',axis=1)
df_encoded = df_encoded.drop('Churn_No',axis=1)


# In[21]:


df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
df_encoded.head()


# In[22]:


from sklearn.preprocessing import MinMaxScaler
numeric_cols = df_encoded.select_dtypes(include=['int']).columns.tolist()
scaler = MinMaxScaler()
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
df_encoded.head()


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[24]:


X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


# In[25]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


logistic_model = LogisticRegression(max_iter=1000)


# In[27]:


logistic_model.fit(X_train, y_train)


# In[28]:


y_pred = logistic_model.predict(X_test)


# In[29]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)


# In[30]:


print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report_str}")


# In[31]:


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:




