#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


pd.set_option('display.max_columns',None)
df = pd.read_csv('https://raw.githubusercontent.com/krishnanandgs/aitraining/main/assignment/testdata/fraudtest.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.duplicated().sum()


# In[6]:


df.describe()


# In[7]:


df.nunique()


# In[9]:


#Data Visulization
fig, axes = plt.subplots(2,2, figsize=(20,20))



# In[10]:


# Total amount spend by gender
print(df.groupby('gender')['amt'].sum())
colors = ['b','r']
df.groupby('gender')['amt'].sum().plot(kind='bar',color=colors)
plt.show()


# In[11]:


# drop unnecessary columns 
df = df.drop(columns=['Unnamed: 0','trans_date_trans_time','lat','long','city_pop','zip','merch_lat','merch_long','unix_time'])


# In[14]:


# Category wise total amount 
plt.figure(figsize=(15,8))
df.groupby('category')['amt'].sum().sort_values(ascending=True).plot(kind='barh',color='crimson')
plt.xlabel('Sum of Amount by Category(in lacs)')
plt.show()


# In[15]:


fraud_count_by_city = df.groupby('city')['is_fraud'].sum().sort_values(ascending=False)
fraud_count_by_city = fraud_count_by_city[fraud_count_by_city > 0]

df2 = pd.DataFrame(data = fraud_count_by_city).reset_index()
df2 = df2.rename(columns={'is_fraud':'total_fraud'})
df2


# In[16]:


fraud_count_by_merchant = df.groupby('merchant')['is_fraud'].sum().sort_values(ascending=False)
fraud_count_by_merchant = fraud_count_by_merchant[fraud_count_by_merchant > 0]

df3 = pd.DataFrame(data=fraud_count_by_merchant).reset_index()
df3.rename(columns={'is_fraud':'total_fraud'},inplace=True)
df3


# In[17]:


df4 = pd.pivot_table(data=df,index=['state'],columns='gender',values='is_fraud',aggfunc='sum',fill_value=0)
df4


# In[18]:


df5 = pd.pivot_table(data=df,index=df['job'],values='is_fraud',columns=df['is_fraud'],aggfunc='sum',fill_value=0)
df5


# In[19]:


df.drop(columns=['cc_num','first','last','street','dob','trans_num'],inplace=True)


# In[20]:


le = LabelEncoder()
for columns in df.columns:
    if df[columns].dtype == 'object':
        df[columns] = le.fit_transform(df[columns])


# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[22]:


skewd = scipy.stats.skew(df.select_dtypes(np.number))
skewd_df = pd.DataFrame(skewd, df.columns)
skewd_df


# In[23]:


data = df.iloc[:,:-1]
target = df.iloc[:,-1:]


# In[24]:


#Scaling

sc = StandardScaler()
data = sc.fit_transform(data)


# In[25]:


x_train , x_test , y_train , y_test = train_test_split(data , target , test_size = .2, random_state=42)


# In[26]:


#Model selection
model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()
model_knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
model_rfc = RandomForestClassifier()


# In[27]:


model_lr.fit(x_train,y_train)
model_dt.fit(x_train,y_train)
model_knn.fit(x_train,y_train)
model_rfc.fit(x_train,y_train)


# In[28]:


y_pred_lr = model_lr.predict(x_test)
y_pred_dt = model_dt.predict(x_test)
y_pred_knn = model_knn.predict(x_test)
y_pred_rfc = model_rfc.predict(x_test)


# In[29]:


lr_score = accuracy_score(y_test,y_pred_lr)
dt_score = accuracy_score(y_test,y_pred_dt)
knn_score = accuracy_score(y_test, y_pred_knn)
rfc_score = accuracy_score(y_test,y_pred_rfc)

print(f'Logistic Regression            :{lr_score}')
print(f'Decision Tree Classifier       :{dt_score}')
print(f'K-Nearest Neighbors Classifier :{knn_score}')
print(f'RandomForest Classifier        :{rfc_score}')


# In[30]:


fig, axes = plt.subplots(2,2, figsize=(20,20))

cfm = confusion_matrix(y_test,y_pred_lr)
sns.heatmap(cfm,annot=True,ax=axes[0,0],cmap='Blues',fmt='d')

cfm = confusion_matrix(y_test,y_pred_dt)
sns.heatmap(cfm,annot=True,ax=axes[0,1],cmap='Blues',fmt='d')

cfm = confusion_matrix(y_test,y_pred_dt)
sns.heatmap(cfm,annot=True,ax=axes[1,0],cmap='Blues',fmt='d')

cfm = confusion_matrix(y_test,y_pred_rfc)
sns.heatmap(cfm,annot=True,ax=axes[1,1],cmap='Blues',fmt='d')

plt.show()


# In[ ]:




