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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/krishnanandgs/aitraining/main/assignment/testdata/fraudtest.csv')
df.head()

df.shape

df.info()

def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summ['Missing#'] = df.isna().sum()
    summ['Missing%'] = (df.isna().sum())/len(df)
    summ['Dups'] = df.duplicated().sum()
    summ['Uniques'] = df.nunique().values
    summ['Count'] = df.count().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['Min'] = desc['min'].values
    summ['Max'] = desc['max'].values
    summ['Average'] = desc['mean'].values
    summ['Standard Deviation'] = desc['std'].values
    summ['First Value'] = df.loc[0].values
    summ['Second Value'] = df.loc[1].values
    summ['Third Value'] = df.loc[2].values

    display(summ)

summary(df)


# Total amount spend by gender
print(df.groupby('gender')['amt'].sum())
colors = ['b','r']
df.groupby('gender')['amt'].sum().plot(kind='bar',color=colors)
plt.show()

# drop unnecessary columns 
df = df.drop(columns=['Unnamed: 0','trans_date_trans_time','lat','long','city_pop','zip','merch_lat','merch_long','unix_time'])

# Category wise total amount 
plt.figure(figsize=(15,8))
df.groupby('category')['amt'].sum().sort_values(ascending=True).plot(kind='barh',color='crimson')
plt.xlabel('Sum of Amount by Category(in lacs)')
plt.show()

fraud_count_by_city = df.groupby('city')['is_fraud'].sum().sort_values(ascending=False)
fraud_count_by_city = fraud_count_by_city[fraud_count_by_city > 0]

df2 = pd.DataFrame(data = fraud_count_by_city).reset_index()
df2 = df2.rename(columns={'is_fraud':'total_fraud'})
df2


fraud_count_by_merchant = df.groupby('merchant')['is_fraud'].sum().sort_values(ascending=False)
fraud_count_by_merchant = fraud_count_by_merchant[fraud_count_by_merchant > 0]

df3 = pd.DataFrame(data=fraud_count_by_merchant).reset_index()
df3.rename(columns={'is_fraud':'total_fraud'},inplace=True)
df3

df4 = pd.pivot_table(data=df,index=['state'],columns='gender',values='is_fraud',aggfunc='sum',fill_value=0)
df4

df5 = pd.pivot_table(data=df,index=df['job'],values='is_fraud',columns=df['is_fraud'],aggfunc='sum',fill_value=0)
df5

df.drop(columns=['cc_num','first','last','street','dob','trans_num'],inplace=True)

le = LabelEncoder()
for columns in df.columns:
    if df[columns].dtype == 'object':
        df[columns] = le.fit_transform(df[columns])

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
plt.show()

data = df.iloc[:,:-1]
target = df.iloc[:,-1:]


#Scaling

sc = StandardScaler()
data = sc.fit_transform(data)

x_train , x_test , y_train , y_test = train_test_split(data , target , test_size = .2, random_state=42)

#Model selection
model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()
model_knn = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
model_rfc = RandomForestClassifier()
model_gbm = GradientBoostingClassifier()

model_lr.fit(x_train,y_train)
model_dt.fit(x_train,y_train)
model_knn.fit(x_train,y_train)
model_rfc.fit(x_train,y_train)
model_gbm.fit(x_train,y_train)


y_pred_lr = model_lr.predict(x_test)
y_pred_dt = model_dt.predict(x_test)
y_pred_knn = model_knn.predict(x_test)
y_pred_rfc = model_rfc.predict(x_test)
y_pred_gbm = model_gbm.predict(x_test)

lr_score = accuracy_score(y_test,y_pred_lr)
dt_score = accuracy_score(y_test,y_pred_dt)
knn_score = accuracy_score(y_test, y_pred_knn)
rfc_score = accuracy_score(y_test,y_pred_rfc)
gbm_score = accuracy_score(y_test,y_pred_gbm)

print(f'Logistic Regression            :{lr_score}')
print(f'Decision Tree Classifier       :{dt_score}')
print(f'K-Nearest Neighbors Classifier :{knn_score}')
print(f'RandomForest Classifier        :{rfc_score}')
print(f'Gradient Boosting Classifier        :{gbm_score}')
