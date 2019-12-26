# -*- coding: utf-8 -*-
"""
Created on Thursday Dec 26 08:03:27 2019

@author: anchal
"""
import json
import csv
import pandas as pd
employ_data = open('/tmp/EmployData.csv', 'w')
Create the csv writer object
csvwriter = csv.writer(employ_data)
count = 0
for emp in emp_data:
      if count == 0:
             header = emp.keys()
             csvwriter.writerow(header)
             count += 1
      csvwriter.writerow(emp.values())
Make sure to close the file in order to save the contents
employ_data.close()
        
        
with open('customersdata.txt') as json_file:
    data = json.load(json_file)

x = json.loads(x)

f = csv.writer(open("test.csv", "wb+"))

# Write CSV Header, If you dont need that, remove this line
f.writerow(["pk", "model", "codename", "name", "content_type"])

for x in x:
    f.writerow([x["pk"],
                x["model"],
                x["fields"]["codename"],
                x["fields"]["name"],
                x["fields"]["content_type"]])

f = open('customersdata.json')
data = json.load(f)
f.close()
json=json.loads("customersdata.json")
# IMPORTING LIBRARIES

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# IMPORTING DATASET
df= pd.read_csv("customersdata.csv")

# Filling the customers data and target variable with previous values to eliminate blank values
df1=df.iloc[:,0:6]
df1 = df1.fillna(method='ffill')
df2=df.iloc[:,6:]
data1=pd.concat([df1,df2],axis=1)

#DATA EXPLORATION
# Summarize data
data1.info()
data1.head(10)
data1.tail(10)
data1.dtypes    # Knowing the data types - Only 2 numeric variables, rest are categorical
data1.shape
data1.columns
data1.isnull().sum()  
data1=data1.drop(['orders__orderId','paymentMethods__paymentMethodId',
'transactions__transactionId','transactions__orderId','transactions__paymentMethodId'],axis=1) # Dropping ID columns

#Studying each variable
# Data visualiztions for all variables

# staudying Target variable
data1['fraudulent'].value_counts()
data1['fraudulent'].value_counts().plot.pie()  #Area of no-fraud customers (0) is higher
data1['fraudulent'] = data1['fraudulent'].astype("int64")
sns.countplot(x='fraudulent',data=data1, palette='hls') # Same can be seen here
plt.show()

fraud = len(data1[data1['fraudulent']==1])
no_fraud = len(data1[data1['fraudulent']==0])
pct_of_fraud = fraud/(fraud+no_fraud)
print("percentage of fraud", pct_of_fraud*100)
pct_of_no_fraud = no_fraud/(fraud+no_fraud)
print("percentage of no_fraud", pct_of_no_fraud*100)
"""Percentage of fraud = 44.90084985835694 < Percentage of no_fraud = 55.09915014164306"""

data1.groupby('fraudulent').mean()
"""Mean of transactions amt for no_fraud=29.363296 < Mean of transactions amt for fraud= 43.507109""" 
#Studying Numerical Variable
summary=data1.describe() 

data1.boxplot()                                                         
for features in data1.dtypes[data1.dtypes != "object"].index:
    sns.boxplot(x=features, data=data1)
    plt.show()
"""Only 1 outlier in data and it looks positively skewed"""

for features in data1.dtypes[data1.dtypes!="object"].index:
    sns.scatterplot(x=features, y="fraudulent",data=data1)               
    plt.show()
"""Outlier>350"""

mean=data1['transactions__transactionAmount'].mean()
data1["transactions__transactionAmount"].min()
data1["transactions__transactionAmount"].max()
data1["transactions__transactionAmount"].median()
"""As mean = 34.6 > median = 34, so very slightly positively skewed"""

data1['Amount']=data1["transactions__transactionAmount"].replace(np.nan,mean)  #Imputing missing values with mean
data1=data1.drop(['orders__orderAmount','transactions__transactionAmount'],axis=1) # Deleting other variables

f,ax=plt.subplots(figsize=(18,8))  
sns.violinplot("Amount", hue="fraudulent", data=data1,split=True) #Violin Plot

pd.crosstab(data1.fraudulent,data1.Amount.sum()).plot(kind='bar') #Bar plot after misssing value imputation
plt.title('Fraud vs Amount')
plt.xlabel('Fraud')
plt.ylabel('Amount')

#  Studying Categorical vars 
data1.isnull().sum()  

missing_cols_prevval=['orders__orderState','paymentMethods__paymentMethodRegistrationFailure',
              'paymentMethods__paymentMethodType','paymentMethods__paymentMethodProvider',
              'paymentMethods__paymentMethodIssuer','transactions__transactionFailed']

Prevval_imputed = data1[missing_cols_prevval].fillna(method='ffill') # Since missing values large so imputing them with their previous values
Prevval_imputed.isnull().any() #verifying if imputation is done

pd.crosstab(data1.transactions__transactionFailed,data1.fraudulent).plot(kind='bar')
plt.title('Transactions failed vs Fraud')
plt.xlabel('Transactions failed')
plt.ylabel('Fraud')
"""Failed transactions < Successfull transactions, proportion of fraud looks same both cases"""

pd.crosstab(data1.paymentMethods__paymentMethodIssuer,data1.fraudulent).plot(kind='bar')
plt.title('Payment Method vs Fraud')
plt.xlabel('Payment Method')
plt.ylabel('Fraud')
"""Some payment methods only show fraudulent activities with "Her majesty trust" being the most reliable one"""

pd.crosstab(data1.paymentMethods__paymentMethodProvider,data1.fraudulent).plot(kind='bar')
plt.title('Payment Method Provider vs Fraud')
plt.xlabel('Payment Method Provider')
plt.ylabel('Fraud')

pd.crosstab(data1.paymentMethods__paymentMethodIssuer,data1.fraudulent).plot(kind='bar')
plt.title('Payment Method Issuer vs Fraud')
plt.xlabel('Payment Method Issuer')
plt.ylabel('Fraud')

pd.crosstab(data1.paymentMethods__paymentMethodRegistrationFailure,data1.fraudulent).plot(kind='bar')
plt.title('Payment Method Registration Failure vs Fraud')
plt.xlabel('Payment Method Registration Failure')
plt.ylabel('Fraud')

pd.crosstab(data1.orders__orderState,data1.fraudulent).plot(kind='bar')
plt.title('Order State vs Fraud')
plt.xlabel('Order State')
plt.ylabel('Fraud')

#Dummy variables creation
cat_vars=['orders__orderState', 'paymentMethods__paymentMethodRegistrationFailure', 'paymentMethods__paymentMethodType', 
            'paymentMethods__paymentMethodProvider', 'paymentMethods__paymentMethodIssuer', 
            'transactions__transactionFailed']
 
for col in cat_vars:
    data1[col] = data1[col].astype('category')
data1.dtypes

data2=pd.get_dummies(data1, columns=['orders__orderState', 'paymentMethods__paymentMethodRegistrationFailure', 'paymentMethods__paymentMethodType', 
            'paymentMethods__paymentMethodProvider', 'paymentMethods__paymentMethodIssuer', 
            'transactions__transactionFailed'], drop_first=True)
data2.columns    
data2=data2.drop(['customer__customerEmail', 'customer__customerPhone',
       'customer__customerDevice', 'customer__customerIPAddress',
       'customer__customerBillingAddress', 'orders__orderShippingAddress'],axis=1)
    
# Normalization of Numeric variables
data2['Amount'] = np.log(data2['Amount']+1)

#Defining X,Y and train,test set

X=data2.iloc[:,1:].values
Y=data2.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Defining first Model - Logistic Reg
logreg = LogisticRegression()
rfe = RFE(logreg, 5)
fit=rfe.fit(X_train, y_train)

print("Num Features:",fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)

#best feautures are = Transaction amount, payments method issuer_X,payments method issuer_e,payments method issuer_c,payments method issuer_B
#Defining second Model - RF
n_estimators=range(100,1000,100)
hyper2={'n_estimators':n_estimators}
kfold=KFold(n_splits=10,random_state=0)
gd2=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper2,verbose=True,cv=kfold)
gd2.fit(X_train,y_train)
print(gd2.best_score_)
print(gd2.best_params_)
print(gd2.best_estimator_)

# Running the best model- RF
rf=RandomForestClassifier(random_state=0,n_estimators=700)
rf.fit(X_train, y_train)
print(rf.feature_importances_)

#Predictions
y_pred = rf.predict(X_test)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
"""Close to 70% accuracy"""

# checking accuracy
from sklearn.metrics import confusion_matrix   #confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report   # classification report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score         #Roc-AUC
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#Saving
joblib.dump(logreg, 'model.pkl')
pipe = joblib.load('model.pkl')
