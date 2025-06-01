import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("LoanApprovalPrediction.csv")



df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())


df = df.drop('Loan_ID', axis=1)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for col in ['Education', 'Self_Employed', 'Property_Area', 'Gender', 'Married', 'Loan_Status']:
    df[col] = label.fit_transform(df[col])


df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['EMI_Ratio'] = df['LoanAmount'] / df['Total_Income']


df = df.drop(['Gender', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome'], axis=1)


X = df.drop(['Loan_Status','Total_Income','LoanAmount'], axis=1)
y = df['Loan_Status']

print(df.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# print(accuracy_score(y_pred,y_test))

import joblib
joblib.dump(rf,'rfmodel.pkl')



