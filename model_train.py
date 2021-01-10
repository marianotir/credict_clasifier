# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:19:07 2021

@author: Mariano
"""

#----------------------
# Load Libraries
#----------------------

import pandas as pd 
import numpy as np
import matplotlib as pl


#----------------------
# Load data
#----------------------

df = pd.read_csv('C:/Users/Mariano/DS_Models/St_Class_Skit_Exp/original.csv')  


#-------------------------
# Preprocess Data
#-------------------------

# Check consistency of duplicates
duplicates = len(df[df.duplicated() == True])
if(duplicates>0):
    print(" Duplicates founds and deleted ")
    df = df[df.duplicated() == False]
    
# drop columns with the same value
df = df.drop(df.std()[(df.std() == 0)].index, axis=1)

# drop id column
df.drop('clientid', axis=1,inplace=True)

# description
description = df.describe()
print(description)

# Change negative values 
df.loc[df.age<0, 'age'] = df[df.age<0].age*(-1)

# Impute na values as mean
from sklearn.impute import SimpleImputer

imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_fit = imp_num.fit_transform(df)
df = pd.DataFrame(imp_fit,columns=df.columns)

# Impute 0 values as mean 
df.loc[df.age.isnull()==True,'age'] = df[df.age.isnull()!=True].age.mean()

#----------------------
# Prepare data
#----------------------

# Split target and feature variables
X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

# Split for training 
from sklearn.model_selection import train_test_split

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33)


#----------------------
# Train model
#----------------------
from sklearn.ensemble import RandomForestClassifier

# Model results before umbalance

model_rf = RandomForestClassifier()
model_rf.fit(X_train,Y_train)

pred_train = model_rf.predict(X_train)
pred       = model_rf.predict(X_test)

# Modeol results after unbalance

  # Checking accuracy
from sklearn.metrics import accuracy_score
print('accuracy_score =',accuracy_score(Y_test, pred))

  # f1 score
from sklearn.metrics import f1_score
print('f1_score =',f1_score(Y_test, pred))

  # recall score
from sklearn.metrics import recall_score
print('recall_score =',recall_score(Y_test, pred))

# Solve umbalance 
# Applky Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=27)
X_train, Y_train = sm.fit_sample(X_train, Y_train)

# Modeol results after unbalance

model_rf = RandomForestClassifier(n_estimators = 50, random_state = 0)
model_rf.fit(X_train,Y_train)

pred_train = model_rf.predict(X_train)
pred       = model_rf.predict(X_test)

  # Checking accuracy
from sklearn.metrics import accuracy_score
print('accuracy_score =',accuracy_score(Y_test, pred))

  # f1 score
from sklearn.metrics import f1_score
print('f1_score =',f1_score(Y_test, pred))

  # recall score
from sklearn.metrics import recall_score
print('recall_score =',recall_score(Y_test, pred))

# Note: Since random forest model is applied there is no need for normalization or scale the data 
#       bofore training

#-----------------------w
# Save model
#-----------------------
import pickle
pkl_filename = "C:/Users/Mariano/DS_Models/St_Class_Skit_Exp/model_rf_10012020.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_rf, file)
    
import joblib
joblib_file = "C:/Users/Mariano/DS_Models/St_Class_Skit_Exp/model_rf_10012020_joblib.pkl"  
joblib.dump(model_rf, joblib_file)

#--------------------------
# Load the model
#--------------------------

model = joblib.load(pkl_filename)

# Generate predictions 
income = 3000
age = 20
loan = 5000
data = {'income': income,
        'age': age,
        'loan': loan}
    
# Data for prediction output
df = pd.DataFrame(data,index=[0])
X_outsample = df.values
    
pred = model.predict(X_outsample)










