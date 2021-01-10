# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:40:40 2021

@author: Mariano
"""

#-------------------------------
# Import libraries
#-------------------------------

# Deploy model libraries
import streamlit as st
import joblib

# Model libraries
from sklearn.ensemble import RandomForestClassifier

# Dataframe manipulation libraries
import pandas as pd

#-------------------------------
# StreamLit Application
#-------------------------------

# Load model 
model = joblib.load('model_rf_10012020.pkl')

def main():

    # Aplication header
    st.write("""
             # Credit Risk Prediction App 
             """)
             
    # Sidebar parameters
    st.sidebar.header('User input parameters')
    
    # Parameters
    income = st.sidebar.slider('income', 15000, 40000, 80000)
    age = st.sidebar.slider('age', 18, 40, 70)
    loan = st.sidebar.slider('loan', 500, 5000, 20000)
    data = {'income': income,
            'age': age,
            'loan': loan}
    
    # Data for prediction output
    df = pd.DataFrame(data,index=[0])
    
    st.subheader(' Data input parameters')
    st.write(df)
    
    
    # Generate predictions on unseen data
    X_outsample = df.values
    
    predictions = model.predict(X_outsample)
    
    prediction_output = predictions[0]

    if prediction_output > 0:
        prediction_output = 'Default probable'
    else: 
        prediction_output = 'Default not probable'
    st.subheader(' Prediction Output')
    st.write(prediction_output)
    
    
if __name__=='__main__':
    main()












