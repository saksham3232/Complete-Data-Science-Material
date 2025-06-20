import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5', compile=False)

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User input with step size and validation
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)

age = st.number_input('Age', min_value=18, max_value=92, value=30, step=1, placeholder="Enter age (18-92)")
balance = st.number_input('Balance', min_value=0.0, value=0.0, step=1000.0, format="%.2f", placeholder="Enter balance")
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600, step=10, placeholder="Enter credit score (300-900)")
exited= st.selectbox('Exited', [0, 1])
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=1, step=1, placeholder="Enter tenure (0-10)")
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1, step=1, placeholder="Enter number of products (1-4)")
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One Hot Encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction of estimated salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]
 
st.write(f'Predicted estimated salary: {predicted_salary:.2f}')