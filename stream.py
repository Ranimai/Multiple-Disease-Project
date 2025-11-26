import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

with open('liver_model.pkl', 'rb') as f:
    l_model = pickle.load(f)

with open('kidney_model.pkl', 'rb') as f:
    k_model = pickle.load(f)

with open('parkinson_model.pkl', 'rb') as f:
    p_model = pickle.load(f)

st.markdown("<h1 style='text-align: center;'>🩺 Multiple Disease Prediction</h1>", unsafe_allow_html=True)

st.divider()

# df_liver = pd.read_csv("F:/Rani/Multiple Disease/indian_liver_patient - indian_liver_patient.csv")
# st.write(df_liver.head(10))

# menu = option_menu(None,['Liver Disease', 'Kidney Disease', 'Parkinsons Disease'], orientation = 'horizontal')

menu = ['Liver Disease', 'Kidney Disease', 'Parkinsons Disease']
choice = st.sidebar.selectbox('⚙️ Select disease', menu)

# Liver Disease......

if choice == "Liver Disease":
    st.header("🫁 Liver Disease Prediction")

    df_liver = pd.read_csv("F:/Rani/Multiple Disease/disease_venv/Liver_cleaned_data.csv")
    st.write(df_liver.head(10))

    features = l_model['feature_names']
    input = []

    for col in features:
        if col == 'Gender':
            gender = st.selectbox('Gender', ['Male', 'Female'])
            value = 1 if gender == 'Male' else 0
        else:
            value = st.number_input(f"Enter {col}", value=0.0)
        input.append(value)

    if st.button("🔍 Predict Liver Disease"):
        x = l_model['scaler'].transform([input])
        pred = l_model['model'].predict(x)[0]

        st.success("🫁 Disease Detected" if pred ==1 else '🫁 No Disease')

# Kidney Disease.............

elif choice == "Kidney Disease":
    st.header("🧪 Kidney Disease Prediction")

    df_kidney = pd.read_csv("F:/Rani/Multiple Disease/disease_venv/Kidney_cleaned_data.csv")
    st.write(df_kidney.head(10))

    features = k_model['feature_names']
    encoders = k_model['encoders']
    input = {}

    for col in features:
        if col in encoders:
            value = st.selectbox(f"{col}", encoders[col].classes_)
            value = encoders[col].transform([value])[0]
            input[col] = value
        else:
            value = st.number_input(f"{col}", value=0.0)
            input[col] = value

    if st.button("🔍 Predict Kidney Disease"):
        orders = [input[col] for col in features]
        x = k_model['scaler'].transform([orders])
        pred = k_model['model'].predict(x)[0]

        st.success("🧪 CKD Detected" if pred ==1 else "🧪 NO CKD")

# Parkinsons Disease.......

else:
    st.header("🧠 Parkinsons Diseases Predicted")

    df_park = pd.read_csv("F:/Rani/Multiple Disease/disease_venv/Parkinsion_cleaned_data.csv")
    st.write(df_park.head(10))

    features = p_model['feature_names']
    input = []

    for col in features:
        value = st.number_input(f"{col}", value=0.0)
        input.append(value)

    if st.button("🔍 Predict Parkinsons Disease"):
        x = p_model['scaler'].transform([input])
        pred = p_model['model'].predict(x)[0]

        st.success("🧠 Parkinson Detected" if pred==1 else "🧠 No Parkinson")


        

        