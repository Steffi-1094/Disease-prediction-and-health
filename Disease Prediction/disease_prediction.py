import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and other components
@st.cache_resource
def load_model_and_components():
    model = joblib.load('xgb_model.pkl')
    selector = joblib.load('selector.pkl')
    le_disease = joblib.load('label_encoder.pkl')
    binary_features_columns = joblib.load('binary_features_columns.pkl')
    return model, selector, le_disease, binary_features_columns

# Load the resources
model, selector, le_disease, binary_features_columns = load_model_and_components()

# Streamlit UI
st.title("Disease Predictor")
st.write("Enter the symptoms to predict the disease.")

# Input Symptoms
all_symptoms = binary_features_columns
selected_symptoms = st.multiselect("Select Symptoms", all_symptoms)

if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create input vector
        input_vector = pd.Series(0, index=binary_features_columns)
        for symptom in selected_symptoms:
            if symptom in input_vector.index:
                input_vector[symptom] = 1
        input_selected = selector.transform([input_vector])

        # Make predictions
        prediction = model.predict(input_selected)[0]
        predicted_proba = model.predict_proba(input_selected)[0]
        probabilities = {le_disease.inverse_transform([i])[0]: prob for i, prob in enumerate(predicted_proba)}

        # Display results
        predicted_disease = le_disease.inverse_transform([prediction])[0]
        st.success(f"Predicted Disease: {predicted_disease}")
        st.subheader("Prediction Probabilities:")
        for disease, proba in probabilities.items():
            st.write(f"{disease}: {proba:.2f}")
