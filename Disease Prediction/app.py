import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Load trained model and preprocessing components
# --------------------------------------------------
@st.cache_resource
def load_model_and_components():
    model = joblib.load("xgb_model.pkl")
    selector = joblib.load("selector.pkl")
    le_disease = joblib.load("label_encoder.pkl")
    binary_features_columns = joblib.load("binary_features_columns.pkl")
    return model, selector, le_disease, binary_features_columns


model, selector, le_disease, binary_features_columns = load_model_and_components()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Predictor")
st.write(
    "Select the symptoms you are experiencing, and the model will predict the most likely disease."
)

# --------------------------------------------------
# Symptom Selection
# --------------------------------------------------
st.subheader("Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose one or more symptoms:",
    options=binary_features_columns
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create binary input vector
        input_vector = pd.Series(0, index=binary_features_columns)
        for symptom in selected_symptoms:
            input_vector[symptom] = 1

        # Convert to DataFrame and apply feature selection
        input_df = pd.DataFrame([input_vector])
        input_selected = selector.transform(input_df)

        # Make predictions
        prediction = model.predict(input_selected)[0]
        predicted_proba = model.predict_proba(input_selected)[0]

        # Sort probabilities (highest first)
        probabilities = sorted(
            zip(le_disease.classes_, predicted_proba),
            key=lambda x: x[1],
            reverse=True
        )

        predicted_disease = le_disease.inverse_transform([prediction])[0]

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------
        st.success(f"✅ Predicted Disease: **{predicted_disease}**")

        st.subheader("📊 Top Predictions")
        for disease, proba in probabilities[:3]:
            st.write(f"**{disease}**: {proba:.2%}")

        # Confidence warning
        if probabilities[0][1] < 0.5:
            st.info(
                "⚠️ Prediction confidence is low. "
                "Try selecting more symptoms for better accuracy."
            )

# --------------------------------------------------
# Footer / Disclaimer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ This tool is for educational purposes only and should not be used as a substitute "
    "for professional medical advice."
)
