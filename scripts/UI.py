import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path
from src import config
import streamlit as st
import pickle

# Load the model and vectorizer
with open(f"{config.MODELS_PATH}random_forest.pickle", "rb") as file:
    model = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Text Classification")

# Text input
user_input = st.text_area("Enter text to classify", "")

# Predict when button is clicked
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input to predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        if prediction == "positive":
            st.success(f"Predicted class: {prediction}")
        elif prediction == "negative":
            st.error(f"Predicted class: {prediction}")