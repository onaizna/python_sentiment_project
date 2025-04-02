import os
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..')) 
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src import config
except ImportError:
    st.error("Couldn't import 'config' from 'src'. Be sure that the project structure and the sys.path are correct.")
    st.stop() # It stop if it can't import config

import streamlit as st
import pickle
import logging 

# Configure logging base
logging.basicConfig(level=logging.INFO)

# Configuring Models
AVAILABLE_MODELS = {
    "Random Forest": "random_forest.pickle",
    "Logistic Regression": "logistic_regression.pickle"
    # You can add other models here
    # "User Model Name": "name_model_file.pickle"
}
VECTORIZER_FILENAME = "vectorizer.pickle" 

# Loading functions with Caching
# Use Streamlit caching to avoid reloading the file at every iteration

@st.cache_resource 
def load_pickle_file(filepath):
    """Loads a pickle  file from a specific path."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        st.error(f"Error: the file wasn't found: {os.path.basename(filepath)}")
        return None
    try:
        with open(filepath, "rb") as file:
            content = pickle.load(file)
        logging.info(f"File loaded successfully: {filepath}")
        return content
    except Exception as e:
        logging.error(f"Error while loading the file {filepath}: {e}", exc_info=True)
        st.error(f"Error while loading the file {os.path.basename(filepath)}.")
        return None

# Loading Objects
vectorizer_path = os.path.join(config.MODELS_PATH, VECTORIZER_FILENAME)
vectorizer = load_pickle_file(vectorizer_path)

# Streamlit Interface
st.title("Classify Text")

# Model Selection
selected_model_name = st.selectbox(
    "Choose which model you want to use:",
    options=list(AVAILABLE_MODELS.keys()) # Shows the readable names
)

# Loads the model selected dinamically
model = None
if selected_model_name:
    model_filename = AVAILABLE_MODELS[selected_model_name]
    model_path = os.path.join(config.MODELS_PATH, model_filename)
    model = load_pickle_file(model_path) # The function manages the caching and the errors

# User Input
user_input = st.text_area("Insert text to classify:", "")

# Classification Button and Logic
if st.button("Classify"):
    # Controlli preliminari
    if vectorizer is None:
        st.warning("The Vectorizer wasn't loaded correctly. Impossible to classify.")
    elif model is None:
        st.warning(f"The model '{selected_model_name}' wasn't loaded correctly. Impossible to classify.")
    elif user_input.strip() == "":
        st.warning("Please, insert text.")
    else:
        try:
            # 1. Trasforms the input using the loaded vectorizer 
            X_transformed = vectorizer.transform([user_input])

            # 2. Executes the prediction using the loaded and selected model
            prediction = model.predict(X_transformed)[0]
            
            # (Optional) Shows the probabilities if the model support them
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_transformed)[0]
                # Finds the predicted class index and its confidence
                try:
                    # model.classes_ gives you the order of the classes in the probabilities
                    predicted_class_index = list(model.classes_).index(prediction) 
                    confidence = probabilities[predicted_class_index]
                    st.write(f"Confidence: {confidence:.2%}")
                except ValueError:
                    st.write("Impossible to determine the confidence (class not found).")
                except Exception as prob_e:
                    st.write(f"Error during the confidence evaluation: {prob_e}")


            # 3. Shows the output to the user
            st.write(f"Modello utilizzato: **{selected_model_name}**")
            # Adapts these conditions to the actual names of your classes ('positive', 'negative', ecc.)
            if prediction == "positive":
                st.success(f"Predicted Class: **{str(prediction).upper()}** 👍")
            elif prediction == "negative":
                st.error(f"Predicted Class: **{str(prediction).upper()}** 👎")
            else: # Manages other possible predicted classes
                 st.info(f"Predicted Class: **{str(prediction)}**")

        except Exception as e:
            logging.error(f"Error during the prediction: {e}", exc_info=True)
            st.error(f"An error has occurred during the classification: {e}")

# Add a note for the user
st.caption(f"Be sure that the model file ({', '.join(AVAILABLE_MODELS.values())}) and the file {VECTORIZER_FILENAME} are in {config.MODELS_PATH}")