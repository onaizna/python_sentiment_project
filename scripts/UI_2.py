import os
import sys
import streamlit as st
import pickle
from pathlib import Path # Adds the parent directory to sys.path

sys.path.append(os.path.abspath('..')) 
try:
    from src import config 
    MODELS_DIR = Path(config.MODELS_PATH) # Using pathlib for the paths
    MODEL_FILE = MODELS_DIR / "random_forest.pickle"
    VECTORIZER_FILE = MODELS_DIR / "vectorizer.pickle"
except ImportError:
    st.error("Error: couldn't import 'config' from 'src' directory. Check the project structure and that 'src/config.py' exists.")
    st.stop() # stops the script execution if config isn't found
except AttributeError:
    st.error("Error: the variable 'MODELS_PATH' isn't defined in the 'config.py' file.")
    st.stop()

# --- Loading Function with Caching and Error Management ---
@st.cache_resource # Cache for complex objects like models/vectorizers
def load_pickle_object(file_path: Path):
    """Loads an object from a pickle file."""
    if not file_path.is_file():
        st.error(f"Error: couldn't find file: {file_path}")
        st.stop() # It stops if the file doesn't exist
        return None # Returns None (even if st.stop() should anticipate that)
    try:
        with open(file_path, "rb") as file:
            loaded_object = pickle.load(file)
        return loaded_object
    except pickle.UnpicklingError:
        st.error(f"Error: couldn't decode the pickle file: {file_path}. It could be corrupted or in an incompatible version.")
        st.stop()
    except Exception as e:
        st.error(f"Error: unexpected error when loading the file {file_path}: {e}")
        st.stop()

# Loading model and vectorizer using the functions with cache
model = load_pickle_object(MODEL_FILE)
vectorizer = load_pickle_object(VECTORIZER_FILE)

# --- UI Streamlit ---
st.title("Sentiment Analysis of the Test")

# Using st.form to enable enter with Enter
with st.form(key="classification_form"):
    user_input = st.text_area("Insert the test you want to classify:", height=150)
    submitted = st.form_submit_button("Classify Test")

    if submitted:
        if not user_input or user_input.strip() == "":
            st.warning("Please, insert some test before pressing 'Classify Test'.")
        else:
            try:
                # 1. Trasform the input using the loaded vectorizer
                X_transformed = vectorizer.transform([user_input])
                
                # 2. Execute the prediction using the loaded model
                prediction = model.predict(X_transformed)[0] 
                # Catchs the first element if predict returns a list

                # 3. Shows the output
                st.write("--- Result ---") # Visual separator
                if prediction == "positive": # Be sure of the labels returned by your model
                    st.success(f"**Sentiment Predicted:** Positive 😊")
                elif prediction == "negative":
                    st.error(f"**Sentiment Predicted:** Negative 😞")
                else:
                    # Manage unexpected outputs from the model
                    st.warning(f"Unrecognized output of the model: {prediction}")

            except Exception as e:
                st.error(f"An error has occurred during the classification: {e}")
                st.warning("Make sure your model and vectorizer are compatible with the input provided.")

st.markdown("---") 
st.caption("Insert a phrase or a paragraph and click 'Classify Test' for analyze the sentiment.")