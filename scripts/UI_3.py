import os
import sys
import streamlit as st
import pickle
from pathlib import Path
# Assuming your vectorizer and model are compatible with sklearn's API
# Import base classes for type hinting if possible (optional but good practice)
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    ModelType = BaseEstimator
    VectorizerType = TransformerMixin
except ImportError:
    # Fallback to 'any' if sklearn is not installed in the environment
    # where you might be reviewing this script, though it's needed at runtime.
    ModelType = "typing.Any"
    VectorizerType = "typing.Any"
import typing # For fallback type hinting

# --- Configuration and Path Setup ---
# Adds the parent directory to sys.path - Keep this if your structure requires it
sys.path.append(os.path.abspath('..'))

try:
    # It's generally recommended to have src/config.py accessible
    # without modifying sys.path if possible (e.g., by running streamlit
    # from the parent directory of 'src' and 'scripts')
    from src import config
    MODELS_DIR = Path(config.MODELS_PATH) # Using pathlib for paths
    MODEL_FILENAME = "random_forest.pickle" # Define filenames separately
    VECTORIZER_FILENAME = "vectorizer.pickle"
    MODEL_FILE = MODELS_DIR / MODEL_FILENAME
    VECTORIZER_FILE = MODELS_DIR / VECTORIZER_FILENAME
except ImportError:
    st.error(
        "Error: Could not import 'config' from 'src' directory. "
        "Please check your project structure and ensure 'src/config.py' exists "
        "and the script is run from the correct location."
    )
    st.stop() # Stop script execution if config is missing
except AttributeError:
    st.error(
        "Error: The variable 'MODELS_PATH' is not defined in your 'config.py' file."
    )
    st.stop() # Stop script execution if config variable is missing

# Define expected sentiment labels
POSITIVE_LABEL = "positive"
NEGATIVE_LABEL = "negative"

# --- Loading Function with Caching and Error Handling ---
@st.cache_resource # Cache resource for complex objects like models/vectorizers
def load_pickle_object(file_path: Path) -> typing.Any:
    """Loads a Python object from a pickle file."""
    if not file_path.is_file():
        st.error(f"Error: File not found at {file_path}")
        st.stop() # Stop execution if file doesn't exist
        # The return below is technically unreachable due to st.stop(),
        # but linters might appreciate it.
        return None
    try:
        with open(file_path, "rb") as file:
            loaded_object = pickle.load(file)
        return loaded_object
    except pickle.UnpicklingError:
        st.error(
            f"Error: Could not unpickle file: {file_path}. "
            "It might be corrupted or saved with an incompatible Python/library version."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error: An unexpected error occurred loading file {file_path}: {e}")
        st.stop()

# --- Load Model and Vectorizer ---
# Add type hints for loaded objects if possible
model: ModelType = load_pickle_object(MODEL_FILE)
vectorizer: VectorizerType = load_pickle_object(VECTORIZER_FILE)

# --- Prediction Function ---
def get_sentiment_prediction(text: str, vect: VectorizerType, mod: ModelType) -> tuple[str, typing.Optional[float]]:
    """
    Analyzes the sentiment of the input text.

    Args:
        text: The text to analyze.
        vect: The fitted vectorizer instance.
        mod: The trained model instance.

    Returns:
        A tuple containing:
        - The predicted label (str).
        - The prediction probability (float, or None if not available).
    """
    # 1. Transform input using the loaded vectorizer
    transformed_text = vect.transform([text])

    # 2. Predict using the loaded model
    prediction = mod.predict(transformed_text)[0]

    # 3. Try to get prediction probabilities
    probability = None
    if hasattr(mod, "predict_proba"):
        try:
            probabilities = mod.predict_proba(transformed_text)[0]
            # Find the probability of the predicted class
            # Ensure class indices match the order in model.classes_
            classes = list(mod.classes_)
            if prediction in classes:
                 pred_index = classes.index(prediction)
                 probability = probabilities[pred_index]
            else:
                # This case should ideally not happen if prediction comes from the model
                st.warning(f"Predicted class '{prediction}' not found in model's known classes: {classes}")

        except Exception as e:
            st.warning(f"Could not retrieve prediction probability: {e}")
            # Proceed without probability if predict_proba fails

    return prediction, probability


# --- Streamlit UI ---
st.title("Text Sentiment Analysis")

# Using st.form to batch input and submission
with st.form(key="classification_form"):
    user_input = st.text_area("Enter the text you want to classify:", height=150, key="text_input")
    submitted = st.form_submit_button("Classify Text")

    if submitted:
        if not user_input or user_input.strip() == "":
            st.warning("Please enter some text before classifying.")
        else:
            try:
                # Show spinner during processing
                with st.spinner("Analyzing sentiment..."):
                    prediction, probability = get_sentiment_prediction(user_input, vectorizer, model)

                # Display results
                st.write("--- Result ---") # Visual separator

                confidence_text = ""
                if probability is not None:
                    # Format probability as percentage
                    confidence_text = f"(Confidence: {probability*100:.2f}%)"

                if prediction == POSITIVE_LABEL:
                    st.success(f"**Sentiment Predicted:** Positive 😊 {confidence_text}")
                elif prediction == NEGATIVE_LABEL:
                    st.error(f"**Sentiment Predicted:** Negative 😞 {confidence_text}")
                else:
                    # Handle unexpected output labels from the model
                    st.warning(f"Model returned an unexpected label: '{prediction}' {confidence_text}")

            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
                st.warning("Ensure the model and vectorizer are compatible with the input provided.")

st.markdown("---") # Visual separator
st.caption("Enter a sentence or paragraph and click 'Classify Text' to analyze its sentiment.")