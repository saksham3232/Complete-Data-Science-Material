import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('8-Deep Learning/Simple RNN/imdb_rnn_model.keras', compile=False)

# Helper function to preprocess input text
def preprocess_text(text, vocab_size=VOCAB_SIZE):
    words = text.lower().split()
    # Encode only words within vocab limit; unknown words get token 2
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    encoded_review = [i for i in encoded_review if i < vocab_size]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

# Initialize session state variables
for key in ["sentiment", "score", "review_input"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# Callback to clear input
def clear_input():
    st.session_state.review_input = ""
    st.session_state.sentiment = ""
    st.session_state.score = None

# App Title
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review with **at least 25 words** to classify it as Positive or Negative.")

# Text input
st.text_area("Movie Review", key="review_input", height=150, placeholder="Type your movie review here...")

# Live word count
word_count = len(st.session_state.review_input.strip().split())
st.write(f"**Word count:** {word_count} (minimum 25 required)")

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Classify"):
        review = st.session_state.review_input.strip()
        if word_count < 25:
            st.warning("🚫 Please enter at least 25 words to classify.")
        else:
            try:
                preprocessed_input = preprocess_text(review)
                prediction = model.predict(preprocessed_input)
                st.session_state.score = float(prediction[0][0])
                st.session_state.sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

with col2:
    st.button("Clear", on_click=clear_input)

# Output
if st.session_state.sentiment:
    st.success(f"**Sentiment:** {st.session_state.sentiment}")
    st.write(f"**Prediction Score:** {st.session_state.score:.2f}")
elif not st.session_state.review_input.strip():
    st.info("✍️ Please enter a movie review.")