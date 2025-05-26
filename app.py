# Import necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.keras')

# Helper functions
## Decode reviews
def decode_review(encoded_review):
  return ''.join([reverse_word_index.get(i-3,'?') for i in encoded_review])
## Preprocess user input
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# Prediction function
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)
  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to predict its sentiment (Positive/Negative):')
# Sample reviews for demonstration
sample_reviews = [
    "This movie was fantastic! I loved every moment of it.",
    "Terrible movie, I regret watching it.",
    "An average film, nothing special but not bad either.",
    "Absolutely amazing! A must-watch for everyone.",
    "I didn't like the plot, it was too predictable."
]
st.write('### Sample Reviews:')
for review in sample_reviews:
    st.write(f'- {review}')
user_input = st.text_area('Review Text', height=200)
if st.button('Predict'):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Confidence: {confidence:.2f}')
    else:
        st.write('Please enter a review to analyze.')