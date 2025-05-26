# IMDB Movie Review Sentiment Analysis

## Overview
This project develops an end-to-end Natural Language Processing (NLP) pipeline for classifying movie reviews from the **TensorFlow IMDB dataset** as either positive or negative. It showcases the complete machine learning workflow, from data preprocessing and model training using a Simple Recurrent Neural Network (RNN) to interactive deployment via a web application.

## Project Highlights
* **Dataset Utilization**: Leverages the widely-used **TensorFlow IMDB dataset** for binary sentiment classification.
* **Deep Learning Model**: Implements a Simple RNN architecture for robust sequence modeling, ideal for text data.
* **Data Preprocessing**: Includes efficient handling of text data, such as tokenization, word-to-index mapping, and sequence padding to ensure uniform input length for the neural network.
* **Model Training & Optimization**: Employs `EarlyStopping` callback during training to prevent overfitting and optimize model performance.
* **Interactive Deployment**: Features a user-friendly Streamlit web application, served by `app.py`, that allows real-time sentiment prediction for custom movie reviews.

## Live Application
Experience the sentiment analysis model in action! The Streamlit application is deployed and accessible at:
[https://imdbsentimentanalysis-bkk5r4ftmebrq46pkyday5.streamlit.app/](https://imdbsentimentanalysis-bkk5r4ftmebrq46pkyday5.streamlit.app/)

## Key Technologies
* **Python**
* **TensorFlow / Keras**: For building and training the deep learning model (Simple RNN).
* **NumPy**: For numerical operations and array manipulation.
* **Pandas**: For data handling and analysis.
* **Streamlit**: For creating the interactive web application.
* **TensorBoard**: For visualizing training metrics and model graphs.

## Model Performance
The Simple RNN model was trained and validated, achieving an accuracy of approximately **75%** on unseen IMDB movie reviews.

## Project Structure
* `imdb_sentiment_analysis.ipynb`: Jupyter Notebook containing the full data loading, preprocessing, model definition, training, evaluation, and saving of the Simple RNN model.
* `app.py`: Python script for the Streamlit web application that loads the trained model and provides an interface for sentiment prediction.
* `requirements.txt`: Lists all necessary Python dependencies for setting up the project environment.
* `simple_rnn_imdb.keras`: The saved Keras model file generated after training, used by the Streamlit application for predictions.
