# IMDB Movie Review Sentiment Analysis

## Overview
This project develops an end-to-end Natural Language Processing (NLP) pipeline for classifying movie reviews from the IMDB dataset as either positive or negative. It showcases the complete machine learning workflow, from data preprocessing and model training using a Simple Recurrent Neural Network (RNN) to interactive deployment via a web application.

## Project Highlights
* **Dataset Utilization**: Leverages the widely-used IMDB movie review dataset for binary sentiment classification[cite: 1].
* **Deep Learning Model**: Implements a Simple RNN architecture for robust sequence modeling, ideal for text data[cite: 1].
* **Data Preprocessing**: Includes efficient handling of text data, such as tokenization, word-to-index mapping, and sequence padding to ensure uniform input length for the neural network[cite: 1].
* **Model Training & Optimization**: Employs `EarlyStopping` callback during training to prevent overfitting and optimize model performance[cite: 1].
* **Interactive Deployment**: Features a user-friendly Streamlit web application that allows real-time sentiment prediction for custom movie reviews[cite: 2].

## Key Technologies
* **Python**
* **TensorFlow / Keras**: For building and training the deep learning model (Simple RNN)[cite: 1, 3].
* **NumPy**: For numerical operations and array manipulation[cite: 1, 3].
* **Pandas**: For data handling and analysis[cite: 2, 3].
* **Streamlit**: For creating the interactive web application[cite: 2, 3].
* **TensorBoard**: For visualizing training metrics and model graphs[cite: 1, 3].

## Model Performance
The Simple RNN model was trained and validated, achieving a validation accuracy of approximately **75%** on unseen IMDB movie reviews[cite: 1].

## Project Structure
* `imdb_sentiment_analysis.ipynb`: Jupyter Notebook containing the full data loading, preprocessing, model definition, training, evaluation, and saving of the Simple RNN model[cite: 1].
* `app.py`: Python script for the Streamlit web application that loads the trained model and provides an interface for sentiment prediction[cite: 2].
* `requirements.txt`: Lists all necessary Python dependencies for setting up the project environment[cite: 3].
* `simple_rnn_imdb.keras`: The saved Keras model file generated after training, used by the Streamlit application for predictions.
