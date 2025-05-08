# Sentiment Analysis of Customer Reviews

## Overview

This project performs sentiment analysis on customer reviews to classify them as positive, neutral, or negative. It includes data preprocessing, feature extraction using TF-IDF, model training, and a Flask web application to interact with the trained model.

## Dataset

The dataset used for this project is the 'Amazon Fine Food Reviews' dataset, which can be downloaded from [kaggle kernels amazon-fine-food-reviews - https://www.kaggle.com/code/chirag9073/amazon-fine-food-reviews-sentiment-analysis?select=Reviews.csv].

## Libraries Used

- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn
- Flask
- joblib

## Setup Instructions

1.  Clone the repository.
2.  Install the required libraries.
3.  Download the dataset `Reviews.csv`
4.  Ensure the trained model files (`model_classifier.pkl` and `tfidf_vectorizer.pkl`) are in the same directory as `app.py`.

## Project Structure

yash_sentiment_analysis_amazon_review/
├── EDA_And_Model_Preparation.ipynb        <- Jupyter Notebook (EDA and Model)
├── app.py                     <- Flask application code
├── templates/
│   └── index.html             <- HTML template for the web app
├── model_classifier.pkl       <- Trained sentiment analysis model
├── tfidf_vectorizer.pkl       <- Trained TF-IDF vectorizer
├── requirements.txt           <- List of Python dependencies
└── README.md                  <- Project documentation 

## Running the Flask App

1.  Navigate to the project directory in your terminal.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and go to `http://127.0.0.1:5000/` to interact with the sentiment analysis tool.
4.  You can also access the API endpoint for sentiment prediction at `http://127.0.0.1:5000/sentiment?Review=[your_review_here]`.

## Steps Taken (in `EDA_And_Model_Preparation.ipynb`)

1.  **Data Loading and Exploration:** The dataset was loaded using pandas, and initial exploration included inspecting the data structure and checking for missing values.
2.  **Data Cleaning and Preprocessing:** Text data was cleaned by converting it to lowercase, removing punctuation and numbers, tokenizing, removing stopwords, and lemmatizing words. Missing values were handled by dropping rows with missing 'Summary' or 'ProfileName'.
3.  **Sentiment Labeling:** A 'Sentiment' column was created based on the 'Score' column, categorizing reviews as 'Positive' (score > 3), 'Neutral' (score = 3), or 'Negative' (score < 3).
4.  **Exploratory Data Analysis (EDA):** The distribution of sentiments was visualized using a bar plot, and the percentage breakdown of sentiments was calculated.
5.  **Feature Engineering:** The cleaned text data was transformed into numerical features using TF-IDF vectorization.
6.  **Model Training:** A Logistic Regression model was trained on the TF-IDF vectorized data to classify the sentiment of reviews.
7.  **Model Evaluation:** The model's performance was evaluated using a classification report and accuracy score.
8.  **Model Saving:** The trained model and TF-IDF vectorizer were saved as `.pkl` files for use in the Flask application.

## Insights

Our sentiment analysis indicates that positive reviews often emphasize the "great taste" and "freshness" of the food items. Customers appreciate the convenience of online ordering and the wide selection available. Conversely, negative reviews tend to center around issues like "poor packaging," "short expiration dates," and "inconsistent product quality." A subset of negative reviews also expresses frustration with "slow delivery times" and "unresponsive customer service." These findings suggest that while the core product attributes are strengths, operational inefficiencies are detracting from the customer experience.
