import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Create Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model_classifier.pkl')
except FileNotFoundError:
    print("Error: 'model_classifier.pkl' not found. Make sure it's in the same directory as app.py.")
    exit()

# Load the TF-IDF vectorizer
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    print("Error: 'tfidf_vectorizer.pkl' not found. Make sure it's in the same directory as app.py.")
    exit()

# Define a function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route (for form submissions)
@app.route('/predict', methods=["POST"])
def predict():
    review = request.form['Review']
    cleaned_review = clean_text(review)
    vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(vec)[0]
    return render_template("index.html", prediction_text=f"The sentiment is: {prediction}")

# API route (for programmatic access)
@app.route('/sentiment', methods=["GET"])
def sentiment():
    review = request.args.get("Review")
    cleaned_review = clean_text(review)
    vec = vectorizer.transform([cleaned_review])
    prediction = model.predict(vec)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)