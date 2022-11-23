# Library imports
import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk

# Load trained Pipeline
model = joblib.load('sentimen-opini-film.pkl')

stopwords = list(STOP_WORDS)

# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
from tokenizer_input import CustomTokenizerExample

@app.route("/index")

# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
#     data = pd.DataFrame(new_review)
#     data.columns = ['new_review']

    predictions = model.predict(new_review)[0]
    if predictions== 'positive':
        return render_template('index.html', prediction_text='Positive')
    else:
        return render_template('index.html', prediction_text='Negative')
        
if __name__ == "__main__":
    app.run(debug=True, port=3000)