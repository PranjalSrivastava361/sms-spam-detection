# SMS Spam Detection using NLP and ML

A machine learning pipeline for SMS classification that achieves **~96.7% accuracy** on 5k+ labeled messages.

## Features
- Text preprocessing (tokenize, stopwords, lemmatize)
- TF-IDF (unigrams + bigrams)
- Model comparison (SVM, Naive Bayes, Logistic Regression)
- Flask API for inference; deployed on Heroku

## Setup (WSL / Linux / Mac)

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
