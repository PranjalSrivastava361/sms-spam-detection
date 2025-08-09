
**train.py** (minimal training script)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    # UCI sms dataset columns v1 (label), v2 (text)
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='spam.csv')
    args = parser.parse_args()

    df = load_data(args.data)
    X = df['text']
    y = df['label'].map({'ham':0, 'spam':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', LinearSVC())
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, 'model.pkl')

if __name__ == '__main__':
    main()

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text','')
    pred = model.predict([text])[0]
    label = 'spam' if pred==1 else 'ham'
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
