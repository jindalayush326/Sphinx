from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from fastapi import FastAPI


app = FastAPI()
classifier_spam = joblib.load('D:/ml/practice/Classification/ServerFile/plks/classifier_spam.pkl')
vectorizer_spam = joblib.load('D:/ml/practice/Classification/ServerFile/plks/vectorizer_spam.pkl')
classifier_cc = joblib.load('D:/ml/practice/Classification/ServerFile/plks/classifier_cc.pkl')
scaler_cc = joblib.load('D:/ml/practice/Classification/ServerFile/plks/scaler_cc.pkl')

app = Flask(__name__)
CORS(app)

@app.route('/predict_spam', methods=['POST'])
def predict_spam():

    email_text = request.json['email_text']
    email_vector = vectorizer_spam.transform([email_text])
    prediction = classifier_spam.predict(email_vector.toarray())
    
    return jsonify(is_spam=bool(prediction[0]))

@app.route('/predict_cc', methods=['POST'])
def predict_cc():

    transaction_data = request.json['transaction_data']
    transaction_df = pd.DataFrame(transaction_data, index=[0])
    scaled_data = scaler_cc.transform(transaction_df)
    prediction = classifier_cc.predict(scaled_data)
    return jsonify(is_fraud=bool(prediction[0]))

if __name__ == '__main__':
    app.run(debug=False)
