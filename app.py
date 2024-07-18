from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model, scaler, and encoders
model = joblib.load('models/lightgbm_aml_model.pkl')
scaler = joblib.load('models/scaler.pkl')

label_encoders = {
    'Payment_currency': joblib.load('models/Payment_currency_label_encoder.pkl'),
    'Received_currency': joblib.load('models/Received_currency_label_encoder.pkl'),
    'Sender_bank_location': joblib.load('models/Sender_bank_location_label_encoder.pkl'),
    'Receiver_bank_location': joblib.load('models/Receiver_bank_location_label_encoder.pkl'),
    'Payment_type': joblib.load('models/Payment_type_label_encoder.pkl')
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features = [
        data['Amount'],
        data['Payment_currency'],
        data['Received_currency'],
        data['Sender_bank_location'],
        data['Receiver_bank_location'],
        data['Payment_type']
    ]

    for i, column in enumerate(['Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type']):
        features[i + 1] = label_encoders[column].transform([features[i + 1]])[0]

    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return jsonify({
        'prediction': int(prediction),
        'probability': probabilities.tolist()
    })

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        
        # Verify the required columns are present
        required_columns = ['Amount', 'Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type']
        if not all(column in df.columns for column in required_columns):
            return jsonify({'error': f'Missing one or more required columns: {required_columns}'}), 400

        # Encode categorical features
        for column in ['Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type']:
            df[column] = label_encoders[column].transform(df[column])

        features = scaler.transform(df[required_columns])
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        # Prepare the response
        results = []
        for prediction, probability in zip(predictions, probabilities):
            results.append({
                'prediction': int(prediction),
                'probability': probability.tolist()
            })
        
        return jsonify(results)
    else:
        return jsonify({'error': 'Allowed file types are csv'}), 400

if __name__ == '__main__':
    app.run(debug=True)
