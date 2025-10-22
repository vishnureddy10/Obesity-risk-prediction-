from flask import Flask, request, render_template, jsonify, make_response, send_from_directory
import pickle
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scaler
try:
    with open('obesity_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file) 
    with open('obesity_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    app.logger.error(f"Error loading model/scaler: {str(e)}")
    raise

# Define categorical feature mappings
categorical_mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'SMOKE': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'MTRANS': {'Public_Transportation': 0, 'Walking': 1, 'Automobile': 2, 'Motorbike': 3, 'Bike': 4}
}

@app.route('/', methods=['GET'])
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

# Override static file serving to disable caching
@app.route('/static/<path:filename>')
def static_files(filename):
    response = send_from_directory('static', filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")

        # Validate required fields
        required_fields = [
            'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS',
            'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Process categorical features
        categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        categorical_data = []
        for feature in categorical_features:
            value = data[feature]
            if value not in categorical_mappings[feature]:
                return jsonify({'error': f'Invalid value for {feature}: {value}'}), 400
            categorical_data.append(categorical_mappings[feature][value])

        # Process numerical features
        numerical_fields = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        numerical_data = []
        for field in numerical_fields:
            try:
                value = float(data[field])
                numerical_data.append(value)
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid numerical value for {field}'}), 400

        # Calculate BMI
        try:
            bmi = float(data['Weight']) / (float(data['Height']) ** 2)
        except (ValueError, ZeroDivisionError):
            return jsonify({'error': 'Invalid Height or Weight for BMI calculation'}), 400
        numerical_data.append(bmi)

        # Combine features
        features = categorical_data + numerical_data
        app.logger.debug(f"Raw features: {features}")

        # Scale numerical features only (indices 8-16 correspond to numerical_features)
        numerical_indices = list(range(8, 17))  # Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE, BMI
        numerical_array = np.array([features[i] for i in numerical_indices]).reshape(1, -1)
        numerical_scaled = scaler.transform(numerical_array)
        app.logger.debug(f"Scaled numerical features: {numerical_scaled}")

        # Reconstruct features with unscaled categorical and scaled numerical
        features_scaled = features[:8] + numerical_scaled[0].tolist()
        features_scaled = np.array([features_scaled])
        app.logger.debug(f"Final features: {features_scaled}")

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Use actual accuracy from Colab
        accuracy = 0.992  # Updated to 99.2%

        return jsonify({
            'prediction': str(prediction),
            'accuracy': accuracy
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
