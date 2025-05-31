# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained models
cls_model = joblib.load('trained_data/student_pass_model.pkl')  # Load the updated student pass/fail model

# Helper function to predict pass/fail based on input
def predict_pass_fail(study_hours, internet_access, sleep_hours, total_score, stress_level):
    # Prepare the input data with only the required columns
    input_data = pd.DataFrame([[study_hours, internet_access, sleep_hours, total_score, stress_level]], 
                              columns=['Study_Hours_per_Week', 'Internet_Access_at_Home', 'Sleep_Hours_per_Night', 'Total_Score', 'Stress_Level (1-10)'])

    # Standardize the input (as the model was trained with standardized data)
    input_data_scaled = cls_model.named_steps['scaler'].transform(input_data)
    
    # Predict using the trained model
    prediction = cls_model.predict(input_data_scaled)
    
    # Return the prediction (0 for Fail, 1 for Pass)
    return "Pass" if prediction[0] == 1 else "Fail"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract the required input features
    study_hours = data['Study_Hours_per_Week']
    internet_access = data['Internet_Access_at_Home']
    sleep_hours = data['Sleep_Hours_per_Night']
    total_score = data['Total_Score']  # Now we also include Total_Score in the input
    stress_level = data['Stress_Level (1-10)']  # Now we also include Stress_Level in the input

    # Use the helper function to predict pass/fail
    prediction = predict_pass_fail(study_hours, internet_access, sleep_hours, total_score, stress_level)

    # Respond with the prediction
    response = {
        "Prediction": prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
