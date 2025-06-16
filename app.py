from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Enables communication with frontend

# Load your trained pipeline (e.g., StandardScaler + Classifier)
cls_model = joblib.load('trained_data/student_pass_model.pkl')

def predict_pass_fail(study_hours, internet_access, sleep_hours, total_score, stress_level):
    input_data = pd.DataFrame([[study_hours, internet_access, sleep_hours, total_score, stress_level]],
                              columns=[
                                  'Study_Hours_per_Week',
                                  'Internet_Access_at_Home',
                                  'Sleep_Hours_per_Night',
                                  'Total_Score',
                                  'Stress_Level (1-10)'
                              ])
    input_data_scaled = cls_model.named_steps['scaler'].transform(input_data)
    prediction = cls_model.predict(input_data_scaled)
    return "Pass" if prediction[0] == 1 else "Fail"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        prediction = predict_pass_fail(
            data['Study_Hours_per_Week'],
            data['Internet_Access_at_Home'],
            data['Sleep_Hours_per_Night'],
            data['Total_Score'],
            data['Stress_Level (1-10)']
        )
        return jsonify({"Prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
