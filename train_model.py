# train_model.py

# 📦 Import libraries needed for data handling, machine learning, and saving models
import pandas as pd  # Used to load and manipulate tabular data
from sklearn.model_selection import train_test_split  # Used to divide data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Used to prepare data for machine learning
from sklearn.ensemble import RandomForestClassifier  # Random Forest model for classification
from sklearn.pipeline import Pipeline  # Allows chaining preprocessing and model steps
import joblib  # Used for saving/loading trained models and encoders
import os  # Used for file and directory operations

# 📁 Make sure the folder for saving trained models exists (create if not)
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the dataset (replace this with your dataset path)
df = pd.read_csv('csv/Students_Grading_Dataset.csv')  # Replace with actual dataset path

# Print columns to check their names
print("Columns in the dataset:", df.columns)

# Convert the categorical variable (Internet_Access_at_Home) from 'yes'/'no' to 1/0
df['Internet_Access_at_Home'] = df['Internet_Access_at_Home'].apply(lambda x: 1 if x == 'yes' else 0)

# 🎯 Check the distribution of Pass/Fail values before creating the 'Pass_Fail' column
print("Pass/Fail distribution before creating the Pass_Fail column:")
print(df['Pass_Fail'].value_counts() if 'Pass_Fail' in df.columns else "Pass_Fail column does not exist")

# 🎯 Create the 'Pass_Fail' column based on a custom threshold for 'Total_Score'
# We'll adjust the threshold to 50 to better balance the pass/fail cases
df['Pass_Fail'] = df['Total_Score'].apply(lambda x: 1 if x >= 50 else 0)  # Adjusted to 50 for better balance

# Print the distribution of Pass/Fail to ensure it's balanced
print("Pass/Fail distribution after creating the Pass_Fail column:")
print(df['Pass_Fail'].value_counts())  # This will give you an idea of the balance between Pass/Fail

# 🎯 Select only the relevant columns for training (Study_Hours_per_Week, Internet_Access_at_Home, Sleep_Hours_per_Night, Total_Score, Stress_Level)
X = df[['Study_Hours_per_Week', 'Internet_Access_at_Home', 'Sleep_Hours_per_Night', 'Total_Score', 'Stress_Level (1-10)']]  # Include Stress_Level

# Dependent variable: 'Pass_Fail'
y = df['Pass_Fail']

# 🧠 Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Define a pipeline for the classification model using RandomForest and class weights
cls_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalize features for better performance
    ('classifier', RandomForestClassifier(
        n_estimators=100,           # Number of trees
        random_state=42,            # Fix seed for reproducibility
        class_weight='balanced'     # Handle class imbalance by giving more weight to the minority class
    ))
])

# 🧠 Train the classification model
cls_pipeline.fit(X_train, y_train)

# 🧠 Save the trained model with the new name
joblib.dump(cls_pipeline, 'trained_data/student_pass_model.pkl')  # Save the trained classification model as 'student_pass_model.pkl'

# ✅ Print final message to indicate the model is ready
print("✅ Training complete. Model saved as 'student_pass_model.pkl' in the 'trained_data/' folder.")
