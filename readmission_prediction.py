# ===============================
# 1. Data Preprocessing
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Load dataset (placeholder path)
df = pd.read_csv("#")

# Handle missing values
imputer = SimpleImputer(strategy="median")
df[['age', 'blood_pressure', 'heart_rate']] = imputer.fit_transform(df[['age', 'blood_pressure', 'heart_rate']])

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'insurance_type', 'discharge_department'], drop_first=True)

# Normalize numerical features
scaler = MinMaxScaler()
df[['age', 'blood_pressure', 'heart_rate']] = scaler.fit_transform(df[['age', 'blood_pressure', 'heart_rate']])

# Define features and target
X = df.drop(columns=['readmitted_30_days'])
y = df['readmitted_30_days']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# ===============================
# 2. Model Training (XGBoost)
# ===============================
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 3. Evaluation (Confusion Matrix, Precision, Recall)
# ===============================
from sklearn.metrics import confusion_matrix, precision_score, recall_score

y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision and Recall
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# ===============================
# 4. Simple API for Deployment (Flask)
# ===============================
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    input_df = pd.DataFrame([input_data])
    
    # Preprocess incoming data like training data
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    
    prediction = model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])})

# Uncomment below to run Flask app (for testing locally)
# if __name__ == '__main__':
#     app.run(debug=True)

# ===============================
# 5. Optional: Concept Drift Monitoring Function
# ===============================
def detect_concept_drift(reference, new_data):
    """
    Example placeholder for concept drift detection logic.
    In practice, you'd compare feature distributions or model outputs over time.
    """
    drift_detected = False
    # Compare distributions or performance metrics here
    # Set drift_detected = True if drift is significant
    return drift_detected

# ===============================
# End of Workflow Script
# ===============================
