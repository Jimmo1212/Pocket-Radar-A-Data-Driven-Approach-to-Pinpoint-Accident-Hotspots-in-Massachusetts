from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Initialize Flask App
app = Flask(__name__)

# Load trained PCA + Logistic Regression model
model = joblib.load("pca_logreg_accident_model.pkl")

# Load historical accident dataset
df = pd.read_csv("US_Accidents_MA.csv")

# Extract required location and infrastructure features
df_infra = df[["Start_Lat", "Start_Lng", "Traffic_Signal", "Junction", "Crossing", "Amenity",
               "Bump", "Give_Way", "No_Exit", "Railway", "Station", "Stop", "Traffic_Calming"]
             ].drop_duplicates().fillna(0)

# Get current month & week for prediction
current_month = datetime.now().month
current_week = datetime.now().isocalendar()[1]

# Default duration value (in seconds)
DEFAULT_DURATION = 300  # 5 minutes

# One-Hot Encoding mappings
weather_map = ["Weather_Fair", "Weather_Cloudy", "Weather_Clear", "Weather_Overcast",
               "Weather_Snow", "Weather_Haze", "Weather_Rain", "Weather_Thunderstorm",
               "Weather_Windy", "Weather_Hail", "Weather_Thunder"]
wind_map = ["Wind_C", "Wind_E", "Wind_N", "Wind_S", "Wind_V", "Wind_W"]

@app.route('/')
def index():
    """ Render the homepage. """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Predict accident severity based on user inputs. """
    try:
        # Extract user inputs from the request form
        user_input = {
            "Hour": int(request.form["Hour"]),
            "Month": int(request.form.get("Month", current_month)),
            "Week": int(request.form.get("Week", current_week)),
            "Weather_Condition": request.form["Weather_Condition"],
            "Temperature(F)": float(request.form["Temperature"]),
            "Humidity(%)": float(request.form["Humidity"]),
            "Pressure(in)": float(request.form["Pressure"]),
            "Visibility(mi)": float(request.form["Visibility"]),
            "Wind_Condition": request.form["Wind_Condition"],
            "Wind_Speed(mph)": float(request.form["Wind_Speed"])
        }

        # Prepare prediction datasets for each severity level
        severity_data = {1: [], 2: [], 3: [], 4: []}

        for _, row in df_infra.iterrows():
            input_features = np.array([row[f] for f in df_infra.columns])

            # Append user inputs
            input_features = np.append(input_features, [
                user_input["Temperature(F)"], user_input["Humidity(%)"],
                user_input["Pressure(in)"], user_input["Visibility(mi)"],
                user_input["Wind_Speed(mph)"], DEFAULT_DURATION,
                user_input["Month"], user_input["Week"], user_input["Hour"]
            ])

            # One-Hot Encoding for Weather Condition
            weather_one_hot = np.zeros(len(weather_map))
            if user_input["Weather_Condition"] in weather_map:
                weather_idx = weather_map.index(user_input["Weather_Condition"])
                weather_one_hot[weather_idx] = 1
            input_features = np.append(input_features, weather_one_hot)

            # One-Hot Encoding for Wind Condition
            wind_one_hot = np.zeros(len(wind_map))
            if user_input["Wind_Condition"] in wind_map:
                wind_idx = wind_map.index(user_input["Wind_Condition"])
                wind_one_hot[wind_idx] = 1
            input_features = np.append(input_features, wind_one_hot)

            # Standardize input features before passing them into PCA
            input_features = model.named_steps['scaler'].transform([input_features])

            # Transform input features using PCA
            input_features = model.named_steps['pca'].transform(input_features)

            # Predict severity using Logistic Regression
            severity_prediction = int(model.named_steps['logreg'].predict(input_features)[0])

            # Store predictions separately by severity
            if severity_prediction in severity_data:
                severity_data[severity_prediction].append({
                    "latitude": row["Start_Lat"],
                    "longitude": row["Start_Lng"]
                })

        return render_template('predicted_map.html', accident_data=json.dumps(severity_data), user_inputs=user_input)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/map')
def show_map():
    """ Render the map visualization page. """
    return render_template('predicted_map.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
