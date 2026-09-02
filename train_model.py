import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset

data = pd.read_csv("road_data.csv")

# Input features

X = data[
[
"road_age",
"traffic_level",
"rainfall",
"previous_repairs",
"damage_severity"
]
]

# Target value

y = data["risk_score"]

# Create the Machine Learning model

model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

# Train the model

model.fit(X, y)

# Save the trained model

joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("Model saved as model.pkl")
