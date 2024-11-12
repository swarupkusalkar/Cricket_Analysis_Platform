# predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
matches_df = pd.read_csv('data/matches.csv')

# Prepare data for training
X = matches_df[['team1', 'team2']]  # Simplified feature set
y = matches_df['winner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/match_prediction_model.pkl')
print("Model trained and saved!")

# Load the model for predictions
model = joblib.load('models/match_prediction_model.pkl')
predictions = model.predict(X_test)
print("Predictions made!")
