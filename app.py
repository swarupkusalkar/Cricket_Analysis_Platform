import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS to handle cross-origin requests

# Load the trained model and encoder
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the teamwise performance data for calculating win rates
teamwise_performance_df = pd.read_csv("data/teamwise_home_and_away.csv")
teamwise_performance_df.set_index('team', inplace=True)

# Calculate win and match counts for each team
team_win_counts = teamwise_performance_df['home_wins'] + teamwise_performance_df['away_wins']
team_match_counts = teamwise_performance_df['home_matches'] + teamwise_performance_df['away_matches']

# Load the matches data for displaying upcoming matches
matches_df = pd.read_csv("data/matches.csv")

@app.route("/api/matches", methods=["GET"])
def get_matches():
    # Filter the upcoming matches (e.g., based on a date column if available)
    upcoming_matches = matches_df[['date', 'team1', 'team2', 'venue']]
    # Optionally, you can format the date or filter matches based on the current date
    
    response = upcoming_matches.to_dict(orient='records')
    return jsonify(response)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    team1 = data['team1']
    team2 = data['team2']
    toss_winner = data['toss_winner']
    toss_decision = data['toss_decision']

    try:
        # Encode the team and toss winner names using the label encoder
        team1_encoded = encoder.transform([team1])[0]
        team2_encoded = encoder.transform([team2])[0]
        toss_winner_encoded = encoder.transform([toss_winner])[0]
        toss_decision_encoded = 0 if toss_decision == 'Bat' else 1

        # Calculate win rates for the input teams using .loc to access by label
        team1_win_rate = team_win_counts.loc[team1_encoded] / team_match_counts.loc[team1_encoded] if team1_encoded in team_win_counts.index and team_match_counts.loc[team1_encoded] > 0 else 0
        team2_win_rate = team_win_counts.loc[team2_encoded] / team_match_counts.loc[team2_encoded] if team2_encoded in team_win_counts.index and team_match_counts.loc[team2_encoded] > 0 else 0

        # Prepare the input array with all 6 features
        input_features = pd.DataFrame([[team1_encoded, team2_encoded, toss_winner_encoded, toss_decision_encoded, team1_win_rate, team2_win_rate]],
                                      columns=['team1', 'team2', 'toss_winner', 'toss_decision', 'team1_win_rate', 'team2_win_rate'])

        # Get probability predictions
        probabilities = model.predict_proba(input_features)[0]

        # Ensure the probabilities sum to 100% and adjust if necessary
        team1_prob = probabilities[0] * 100
        team2_prob = probabilities[1] * 100

        if team1_prob + team2_prob == 0:
            # If both probabilities are zero, assign random probabilities that sum to 100%
            team1_prob = np.random.uniform(1, 99)
            team2_prob = 100 - team1_prob
        else:
            # Normalize the probabilities to ensure they sum to 100%
            team1_prob = (team1_prob / (team1_prob + team2_prob)) * 100
            team2_prob = 100 - team1_prob

        response = {
            "team1_win_prob": round(team1_prob, 2),
            "team2_win_prob": round(team2_prob, 2)
        }
    except Exception as e:
        response = {
            "error": str(e)
        }

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5000)
