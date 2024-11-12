import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load data from CSV files
matches_df = pd.read_csv("data/matches.csv")
teamwise_performance_df = pd.read_csv("data/teamwise_home_and_away.csv")

# Handle missing values in matches_df
matches_df['winner'] = matches_df['winner'].fillna("No Result")

# Encode categorical variables using LabelEncoder
encoder = LabelEncoder()
encoder.fit(pd.concat([matches_df['team1'], matches_df['team2'], matches_df['toss_winner'], matches_df['winner']]))

# Apply encoding to columns
matches_df['team1'] = encoder.transform(matches_df['team1'].astype(str))
matches_df['team2'] = encoder.transform(matches_df['team2'].astype(str))
matches_df['toss_winner'] = encoder.transform(matches_df['toss_winner'].astype(str))
matches_df['winner'] = encoder.transform(matches_df['winner'].astype(str))

# Handle NaN values in toss_decision by filling with a default value (0 for 'Bat')
matches_df['toss_decision'] = matches_df['toss_decision'].map({'Bat': 0, 'Field': 1}).fillna(0)

# Set the team column as the index for easier referencing
teamwise_performance_df.set_index('team', inplace=True)

# Calculate win counts and match counts for each team from teamwise data
team_win_counts = teamwise_performance_df['home_wins'] + teamwise_performance_df['away_wins']
team_match_counts = teamwise_performance_df['home_matches'] + teamwise_performance_df['away_matches']

# Initialize columns for historical win rates in matches_df
matches_df['team1_win_rate'] = np.nan
matches_df['team2_win_rate'] = np.nan

# Calculate win rates for each match
for i, row in matches_df.iterrows():
    team1 = row['team1']
    team2 = row['team2']

    # Calculate team1 win rate
    if team1 in team_win_counts.index and team_match_counts.loc[team1] > 0:
        matches_df.at[i, 'team1_win_rate'] = team_win_counts.loc[team1] / team_match_counts.loc[team1]
    else:
        matches_df.at[i, 'team1_win_rate'] = 0  # Set a default if no match data is available

    # Calculate team2 win rate
    if team2 in team_win_counts.index and team_match_counts.loc[team2] > 0:
        matches_df.at[i, 'team2_win_rate'] = team_win_counts.loc[team2] / team_match_counts.loc[team2]
    else:
        matches_df.at[i, 'team2_win_rate'] = 0  # Set a default if no match data is available

# Ensure no NaN values in team1_win_rate and team2_win_rate
matches_df[['team1_win_rate', 'team2_win_rate']] = matches_df[['team1_win_rate', 'team2_win_rate']].fillna(0)

# Prepare features (X) and target (y)
X = matches_df[['team1', 'team2', 'toss_winner', 'toss_decision', 'team1_win_rate', 'team2_win_rate']]
y = matches_df['winner']

# Fill any remaining NaN values in X with column means (as a final check)
X = X.fillna(X.mean())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler to scale features for models that benefit from scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate the minimum samples in the smallest class for dynamic n_neighbors
min_samples_per_class = y_train.value_counts().min()
n_neighbors = min(5, min_samples_per_class - 1) if min_samples_per_class > 1 else 1  # Ensure n_neighbors <= samples in the smallest class

# Apply SMOTE with the dynamically calculated n_neighbors
oversample = SMOTE(sampling_strategy='auto', k_neighbors=n_neighbors, random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)
print(f"Applied SMOTE with n_neighbors={n_neighbors} based on smallest class sample size.")

# Dictionary to store the models and their filenames
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Dictionary to store model accuracies for visualization
model_accuracies = {}

# Train each model, evaluate accuracy, and save it
for model_name, model in models.items():
    # Optional: Hyperparameter tuning for RandomForest
    if model_name == "RandomForest":
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best params for {model_name}: {grid_search.best_params_}")

    model.fit(X_train, y_train)  # Train the model
    accuracy = model.score(X_test, y_test)  # Evaluate the model on the test set
    model_accuracies[model_name] = accuracy * 100
    print(f"{model_name} Accuracy: {model_accuracies[model_name]:.2f}%")

    # Save the model
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save the label encoder and scaler
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model training complete and all models saved.")

# Plot the accuracies of each model
plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.show()

# Cross-validation for each model
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{model_name} Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
