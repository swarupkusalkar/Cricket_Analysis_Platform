# data_loader.py
import pandas as pd
from sqlalchemy import create_engine

# Database connection setup
engine = create_engine('postgresql+psycopg2://postgres:postgres@localhost:5432/cricket_analytics')

# Load CSV files into DataFrames
matches_df = pd.read_csv('data/matches.csv')
deliveries_df = pd.read_csv('data/deliveries.csv')
players_df = pd.read_csv('data/players.csv')
teams_df = pd.read_csv('data/teams.csv')
most_runs_avg_df = pd.read_csv('data/most_runs_average_strikerate.csv')
teamwise_home_away_df = pd.read_csv('data/teamwise_home_and_away.csv')

# Data cleaning and transformations
matches_df['date'] = pd.to_datetime(matches_df['date'], dayfirst=True)
deliveries_df.fillna({'extras': 0}, inplace=True)
matches_df.drop_duplicates(inplace=True)

# Load DataFrames into PostgreSQL
matches_df.to_sql('matches', engine, if_exists='replace', index=False)
deliveries_df.to_sql('deliveries', engine, if_exists='replace', index=False)
players_df.to_sql('players', engine, if_exists='replace', index=False)
teams_df.to_sql('teams', engine, if_exists='replace', index=False)
most_runs_avg_df.to_sql('most_runs_averages', engine, if_exists='replace', index=False)
teamwise_home_away_df.to_sql('teamwise_home_away', engine, if_exists='replace', index=False)

print("Data loaded successfully!")
