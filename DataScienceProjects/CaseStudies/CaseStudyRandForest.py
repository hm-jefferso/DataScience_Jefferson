import pandas as pd
import itertools
import csv

# reading csv candy-data
dataset = pd.read_csv('/Users/anthonyjefferson/Desktop/''Bewerbung/Case Study/candy-data.csv')

# Features X include all Rows
X = dataset.iloc[:, 1:-3].values
y = dataset.iloc[:, -1].values

# importing RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Generate all possible scenarios
scenarios = list(itertools.product([0, 1],repeat=len(X[0])))

#Initialize variables for highest and lowest predictions
highest_prediction = float('-inf')
lowest_prediction = float('inf')

# Iterate over each scenario and make predictions
for scenario in scenarios:
    single_scenario = [list(scenario)]
    prediction = regressor.predict(single_scenario)
    print(f"Scenario: {single_scenario}, Prediction: {prediction}")

    # Update highest and lowest predictions
    if prediction > highest_prediction:
        highest_prediction = prediction

    if prediction < lowest_prediction:
        lowest_prediction = prediction

print(f'Highest: {highest_prediction}')
print(f'Lowest: {lowest_prediction}')

