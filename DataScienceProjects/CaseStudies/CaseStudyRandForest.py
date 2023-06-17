#TODO: The goal of this code is to use Machine
# Learning in order to predict, what Feature X a
# Sweet needs in order to be successfull (see dataset)

#importing neccessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading csv candy-data
dataset = pd.read_csv('/Users/anthonyjefferson/Desktop/'
                      'Bewerbung/Case Study/candy-data.csv')

#Features X include all Rows,
X = dataset.iloc[:, 1:-3].values
y = dataset.iloc[:, -1].values

#importing RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# Predict for a single scenario
single_scenario = [[1, 0, 0, 0, 0, 1, 0, 1, 1]]
prediction = regressor.predict(single_scenario)
print(f"Prediction for the single scenario: {prediction}")

#TODO: Currently Predicting winpercent for single scenario
#TODO: Task is to create Scenarios for all 9 Features X,
#TODO: preferrably by using for loop

# Predict for multiple scenarios
scenarios = [[scenario_1_features], [scenario_2_features], ...]

#loop trough all 9 features to get min and max for new sweet
for i, scenario in enumerate(scenarios):
    prediction = regressor.predict(scenario)
    print(f"Prediction for scenario {i+1}: {prediction}")