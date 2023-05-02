#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##importing dataset
dataset = pd.read_csv('/Users/anthonyjefferson/PycharmProjects/DataScience_Jefferson/DataScienceProjects/PositionSalaries/Position_Salaries-2.csv')
print(dataset.head())

#definining Features X and dependent variable y
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.scatter(X,y)
plt.show()

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators= 20, random_state=42)
rfr.fit(X, y)

#predicting singular result
example = rfr.predict([[5.5]])
print(f'at 5.5 the estimated salary is: {example}')

#predicting test set results
y_pred = rfr.predict(X_test)
print(f'predicted values: {y_pred}')



