#importing libraries
import pandas as pd


##importing dataset
dataset = pd.read_csv('/Users/anthonyjefferson/PycharmProjects/DataScience_Jefferson/DataScienceProjects/PositionSalaries/Position_Salaries-2.csv')
print(dataset.head())

#definining Features X and dependent variable y
X = dataset.iloc[:, -1]
y = dataset.iloc[:, :-1]

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(test_size=0.3, train_size=0.7, random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X, y)

#predicting test set results
y_pred = rfr.predict(X_test)
print(y_pred)

#checking model for accuracy and
from sklearn.metrics import accuracy_score
acs = accuracy_score(y_pred)
print(acs)