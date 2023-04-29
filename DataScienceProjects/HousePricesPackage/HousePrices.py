
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the Dataset
train_data = pd.read_csv("/Users/anthonyjefferson/PycharmProjects/"
                         "DataScience_Jefferson/house-prices-advanced-regression-techniques/"
                         "train.csv")

test_data = pd.read_csv("/Users/anthonyjefferson/PycharmProjects/"
                        "DataScience_Jefferson/house-prices-advanced-regression-techniques/"
                        "test.csv")

X_train = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

#importing the xgbClassifier to fit the data
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y)




