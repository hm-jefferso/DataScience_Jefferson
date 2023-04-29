import pandas as pd

train_data = pd.read_csv("/Users/anthonyjefferson/PycharmProjects/"
                         "DataScience_Jefferson/house-prices-advanced-regression-techniques/"
                         "train.csv")

test_data = pd.read_csv("/Users/anthonyjefferson/PycharmProjects/"
                        "DataScience_Jefferson/house-prices-advanced-regression-techniques/"
                        "test.csv")


price = train_data.loc [train_data.Street == "Pave"]["SalePrice"]

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

y = train_data["SalePrice"]

