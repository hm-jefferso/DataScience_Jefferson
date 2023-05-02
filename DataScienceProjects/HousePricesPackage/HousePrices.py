
#importing the libraries
import pandas as pd
import seaborn as sns


#for demonstrational purpose to print n lines where n<=3000
#Code found on Stackoverflow
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


#importing the Dataset
train_data = pd.read_csv(


    "/DataScienceProjects/HousePricesPackage/house-prices-advanced-regression-techniques/"
    "train.csv")
print(train_data.head())

test_data = pd.read_csv(
    "/DataScienceProjects/HousePricesPackage/house-prices-advanced-regression-techniques/"
    "test.csv")
print(test_data.head())

print(train_data.isnull().sum())
print(sns.heatmap(train_data.isnull(), yticklabels=False,cbar=False))

train_data = train_data.drop(columns=['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
print(train_data.isnull().sum())