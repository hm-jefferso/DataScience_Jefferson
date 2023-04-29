import pandas as pd

# First 5 rows of Train Data
train_data = pd.read_csv("/Users/anthonyjefferson/Downloads/titanic/train.csv")
#print (train_data.head())

#First 5 rows of Test data
test_data = pd.read_csv("/Users/anthonyjefferson/Downloads/titanic/test.csv")
#print (test_data.head())

#Survival Rate Women
women = train_data.loc [train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print(f"Percentage of Women who survived Titanic: {rate_women}")

#Survival Rate Men
men = train_data.loc[train_data.Sex == "male"] ["Survived"]
rate_men = sum(men) / len(men)
print(f"Percentage of Men who survived Titanic: {rate_men}")


