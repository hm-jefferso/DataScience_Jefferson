import pandas as pd

# First 5 rows of Train Data
train_data = pd.read_csv("/Users/anthonyjefferson/Downloads/titanic/train.csv")
#print (train_data.head())

#First 5 rows of Test data
test_data = pd.read_csv("/Users/anthonyjefferson/Downloads/titanic/test.csv")
print (test_data.head())

#Survival Rate Women
women = train_data.loc [train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print(f"Percentage of Women who survived Titanic: {rate_women}")

#Survival Rate Men
men = train_data.loc[train_data.Sex == "male"]["Survived"]
rate_men = sum(men) / len(men)
print(f"Percentage of Men who survived Titanic: {rate_men}")

#looking at all people that survived
y = train_data["Survived"]

#defining features X by selecting the important columns
features = ["Pclass", "SibSp", "Parch", "Sex"]

#get_dummies
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#RF Algorithm
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,max_depth=5, random_state=1)
rfc.fit(X, y)

predictions = rfc.predict(X_test)
print(predictions)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index = True)
print("Your Submission was succesfully saved")
print(output)