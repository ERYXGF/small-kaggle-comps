"""Trains a model to create a csv prediction file on who survived the titanic."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#Establishes the path to the training csv and reads it
titanic_data_path = r"C:\Users\eryxg\Documents\CODING\small-kaggle-comps\01-TITANIC-ML-FROM-DISASTER\titanic_data\train.csv"
titanic_data = pd.read_csv(titanic_data_path)

#Drops rows where fare is missing and updates titanic_data (inplace=True)
titanic_data.dropna(subset=["Fare"], inplace=True)

#Defines y (the target)
y = titanic_data.Survived

#Defines the Sex by converting the strings into numbers
titanic_data["Sex"] = titanic_data["Sex"].map({"male": 0 , "female" : 1})

#Fills missing aga data with the age median:
titanic_data["Age"] = titanic_data["Age"].fillna(titanic_data["Age"].median())

#Creates a family size column that shows the amount of family of a single passenger
titanic_data["Family_Size"] = titanic_data["SibSp"] + titanic_data["Parch"] +1

#Creates X, the data used
passenger_info = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Family_Size"]
X = titanic_data[passenger_info]

#Splits the data into a training and validation set
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Calculates the optimal RandomForestClassifier settings based on the options given in dict
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5)
grid_search.fit(train_X, train_y)

print(grid_search.best_params_)

#Creates, tunes and fits the model
titanic_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
titanic_model.fit(train_X, train_y)

#Calculates the accuracy score to evaluate the model
predictions = titanic_model.predict(val_X)
accuracy = accuracy_score(val_y, predictions)

print(f"Validation accuracy: {accuracy*100:.2f}%")

#The following code will produce the actual code to submit to the comp using the testing data
#Loads the path to the data and reads the csv file
titanic_testing_data_path = r"C:\Users\eryxg\Documents\CODING\small-kaggle-comps\01-TITANIC-ML-FROM-DISASTER\titanic_data\test.csv"
titanic_test_data = pd.read_csv(titanic_testing_data_path)

#Rehandles the missing data and other columns I created before:
titanic_test_data["Age"] = titanic_test_data["Age"].fillna(titanic_test_data["Age"].median())
titanic_test_data["Sex"] = titanic_test_data["Sex"].map({"male" : 0 , "female" : 1})
titanic_test_data["Fare"] = titanic_test_data["Fare"].fillna(titanic_test_data["Fare"].median())
titanic_test_data["Family_Size"] = titanic_test_data["SibSp"] + titanic_test_data["Parch"] + 1

#Defines test_X
testing_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Family_Size"]
test_X = titanic_test_data[testing_columns]

#Uses the trained model to make the predictions
test_predictions = titanic_model.predict(test_X)

#Creates a new Dataframe with the predictions in Kaggle's wanted format
submission = pd.DataFrame({
    "PassengerId": titanic_test_data["PassengerId"],
    "Survived": test_predictions
})

#Saves the dataframe to a csv file
submission.to_csv("kaggle_submission.csv", index=False)
print("Submission file successfully created!")
