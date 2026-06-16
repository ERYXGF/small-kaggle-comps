"""
This file contains all functions related to reading files, cleaning columns,
creating new features, and preparing the raw data.

It does not contain any machine learning models.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


#Defines the path to the training data
TRAINING_DATA_PATH = r"C:\Users\eryxg\Documents\CODING\small-kaggle-comps\02-SPACESHIP-TITANIC\data\raw\train.csv"
TESTING_DATA_PATH = r"C:\Users\eryxg\Documents\CODING\small-kaggle-comps\02-SPACESHIP-TITANIC\data\raw\test.csv"

def load_data(path: str) -> pd.DataFrame:
    """
    Loads the data
    """
    data = pd.read_csv(path)
    return data


def feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes columns that need feature engineering
    """
    #Handles missing values in the Cabin column
    data["Cabin"] = data["Cabin"].fillna("Unknown/Unknown/Unknown")
    #Splits the Cabin column and expands it in a temporary dataframe
    cabin_split = data["Cabin"].str.split("/", expand=True)
    #Adds the columns to the dataframe
    data["CabinDeck"] = cabin_split[0]
    data["CabinNum"] = pd.to_numeric(cabin_split[1], errors='coerce')  # Convert to numbers (float/int)
    data["CabinSide"] = cabin_split[2]
    #Drops the original Cabin column
    data = data.drop(columns=["Cabin"])
    #Returns the full data
    return data


def split_data(data: pd.DataFrame, target_column, validation_size, wanted_random_state=1):
    """
    Separates the target column from features and splits the data into training
    and validation sets
    """
    #Separates features (x) and target column (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    #Splits the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=validation_size,
        random_state=wanted_random_state
    )

    #Returns the values
    return X_train, X_val, y_train, y_val


def create_submission(predictions: pd.array, passenger_ids, output_file_path):
    """
    Creates a submission CSV file based on predictions and passenger id
    """
    #Creates new DataFrame with results
    submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Transported": predictions
    })
    #Saves the dataframe to a csv file
    submission.to_csv(output_file_path, index=False)
    #Prints confirmation
    print("Submission file successfully created!")
