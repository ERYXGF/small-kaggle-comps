"""
This file import my helpers from utils.py and focuses entirely on creating
preprocessors, constructing pipelines, training, and making predictions.

It then ties and executes everything together through main.py
"""
#Code imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

#Utils imports
from .utils import load_data
from .utils import feature_engineer
from .utils import split_data
from .utils import create_submission
from .utils import TRAINING_DATA_PATH
from .utils import TESTING_DATA_PATH

def preprocess_data(numerical_columns: list, non_numerical_columns: list) -> ColumnTransformer:
    """
    Preprocesses missing values and encodes non-numerical columns
    """
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing for categorical (non-numerical) data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, non_numerical_columns)
        ]
    )

    return preprocessor


def build_pipeline(
        preprocessor:ColumnTransformer,
        classifier: XGBClassifier
):
    """
    Builds the models pipeline
    """

    #Builds the pipeline
    my_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    #Returns the pipeline
    return my_pipeline


def train_and_evaluate_model(pipeline: Pipeline, train_X, train_y, val_X, val_y):
    """
    Trains and evaluates the model
    """

    #Fits the pipline
    updated_pipeline = pipeline.fit(train_X, train_y)

    #Predicts validation features
    predictions = pipeline.predict(val_X)

    #Calculates the accuracy score
    accuracy = accuracy_score(predictions, val_y)

    #Prints the score as a percentage
    print(f"Accuracy Score: {accuracy:.4f}%")

    #Returns the trained pipeline
    return updated_pipeline

def main():
    """
    Main execution block that executes all other functions
    """
    #Loads the training and validation data
    training_data = load_data(TRAINING_DATA_PATH)
    validation_data = load_data(TESTING_DATA_PATH)

    #Performs feature engineering
    training_data = feature_engineer(training_data)
    validation_data = feature_engineer(validation_data)

    #Column definition
    numerical_column_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    categorical_column_features = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "CabinNum"]
    target_column = "Transported"

    #Splis the training data
    X_train, X_val, y_train, y_val = split_data(
        data=training_data,
        target_column=target_column,
        validation_size=0.3,
        wanted_random_state=1
    )

    #Creates the preprocessor
    established_preprocessor = preprocess_data(
        numerical_columns=numerical_column_features,
        non_numerical_columns=categorical_column_features
    )

    #Defines the model
    model = XGBClassifier(n_estimators=100,max_depth=5, random_state=1)
    #Builds the pipeline
    my_pipeline = build_pipeline(
        preprocessor=established_preprocessor,
        classifier=model
    )

    #Trains and evaluates the model and prints the accuracy
    trained_pipeline = train_and_evaluate_model(
        pipeline=my_pipeline,
        train_X=X_train,
        train_y=y_train,
        val_X=X_val,
        val_y=y_val
    )

    #Generate test Predictions
    passenger_id_col = validation_data["PassengerId"].copy()
    test_features = validation_data.loc[:, X_train.columns]
    test_predictions = trained_pipeline.predict(test_features).astype(bool)

    #Save the results
    create_submission(
        predictions=test_predictions,
        passenger_ids=passenger_id_col,
        output_file_path=r"C:\Users\eryxg\Documents\CODING\small-kaggle-comps\02-SPACESHIP-TITANIC\submission.csv"
    )

if __name__ == "__main__":
    main()
