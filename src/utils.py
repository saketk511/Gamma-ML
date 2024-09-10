import os  # Import the os module for interacting with the operating system, such as file and directory handling
import sys  # Import the sys module to access system-specific parameters and functions

import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
import dill  # Import dill for advanced object serialization (not used in the current code)
import pickle  # Import pickle for serializing and deserializing Python objects
from sklearn.metrics import r2_score  # Import r2_score to evaluate the performance of regression models
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning of models

from src.exception import CustomException  # Import a custom exception class for handling and raising errors

# Function to save an object to a file using pickle
def save_object(file_path, obj):
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # If an error occurs, raise a custom exception with the error details
        raise CustomException(e, sys)

# Function to evaluate multiple models with hyperparameter tuning
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # Initialize an empty dictionary to store evaluation results

        # Iterate over each model in the provided models dictionary
        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model instance
            para = param[list(models.keys())[i]]  # Get the corresponding hyperparameters

            # Perform grid search with cross-validation to find the best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)  # Fit the grid search model

            # Update the model with the best hyperparameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Train the model on the training data

            # Predict on the training data
            y_train_pred = model.predict(X_train)

            # Predict on the test data
            y_test_pred = model.predict(X_test)

            # Calculate the R-squared score for the training data
            train_model_score = r2_score(y_train, y_train_pred)

            # Calculate the R-squared score for the test data
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test model score in the report dictionary with the model name as the key
            report[list(models.keys())[i]] = test_model_score

        return report  # Return the report with evaluation results

    except Exception as e:
        # If an error occurs, raise a custom exception with the error details
        raise CustomException(e, sys)

# Function to load a saved object from a file using pickle
def load_object(file_path):
    try:
        # Open the file in read-binary mode and load the object using pickle
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        # If an error occurs, raise a custom exception with the error details
        raise CustomException(e, sys)
