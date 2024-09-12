import os  # Import the os module for interacting with the operating system
import sys  # Import the sys module for system-specific parameters and functions
from dataclasses import dataclass  # Import dataclass to create simple data classes

from catboost import CatBoostRegressor  # Import CatBoostRegressor for CatBoost models
from sklearn.ensemble import (
    AdaBoostRegressor,  # Import AdaBoostRegressor for boosting
    GradientBoostingRegressor,  # Import GradientBoostingRegressor for gradient boosting
    RandomForestRegressor,  # Import RandomForestRegressor for random forest models
)
from sklearn.linear_model import LinearRegression  # Import LinearRegression for linear models
from sklearn.metrics import r2_score  # Import r2_score to evaluate the performance of regression models
from sklearn.neighbors import KNeighborsRegressor  # Import KNeighborsRegressor for k-nearest neighbors regression
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor for decision tree models
from xgboost import XGBRegressor  # Import XGBRegressor for XGBoost models

from src.exception import CustomException  # Import custom exception class for error handling
from src.logger import logging  # Import logging functionality to log information and errors

from src.utils import save_object, evaluate_models  # Import utility functions for saving objects and evaluating models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  # Initialize the configuration for model training

    def initiate_model_trainer(self, train_array, test_array):
        '''
        This function is responsible for training various regression models, selecting the best model, 
        and saving the trained model.

        Args:
            train_array (numpy.ndarray): The array containing training data (features and target).
            test_array (numpy.ndarray): The array containing testing data (features and target).

        Returns:
            float: R2 score of the best model on the test data.
        '''
        try:
            logging.info("Split training and test input data")  # Log the start of data splitting
            # Split the data into features and target variables for training and testing
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features for training
                train_array[:, -1],   # Target for training
                test_array[:, :-1],   # Features for testing
                test_array[:, -1]     # Target for testing
            )

            # Define a dictionary of models to be trained and their hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameter grids for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate the models and get the performance report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            # Find the best model based on the highest R2 score
            best_model_score = max(sorted(model_report.values()))  # Get the best model score

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  # Get the name of the best model

            best_model = models[best_model_name]  # Get the best model instance

            # Raise an exception if the best model score is below a threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")  # Log the best model found

            # Save the best model to a file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict using the best model on the test data
            predicted = best_model.predict(X_test)

            # Calculate and return the R2 score of the best model
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # If an error occurs, raise a custom exception with the error details
            raise CustomException(e, sys)
