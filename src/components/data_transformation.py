import sys  # Import the sys module for accessing system-specific parameters and functions
from dataclasses import dataclass  # Import dataclass to create simple data classes

import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for applying different preprocessing steps to columns
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.pipeline import Pipeline  # Import Pipeline for chaining preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Import encoders and scalers for data preprocessing

from src.exception import CustomException  # Import custom exception class for error handling
from src.logger import logging  # Import logging functionality to log information and errors
import os  # Import os module to interact with the operating system

from src.utils import save_object  # Import utility function for saving objects

# Define a configuration class for data transformation using the @dataclass decorator
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save the preprocessing object

# Define the DataTransformation class responsible for transforming data
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Initialize the configuration for data transformation

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline.

        Returns:
            preprocessor (ColumnTransformer): The preprocessing object with pipelines for numerical and categorical features.
        '''
        try:
            # List of numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Define the pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with the median
                    ("scaler", StandardScaler())  # Scale numerical features
                ]
            )

            # Define the pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with the most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Apply one-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical features (no mean scaling for one-hot encoded features)
                ]
            )

            # Log the column names for numerical and categorical features
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create a ColumnTransformer to apply the pipelines to the respective columns
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline to numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply categorical pipeline to categorical columns
                ]
            )

            return preprocessor  # Return the preprocessing object

        except Exception as e:
            # If an error occurs, raise a custom exception with the error details
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for initiating data transformation on the provided training and testing data.

        Args:
            train_path (str): The file path to the training data.
            test_path (str): The file path to the testing data.

        Returns:
            tuple: A tuple containing:
                - train_arr (numpy.ndarray): Transformed training data.
                - test_arr (numpy.ndarray): Transformed testing data.
                - preprocessor_obj_file_path (str): Path to the saved preprocessing object.
        '''
        try:
            # Read the training and testing data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")  # Log successful data reading

            logging.info("Obtaining preprocessing object")  # Log that preprocessing object is being created

            preprocessing_obj = self.get_data_transformer_object()  # Get the preprocessing object

            target_column_name = "math_score"  # Name of the target column
            numerical_columns = ["writing_score", "reading_score"]  # List of numerical columns

            # Separate features and target variable for training and testing datasets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )  # Log that preprocessing is being applied

            # Apply the preprocessing object to transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")  # Log that the preprocessing object has been saved

            # Save the preprocessing object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )  # Return the transformed training and testing data, and the path to the saved preprocessing object

        except Exception as e:
            # If an error occurs, raise a custom exception with the error details
            raise CustomException(e, sys)

