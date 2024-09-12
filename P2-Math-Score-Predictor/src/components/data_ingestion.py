import os  # Import the os module to interact with the operating system (for file and directory handling)
import sys  # Import the sys module to access system-specific parameters and functions
from src.exception import CustomException  # Import the custom exception class for handling and raising exceptions
from src.logger import logging  # Import the logging functionality to log information and errors
import pandas as pd  # Import pandas for data manipulation and analysis

from sklearn.model_selection import train_test_split  # Import train_test_split to split the dataset into training and testing sets
from dataclasses import dataclass  # Import dataclass to create simple data classes

from src.components.data_transformation import DataTransformation  # Import the DataTransformation component
from src.components.data_transformation import DataTransformationConfig  # Import the DataTransformationConfig for configuration

from src.components.model_trainer import ModelTrainerConfig  # Import the ModelTrainerConfig for configuration
from src.components.model_trainer import ModelTrainer  # Import the ModelTrainer component for model training

# Define a configuration class for data ingestion using the @dataclass decorator
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save the training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save the test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to save the raw data

# Define the DataIngestion class responsible for ingesting the data
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize the configuration for data ingestion

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log that the data ingestion process has started
        try:
            # Read the data from a CSV file into a DataFrame
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')  # Log that the dataset was successfully read

            # Create the directory to store the artifacts (if it doesn't already exist)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log that the train-test split is starting
            # Split the dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training data to a CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the testing data to a CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log that the data ingestion process is complete

            # Return the paths to the training and testing data files
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If an error occurs, raise a custom exception with the error details
            raise CustomException(e, sys)

# Main block to execute the data ingestion, data transformation, and model training processes
if __name__ == "__main__":
    # Create an instance of DataIngestion and initiate the data ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an instance of DataTransformation and initiate the data transformation process
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Create an instance of ModelTrainer and initiate the model training process
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))  # Print the model training results




