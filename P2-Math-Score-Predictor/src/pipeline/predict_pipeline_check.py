import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Debugging: Check if the model and preprocessor are loaded
            print(f"Model: {model}")
            print(f"Preprocessor: {preprocessor}")

            # Debugging: Hardcoding some feature values for testing
            test_features = pd.DataFrame({
                "gender": ["male"],
                "race_ethnicity": ["group B"],
                "parental_level_of_education": ["bachelor's degree"],
                "lunch": ["standard"],
                "test_preparation_course": ["completed"],
                "reading_score": [85],
                "writing_score": [98],
            })

            print("Test Features:")
            print(test_features)

            data_scaled = preprocessor.transform(test_features)
            print("Scaled Data:")
            print(data_scaled)

            preds = model.predict(data_scaled)
            print("Predictions:")
            print(preds)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

# Testing the PredictPipeline class directly
if __name__ == "__main__":
    pipeline = PredictPipeline()
    test_features = pd.DataFrame({
        "gender": ["male"],
        "race_ethnicity": ["group B"],
        "parental_level_of_education": ["bachelor's degree"],
        "lunch": ["standard"],
        "test_preparation_course": ["completed"],
        "reading_score": [85],
        "writing_score": [98],
    })
    print("Running Prediction Pipeline Test...")
    predictions = pipeline.predict(test_features)
    print(f"Test Predictions: {predictions}")
