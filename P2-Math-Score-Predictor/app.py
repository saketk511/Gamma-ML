from flask import Flask, request, render_template
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        gender = request.form['gender']
        ethnicity = request.form['ethnicity']
        parental_level_of_education = request.form['parental_level_of_education']
        lunch = request.form['lunch']
        test_preparation_course = request.form['test_preparation_course']
        reading_score = int(request.form['reading_score'])
        writing_score = int(request.form['writing_score'])

        # Create CustomData instance
        custom_data = CustomData(
            gender=gender,
            ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Convert to DataFrame
        data_frame = custom_data.get_data_as_data_frame()

        # Make prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(data_frame)

        return f"Prediction: {prediction[0]}"
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
