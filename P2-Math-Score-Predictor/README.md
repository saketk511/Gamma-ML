## End to End Machine Learning Project

Student Performance Prediction
This project is aimed at predicting students' math scores based on various features such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course. The dataset includes scores in math, reading, and writing, providing insights into the factors that influence student performance.

Key Features:
Data Analysis & Visualization: Exploratory Data Analysis (EDA) is performed to uncover relationships between different features and student performance.
Model Development: A machine learning pipeline is used to train models for predicting math scores, leveraging various regression techniques.
Web Application: An interactive web app is built using Flask, where users can input student information to predict their math scores.

How to run this:

To create new environment : conda create -p venv python==3.8 -y
To activate the environment : conda activate venv/
To install requirements: pip install -r requirements.txt
To complete data ingestion-transformation-model building: python src/components/data_ingestion.py
To open flask app: python app.py
Open a web browser (e.g., Chrome, Firefox).
Go to http://127.0.0.1:5000/ or http://localhost:5000/.

See the UI in Input and Output image.
