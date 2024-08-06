# IPL Win Predictor

Welcome to the "IPL Win Predictor" project! This machine learning model, built using logistic regression, predicts the probability of a team winning an IPL match based on the current match situation. Get ready to make data-driven predictions!

## About This Project

The "IPL Win Predictor" leverages logistic regression to provide insights into the probability of a team winning an IPL match. This model analyzes various match features, team performance, and player statistics to offer real-time predictions.

### Features

- **Real-Time Predictions**: Get live predictions for IPL match outcomes based on the current match situation.

- **Interactive Interface**: The predictor is deployed on Streamlit, offering a user-friendly interface for exploring match scenarios.

- **Customizable Inputs**: Adjust the match parameters and teams to simulate different match scenarios.

- **Deployment**: Hosted on Streamlit Cloud for easy access and sharing.

## Usage

To make predictions, provide the following parameters when prompted:

- **Batting Team**: The team currently at bat.
- **Bowling Team**: The team currently bowling.
- **City**: The location of the match.
- **Current runs**: The current score of batting team.
- **Overs Completed**: The number of overs completed.
- **Wickets**: The number of wickets lost.
- **Target Runs**: The total runs scored by a bowling team.

The predictor will calculate the probability of the batting team winning based on these parameters and the current match situation.

## Technologies Used

This project leverages the following technologies:

- [Python](https://www.python.org/)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Streamlit](https://www.streamlit.io/)

## Installation

To run this project locally, follow these steps:

Clone the repository to your local machine.

Navigate to the project directory:
shell
Copy code
cd IPL 2024 Win Predictor

Install the required Python libraries:
shell
Copy code
pip install -r requirements.txt

Run the Streamlit app locally:
Go to gitbash
1. cd "D:\End To End Project\IPL 2024 Win Predictor"
2. dir
3. streamlit run app.py

Open the provided local URL in your web browser to access the IPL Win Predictor.

Usage
To make predictions, provide the current match situation including team performance, player statistics, and match conditions. The predictor will calculate the probability of a team winning.

Predict with Confidence
Explore the "IPL Win Predictor" and make data-driven predictions about IPL match outcomes. Get real-time insights and enhance your understanding of match dynamics. Visit the Live Demo and elevate your cricket analysis.

Contribute
If you'd like to contribute to this project or have suggestions for improvement, please feel free to submit issues or pull requests on GitHub.

Thank you for exploring the "IPL Win Predictor"! We hope this tool assists your IPL match predictions. 