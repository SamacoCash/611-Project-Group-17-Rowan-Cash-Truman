# 611-Project-Group-17-Rowan-Cash-Truman

Premier League Match Prediction Model Project
Project Overview
This project aims to develop machine learning models that leverage historical English Premier League (Premier League) match statistics to predict the outcome and final score of upcoming football fixtures. Our proposed client is a Premier League betting application, and the goal is to provide them with more accurate and detailed match insights.

Research Objectives and Prediction Tasks
The project is structured around two core machine learning tasks:
1. Match Outcome Classification Task
  - Goal: To predict the final result of the match.
  - Class Definitions:
    - 0: Home Team Win
    - 1: Away Team Win
    - 2: Draw
2. Final Score Regression Task
- Goal: To predict the final score for both the home and away teams, offering deeper insights beyond a simple win/loss prediction.

Dataset
- Source: Premier League Matchup Stats data (Kaggle)
- Seasons Covered:
  - 2019/2020 Season
  - 2020/2021 Season
  - 2021/2022 Season
- Data Preparation: The datasets from the three seasons will be concatenated to form a single, comprehensive training data source.

Models and Methodology
Classification Task: Predicting Match Outcome
(Classes: 0 = Home Win, 1 = Away Win, 2 = Draw)
- Classification (outcome): SVM, Random Forest, Logistic Regression
Regression Task: Predicting Final Scores:
- Regression (score): Linear Regression, Random Forest Regressor, Gradient Boosting

Evaluation and Visualization
Evaluation Metrics
- Classification (Outcome):
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Regression (Score):
  - Root Mean Square Error (RMSE)
  - Coefficient of Determination (R2)

Visualization
- Classification: Confusion Matrices and model comparison charts.
- Regression: Predicted vs. Actual Score Scatter Plots.
- Combined Analysis: Outcome prediction accuracy segmented by score margin.


Getting Started
To run this project, you will need to:
1. Download the Premier League Matchup Stats dataset from Kaggle.
2. Install Python and necessary machine learning libraries (e.g., Scikit-learn, Pandas, Matplotlib).
3. Execute the project script, following the steps outlined in the Framework section for data processing, model training, and evaluation.
