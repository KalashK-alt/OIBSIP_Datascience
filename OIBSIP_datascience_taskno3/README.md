# Objective

The goal of this project is to build a machine learning model that can predict the price of a car based on its features such as brand, mileage, horsepower, fuel type, and transmission type.
This helps users and dealerships estimate fair market values before buying or selling vehicles.

# Steps Involved

Data Collection

Used a dataset (car data.csv) containing car attributes and corresponding prices.

Data Preprocessing

Cleaned missing values, handled duplicates, and outliers.

Encoded categorical features (fuel type, transmission, brand, etc.).

Scaled/normalized numerical features.

Exploratory Data Analysis (EDA)

Analyzed feature distributions and correlations with price.

Visualized important relationships using charts and plots.

Model Training

Trained ML models such as Linear Regression, Random Forest, and XGBoost.

Split dataset into training and test sets.

Tuned hyperparameters for better accuracy.

Model Evaluation

Evaluated performance using R² Score, MSE, and MAE.

Saved the best-performing pipeline model (car_price_pipeline.pkl).

Deployment

Built a Flask/Streamlit web app (app.py) for predictions.

Added a CLI tool (predict_cli.py) for command-line predictions.

# Tools & Technologies Used

Programming Language: Python

Libraries & Frameworks:

Data Handling: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn, xgboost

Deployment: Flask / Streamlit

Model Saving: joblib, pickle

#Outcome

Built a highly accurate car price prediction model.

Developed an interactive web application and CLI tool.

Helps users make data-driven pricing decisions for cars.
