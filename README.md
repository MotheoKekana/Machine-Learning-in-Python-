# House Price Predictor

A simple machine learning program that predicts house prices based on square meters using **Linear Regression**.


## Code Breakdown


**1. Import** - import numpy as np (Imports NumPy for arrays)
              - from sklearn.linear_model import LinearRegression (Imports the ML model)


**2. Data** - sq_meters = np.array([80, 100, ...]) (Stores house sizes)
            - prices = np.array([200000, 250000, ...]) (Stores house prices)


**3. Reshape** - X = sq_meters.reshape(-1, 1) (Converts data to 2D format)


**4. Train** - model = LinearRegression() (Creates an empty model)
             - model.fit(X, y) (Trains the model (finds the best line))


**5. Learn** - slope = model.coef_[0] (Gets the price per square meter)
             - intercept = model.intercept_ (Gets the base price)


**6. Predict** - new_house = np.array([[user_input]]) (Converts user input to 2D)
               - model.predict(new_house) (Predicts the price)


## What This Project Does

This program demonstrates the basics of machine learning by:

1. Taking 6 houses with known square meters and prices (hardcoded).
2. Training a Linear Regression model to find the relationship between size and price.
3. Predicting the price of a new house from user input square meters based on what it learned.

The model learns a simple formula:
- **Price = (slope × square_meters) + intercept**


## How To Run THis Project

1. install numpy and scikit-learn (pip install numpy scikit-learn - in terminal).
2. Clone the repository from GitHub.
3. Open the file in VS Code and run.
