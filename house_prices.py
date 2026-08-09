#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    sq_meters = np.array([80, 100, 120, 150, 180, 200]) # 'np.array' is numpy's way of storing lists. this is a list of the square meters of 6 houses
    prices = np.array([200000, 250000, 280000, 320000, 380000, 420000]) # this is a list of prices of the 6 houses stored in numpy's version of lists

    print("\bHouse List")
    print()
    print("1. R200 000 - (80 square meters)")
    print("2. R250 000 - (100 square meters)")
    print("3. R280 000 - (120 sqaure meters)")
    print("4. R320 000 - (150 sqaure meters)")
    print("5. R380 000 - (180 square meters)")
    print("6. R420 000 - (200 sqaure meters)")

    X = sq_meters.reshape(-1, 1)  # Take all my data and arrange it into as many rows as needed, but only 1 column. (6 rows, 1 column. the -1 is for detemining the number of rows needed
    y = prices
    # 'reshape' changs the data from a 1D to a 2D because that is what sklearn uses.

    print("Training the model...")
    model = LinearRegression() # creating a 'blank page' to work on called model
    model.fit(X, y) # using the inputs (X) and the output (y) to generate a formula 
    print("Done!")
    print()

    slope = model.coef_[0]          # Price increase per square meter
    intercept = model.intercept_    # Base price when sqm = 0

    print(f"Price = R{slope:.2f} × sqm + R{intercept:.2f}")
    print(f"R{intercept:.2f} is the base price when the house has 0 square meters.")
    print(f"For 1 sqm: R{slope * 1 + intercept:,.2f}")
    print()

    user_input = input("Enter the square meters of your house and let the algorithm predict it's price based off the house list above: ")
    new_sqm = int(user_input)

    new_house = np.array([[new_sqm]])
    predicted_price = model.predict(new_house) #makes guesses until the error is really small

    print(f"Predicted price for {new_sqm} sqm: R{predicted_price[0]:,.2f}")
    print()

if __name__ == "__main__":
    main()