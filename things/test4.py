import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Generate 100 quadratic data points: y = x^2
x_values = np.linspace(1, 10, 10)  # 100 points from x = 1 to x = 10
y_values = x_values ** 2  # y = x^2 for each x-value

# Store the data in a DataFrame
df = pd.DataFrame({'x': x_values, 'y': y_values})

def estimate_derivatives(df):
    # Initialize a list to store the first derivative estimates
    first_derivatives = []
    
    # Iterate over the rows in the dataframe (except the last one)
    for i in range(len(df) - 1):
        # Get x and y values of consecutive points
        x1, y1 = df.iloc[i]
        x2, y2 = df.iloc[i + 1]
        
        # Calculate the first derivative (slope) between points (x1, y1) and (x2, y2)
        slope = (y2 - y1) / (x2 - x1)
        
        # Calculate the x-value at which the derivative is estimated (midpoint)
        x_mid = (x1 + x2) / 2
        
        # Store the slope and the corresponding x-value
        first_derivatives.append((x_mid, slope))
    
    # Convert the list of first derivatives into a DataFrame
    first_derivatives_df = pd.DataFrame(first_derivatives, columns=["x", "y"])
    
    return first_derivatives_df


def estimate_second_derivative(first_derivatives_df):
    # Initialize a list to store the second derivative estimates
    second_derivatives = []
    
    # Iterate over the rows in the first_derivative dataframe (except the last one)
    for i in range(len(first_derivatives_df) - 1):
        # Get the first derivative (slope) values of consecutive points
        x1, slope1 = first_derivatives_df.iloc[i]
        x2, slope2 = first_derivatives_df.iloc[i + 1]
        
        # Calculate the second derivative (rate of change of the first derivative)
        second_derivative = (slope2 - slope1) / (x2 - x1)
        
        # Calculate the x-value at which the second derivative is estimated (midpoint)
        x_mid = (x1 + x2) / 2
        
        # Store the second derivative and the corresponding x-value
        second_derivatives.append((x_mid, second_derivative))
    
    # Convert the list of second derivatives into a DataFrame
    second_derivatives_df = pd.DataFrame(second_derivatives, columns=["x", "y"])
    
    return second_derivatives_df

# Estimate the first derivatives for the quadratic data
first_derivatives_df = estimate_derivatives(df)
print(first_derivatives_df)

# Estimate the second derivatives based on the first derivatives
second_derivatives_df = estimate_second_derivative(first_derivatives_df)


# Constant for multiplication (can be adjusted as needed)
constant = 5

# List to store the new x-values and y-values based on the specified logic
new_data = []

# Create an interpolation function
interp_func = interp1d(first_derivatives_df['x'], first_derivatives_df['y'], kind='linear', fill_value="extrapolate")

# Step through each x-value in the second derivative DataFrame
for x_val, second_derivative in second_derivatives_df.values:
    first_derivative_x_interp = interp_func(x_val)
    second_derivative_value = second_derivatives_df.loc[second_derivatives_df['x'] == x_val, 'y'].values[0]

    # # Multiply the x-value of the second derivative with the first derivative, then by the constant
    new_y_value = constant * second_derivative_value * first_derivative_x_interp
    
    # # Store the new x-value and calculated y-value
    new_data.append((x_val, new_y_value))

# Convert the new data into a DataFrame
new_df = pd.DataFrame(new_data, columns=["x", "new_y_value"])

# Display the new DataFrame
print(new_df.head())

# Plotting the original data points, first derivative, second derivative, and new y-values
plt.figure(figsize=(10, 8))

# Plot the original quadratic data (x, y)
plt.plot(df['x'], df['y'], label='Original Data (y = x^2)', color='blue', marker='o')

# Plot the first derivative estimates (x_mid, first_derivative)
plt.plot(first_derivatives_df['x'], first_derivatives_df['y'], label='First Derivative', color='red', marker='x')

# Plot the second derivative estimates (x_mid, second_derivative)
plt.plot(second_derivatives_df['x'], second_derivatives_df['y'], label='Second Derivative', color='green', marker='s')

# Plot the new y-values based on second derivative and first derivative
plt.plot(new_df['x'], new_df['new_y_value'], label='New Y-Values', color='purple', marker='d')

# Adding labels and title
plt.xlabel('x')
plt.ylabel('y / Derivative / New Y-Values')
plt.title('First and Second Derivative with New Y-Values from Interpolation')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
