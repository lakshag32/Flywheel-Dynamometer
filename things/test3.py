import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Step 1: Generate a noise-free set of sine function points
x = np.linspace(0, 10, 20)  # 20 points from 0 to 10
y = np.sin(x)  # True sine values (noise-free)

# Step 2: Apply a very small moving average (window size = 3)
window_size = 3  # Extremely small window to closely match the actual function
y_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='same')

# Step 3: Fit cubic splines to the smoothed data
spline = CubicSpline(x, y_smoothed, bc_type='natural')

# Step 4: Take the derivative of each cubic spline (First derivative)
spline_derivative = spline.derivative(1)  # First derivative of the spline

# Step 5: Construct the final derivative function
# This function evaluates the first derivative of the spline at any point
def final_derivative(x_val):
    return spline_derivative(x_val)

# Step 6: Evaluate and plot everything
x_fine = np.linspace(0, 10, 1000)  # Fine grid for smooth plotting
y_spline = spline(x_fine)  # Evaluated spline
y_derivative = final_derivative(x_fine)  # Evaluated first derivative

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the original sine data and the smoothed data
plt.subplot(2, 1, 1)
plt.plot(x, y, label='Original Sine Function', color='blue', linewidth=2)
plt.plot(x, y_smoothed, label='Smoothed Data (Moving Average)', color='red', linestyle='--', linewidth=2)
plt.plot(x_fine, y_spline, label='Cubic Spline Fit', color='green', linewidth=2)
plt.title('Cubic Spline Fitting with Sine Function (No Noise)')
plt.legend()

# Plot the derivative of the spline (First Derivative)
plt.subplot(2, 1, 2)
plt.plot(x_fine, y_derivative, label="Spline Derivative (First Derivative)", color='purple', linewidth=2)
plt.title('Derivative of the Cubic Spline')
plt.legend()

plt.tight_layout()
plt.show()
