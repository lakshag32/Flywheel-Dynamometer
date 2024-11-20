import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic erratic data (sine wave with noise)
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data) + np.random.normal(0, 0.5, size=x_data.size)

# Apply a Simple Moving Average (SMA) with a window size of 5
window_size = 5
smoothed_data = np.convolve(y_data, np.ones(window_size)/window_size, mode='same')

# Compute the derivative of the smoothed data using central difference
dy_dx = np.gradient(smoothed_data, x_data)  # Numerically compute the derivative

# Plot original data, smoothed data, and derivative
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(x_data, y_data, label="Original Data", color='blue')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x_data, smoothed_data, label="Smoothed Data (SMA)", color='green')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x_data, dy_dx, label="Derivative of Smoothed Data", color='red')
plt.legend()

plt.tight_layout()
plt.show()
