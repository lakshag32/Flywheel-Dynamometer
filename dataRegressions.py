import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample discrete data points (replace with your own data)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 6, 14, 28, 45])

# Function to calculate R²
def calculate_r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Linear Regression
def linear_regression(x, y):
    coefficients = np.polyfit(x, y, 1)
    a, b = coefficients
    y_pred = a * x + b
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Linear regression: y = {a:.2f}x + {b:.2f}, R² = {r_squared:.4f}")
    return a, b, r_squared

# Quadratic Regression
def quadratic_regression(x, y):
    coefficients = np.polyfit(x, y, 2)
    a, b, c = coefficients
    y_pred = a * x**2 + b * x + c
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Quadratic regression: y = {a:.2f}x² + {b:.2f}x + {c:.2f}, R² = {r_squared:.4f}")
    return a, b, c, r_squared

# Cubic Regression
def cubic_regression(x, y):
    coefficients = np.polyfit(x, y, 3)
    a, b, c, d = coefficients
    y_pred = a * x**3 + b * x**2 + c * x + d
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Cubic regression: y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x + {d:.2f}, R² = {r_squared:.4f}")
    return a, b, c, d, r_squared

# Quartic Regression
def quartic_regression(x, y):
    coefficients = np.polyfit(x, y, 4)
    a, b, c, d, e = coefficients
    y_pred = a * x**4 + b * x**3 + c * x**2 + d * x + e
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Quartic regression: y = {a:.2f}x⁴ + {b:.2f}x³ + {c:.2f}x² + {d:.2f}x + {e:.2f}, R² = {r_squared:.4f}")
    return a, b, c, d, e, r_squared

# Exponential Regression
def exponential_regression(x, y):
    def model(x, a, b):
        return a * np.exp(b * x)
    
    popt, _ = curve_fit(model, x, y, p0=(1, 0.1))
    a, b = popt
    y_pred = a * np.exp(b * x)
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Exponential regression: y = {a:.2f} * exp({b:.2f} * x), R² = {r_squared:.4f}")
    return a, b, r_squared

# Logarithmic Regression
def logarithmic_regression(x, y):
    def model(x, a, b):
        return a * np.log(b * x)
    
    popt, _ = curve_fit(model, x, y, p0=(1, 1))
    a, b = popt
    y_pred = a * np.log(b * x)
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Logarithmic regression: y = {a:.2f} * log({b:.2f} * x), R² = {r_squared:.4f}")
    return a, b, r_squared

# Power Regression
def power_regression(x, y):
    def model(x, a, b):
        return a * x**b
    
    popt, _ = curve_fit(model, x, y, p0=(1, 1))
    a, b = popt
    y_pred = a * x**b
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Power regression: y = {a:.2f} * x^{b:.2f}, R² = {r_squared:.4f}")
    return a, b, r_squared

# Logistic Regression
def logistic_regression(x, y):
    def model(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    popt, _ = curve_fit(model, x, y, p0=(max(y), 1, np.median(x)))
    L, k, x0 = popt
    y_pred = L / (1 + np.exp(-k * (x - x0)))
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Logistic regression: y = {L:.2f} / (1 + exp(-{k:.2f} * (x - {x0:.2f}))), R² = {r_squared:.4f}")
    return L, k, x0, r_squared

# Sinusoidal Regression
def sinusoidal_regression(x, y):
    def model(x, A, B, C, D):
        return A * np.sin(B * (x - C)) + D
    
    popt, _ = curve_fit(model, x, y, p0=(1, 1, 0, 0))
    A, B, C, D = popt
    y_pred = A * np.sin(B * (x - C)) + D
    r_squared = calculate_r_squared(y, y_pred)
    print(f"Sinusoidal regression: y = {A:.2f} * sin({B:.2f} * (x - {C:.2f})) + {D:.2f}, R² = {r_squared:.4f}")
    return A, B, C, D, r_squared

# Function to plot the regressions
def plot_regressions(x, y):
    plt.scatter(x, y, color='red', label='Data points')

    # Linear regression
    a, b, r_squared = linear_regression(x, y)
    y_linear = a * x + b
    plt.plot(x, y_linear, label=f'Linear fit: R² = {r_squared:.4f}')

    # Quadratic regression
    a, b, c, r_squared = quadratic_regression(x, y)
    y_quadratic = a * x**2 + b * x + c
    plt.plot(x, y_quadratic, label=f'Quadratic fit: R² = {r_squared:.4f}')

    # Cubic regression
    a, b, c, d, r_squared = cubic_regression(x, y)
    y_cubic = a * x**3 + b * x**2 + c * x + d
    plt.plot(x, y_cubic, label=f'Cubic fit: R² = {r_squared:.4f}')

    # Quartic regression
    a, b, c, d, e, r_squared = quartic_regression(x, y)
    y_quartic = a * x**4 + b * x**3 + c * x**2 + d * x + e
    plt.plot(x, y_quartic, label=f'Quartic fit: R² = {r_squared:.4f}')

    # Exponential regression
    a, b, r_squared = exponential_regression(x, y)
    y_exponential = a * np.exp(b * x)
    plt.plot(x, y_exponential, label=f'Exponential fit: R² = {r_squared:.4f}')

    # Logarithmic regression
    a, b, r_squared = logarithmic_regression(x, y)
    y_logarithmic = a * np.log(b * x)
    plt.plot(x, y_logarithmic, label=f'Logarithmic fit: R² = {r_squared:.4f}')

    # Power regression
    a, b, r_squared = power_regression(x, y)
    y_power = a * x**b
    plt.plot(x, y_power, label=f'Power fit: R² = {r_squared:.4f}')

    # Logistic regression
    L, k, x0, r_squared = logistic_regression(x, y)
    y_logistic = L / (1 + np.exp(-k * (x - x0)))
    plt.plot(x, y_logistic, label=f'Logistic fit: R² = {r_squared:.4f}')

    # Sinusoidal regression
    A, B, C, D, r_squared = sinusoidal_regression(x, y)
    y_sinusoidal = A * np.sin(B * (x - C)) + D
    plt.plot(x, y_sinusoidal, label=f'Sinusoidal fit: R² = {r_squared:.4f}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Various Polynomial and Non-Linear Regression Fits')
    plt.grid(True)
    plt.show()

# Plot the results
plot_regressions(x, y)
