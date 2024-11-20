import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate the logistic curve
x = np.linspace(-6, 6, 100)  # x values
L = 1                        # Maximum value of the logistic curve
k = 1                        # Growth rate
x0 = 0                       # Midpoint of the curve

# Logistic function
y_logistic = L / (1 + np.exp(-k * (x - x0)))  # Logistic curve

# Step 2: Add noise to the logistic curve
noise = np.random.normal(0, 0.1, size=y_logistic.shape)  # Gaussian noise
y_noisy = y_logistic + noise  # Noisy data

# Step 3: Fit a polynomial to the noisy data (degree 5)
degree = 5
coeffs = np.polyfit(x, y_noisy, degree)  # Fit polynomial of degree 5
polynomial = np.poly1d(coeffs)  # Create a polynomial function from the coefficients

# Step 4: Generate the smooth polynomial curve for plotting
y_poly_smooth = polynomial(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y_logistic, label='Logistic Curve', color='blue', linestyle='--')  # Original logistic curve
plt.scatter(x, y_noisy, color='red', label='Noisy Data')  # Noisy data points
plt.plot(x, y_poly_smooth, label=f'Polynomial Fit (degree {degree})', color='green', linestyle='-.')  # Polynomial fit
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Polynomial Fit to Noisy Logistic Curve (degree {degree})')
plt.grid(True)
plt.show()

# Print out the polynomial equation
equation = f"Polynomial Equation (degree {degree}):\n"
equation += "y = "
terms = []
for i, coeff in enumerate(coeffs):
    power = degree - i
    if power == 0:
        terms.append(f"{coeff:.4f}")
    elif power == 1:
        terms.append(f"{coeff:.4f}x")
    else:
        terms.append(f"{coeff:.4f}x^{power}")
equation += " + ".join(terms)
print(equation)
