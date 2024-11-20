#https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
#https://www.youtube.com/watch?v=VN3HJm3spRE
#https://www.geeksforgeeks.org/writing-csv-files-in-python/

import serial
import time
import csv 
import chardet
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


arduino = serial.Serial(port='COM4',  baudrate=115200, timeout=None)
time.sleep(1)

data = []

print("After this message, you will be prompted to type 'start' to start collecting Dynamometer data. After that, you have 10 seconds to get the dynamometer to full speed before data stops being collected.")
input("Type 'start' to begin collecting data: ")
start_time = time.time()


while True: 
    #while there no data at the serial port, don't try to read data(as readline is a hanging operation)
    while(arduino.in_waiting == 0):
        pass

    data_pt = arduino.readline()
    try:
        encoding = chardet.detect(data_pt)['encoding']
        decoded_data = data_pt.decode(encoding)
        decoded_data = decoded_data.rstrip("\n")
        decoded_data = decoded_data.split(',') 
    except:
        pass
    try:
        decoded_data = [float(number) for number in decoded_data] 
        data.append(decoded_data)
    except: 
        pass

    if(time.time()-start_time >=40):
        break

fields = ["RPM", "Timestamp"]
 
# Define the base name for the CSV files
base_filename = 'run'
file_extension = '.csv'

# Function to find the next available file name
def get_next_filename():
    i = 0
    while True:
        filename = f"{base_filename}{i}{file_extension}"
        if not os.path.exists(filename):
            return filename
        i += 1

# Get the next available CSV filename
csv_file = get_next_filename()

# Create a new CSV file and write to it
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers (optional)
    writer.writerow(fields)
    writer.writerows(data)

# degree = 20
# coeffs = np.polyfit(df['Timestamp'], df['RPM'], degree)
# polynomial = np.poly1d(coeffs) 

# polynomial_derivative = np.polyder(polynomial) 

# # # Step 4: Generate smooth values for plotting the polynomial and its derivative
# x_plotting_vals = np.linspace(df["Timestamp"].iloc[0], df["Timestamp"].iloc[-1], 100)
# y_plotting_vals = polynomial(x_plotting_vals)
# y_plotting_vals_deriv = polynomial_derivative(x_plotting_vals)

# # # Step 5: Plot the data, polynomial fit, and its derivative
# plt.figure(figsize=(10, 6))
# plt.scatter(df['RPM'], df['Timestamp'], color='red', label='Data points', zorder=5)  # Plot the data points
# plt.plot(x_plotting_vals, y_plotting_vals, label=f'Polynomial Fit (degree {degree})', color='blue')  # Polynomial fit
# plt.plot(x_plotting_vals, y_plotting_vals_deriv, label='Derivative of Polynomial', color='green', linestyle='--')  # Derivative
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'Polynomial Fit and its Derivative (degree {degree})')
# plt.grid(True)
# plt.show()

# # Step 6: Print out the polynomial equation and its derivative
# equation = f"Polynomial Equation (degree {degree}):\n"
# equation += "y = "
# terms = []
# for i, coeff in enumerate(coeffs):
#     power = degree - i
#     if power == 0:
#         terms.append(f"{coeff:.4f}")
#     elif power == 1:
#         terms.append(f"{coeff:.4f}x")
#     else:
#         terms.append(f"{coeff:.4f}x^{power}")
# equation += " + ".join(terms)

# # Print the polynomial equation
# print(equation)

# # Print the derivative of the polynomial
# derivative_terms = []
# for i, coeff in enumerate(polynomial_derivative.coefficients):
#     power = len(polynomial_derivative.coefficients) - i - 1
#     if power == 0:
#         derivative_terms.append(f"{coeff:.4f}")
#     elif power == 1:
#         derivative_terms.append(f"{coeff:.4f}x")
#     else:
#         derivative_terms.append(f"{coeff:.4f}x^{power}")

# derivative_equation = "Derivative of Polynomial:\n"
# derivative_equation += "dy/dx = "
# derivative_equation += " + ".join(derivative_terms)
# print(derivative_equation)
