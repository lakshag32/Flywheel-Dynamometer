import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

#ask user what data file they would like to process
csv_to_read = input("what csv to read?: ")
angle_df = pd.read_csv(csv_to_read)

#remove repeated data and convert time from ms to s
angle_df = angle_df.drop_duplicates(subset=['total_angle'])
angle_df['timestamp'] = angle_df['timestamp']/1000
#Remove noise from the data by generating points from a moving average of the data
angle_df['total_angle'] = angle_df['total_angle'].rolling(window=80).mean()
#keep only rows that have a timestamp and total angle displacement value
angle_df = angle_df[angle_df.count(axis=1) == 2]

#function to estimate derivative values using 
def estimate_derivative(df):
    derivative = []
    
    for i in range(len(df) - 1):
        x1, y1 = df.iloc[i]
        x2, y2 = df.iloc[i + 1]
        
        slope = (y2 - y1) / (x2 - x1)
        
        x_mid = (x1 + x2) / 2
        
        derivative.append((x_mid, slope))
    
    derivative_df = pd.DataFrame(derivative, columns=["x", "y"])
    
    return derivative_df

#generate points for the derivative of the total angle vs time function.
velocity_df = estimate_derivative(angle_df)
#filter with moving average
velocity_df['y'] = velocity_df['y'].rolling(window=240).mean()
#drop rows that don't have a timestamp and velocity value
velocity_df = velocity_df[velocity_df.count(axis=1) == 2]

#repeat for acceleration
acceleration_df = estimate_derivative(velocity_df)
acceleration_df['y'] = acceleration_df['y'].rolling(window=80).mean()
acceleration_df = acceleration_df[acceleration_df.count(axis=1) == 2]

moment_of_inertia = 0.006767

power_time_data = []
power_velocity_data = []

#create pieceswise interpoltaion for velocity. Ensures that we get a velocity value for any x-value we ask the function for
interp_func = interp1d(velocity_df['x'], velocity_df['y'], kind='linear', fill_value="extrapolate")

#loop for calculating power values
for timestamp, acceleration in acceleration_df.values:
    velocity_interp = interp_func(timestamp)

    new_y_value = moment_of_inertia * acceleration * velocity_interp
    
    power_time_data.append((timestamp, new_y_value))
    power_velocity_data.append((velocity_interp,new_y_value))

power_time_df = pd.DataFrame(power_time_data, columns=["x", "y"])
power_time_df['y'] = power_time_df['y'].rolling(window=160).mean()
power_time_df = power_time_df[power_time_df.count(axis=1) == 2]


power_velocity_df = pd.DataFrame(power_velocity_data, columns=["x", "y"])
power_velocity_df['y'] = power_velocity_df['y'].rolling(window=160).mean()
power_velocity_df = power_velocity_df[power_velocity_df.count(axis=1) == 2]


plt.figure(figsize=(10, 8))

plt.plot(power_velocity_df['x'], power_velocity_df['y'], label='Power(watts) vs Velocity(rad/s)', color='orange', marker='o')

plt.xlabel('Velocity(rad/sec)')
plt.ylabel('Power(watts)')
plt.title('Power(watts) vs Velocity(rad/sec)')
plt.legend()

plt.grid(True)
plt.show()

print("NaN values in each column:")
print(power_time_df.isna().sum())

# Drop NaN values (if any)
df = power_time_df.dropna()

# Ensure correct data types
power_time_df['x'] = pd.to_numeric(power_time_df['x'], errors='coerce')
power_time_df['y'] = pd.to_numeric(power_time_df['y'], errors='coerce')

# Calculate the integral using trapezoidal rule
integral = np.trapz(power_time_df['y'], power_time_df['x'])

print("Integral:", integral)