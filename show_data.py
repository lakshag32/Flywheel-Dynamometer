import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d


csv_to_read = input("what csv to read?: ")
angle_df = pd.read_csv(csv_to_read)

# #remove repeated data and convert time from ms to s
angle_df = angle_df.drop_duplicates(subset=['total_angle'])
angle_df['timestamp'] = angle_df['timestamp']/1000

# # Generate 100 quadratic data points: y = x^2
# x_values = np.linspace(1, 10, 10)  # 100 points from x = 1 to x = 10
# y_values = x_values ** 2  # y = x^2 for each x-value

# # Store the data in a DataFrame
# angle_df = pd.DataFrame({'timestamp': x_values, 'total_angle': y_values})
# Set random seed for reproducibility
#np.random.seed(42)

# Generate 137 x values evenly spaced
#x_values = np.linspace(0, 10, 2000)

# Generate noisy parabola: y = x^2 + noise
#noise = np.random.normal(0, 20, 2000)  # Add some Gaussian noise
#y_values = x_values**2 + noise

# Create the DataFrame
#angle_df = pd.DataFrame({
#    'timestamp': x_values,
#    'total_angle': y_values
#})

angle_df['total_angle'] = angle_df['total_angle'].rolling(window=80).mean()

def estimate_derivatives(df):
    first_derivatives = []
    
    for i in range(len(df) - 1):
        x1, y1 = df.iloc[i]
        x2, y2 = df.iloc[i + 1]
        
        slope = (y2 - y1) / (x2 - x1)
        
        x_mid = (x1 + x2) / 2
        
        first_derivatives.append((x_mid, slope))
    
    first_derivatives_df = pd.DataFrame(first_derivatives, columns=["x", "y"])
    
    return first_derivatives_df


def estimate_second_derivative(first_derivatives_df):
    second_derivatives = []
    
    for i in range(len(first_derivatives_df) - 1):
        x1, slope1 = first_derivatives_df.iloc[i]
        x2, slope2 = first_derivatives_df.iloc[i + 1]
        
        second_derivative = (slope2 - slope1) / (x2 - x1)
        
        x_mid = (x1 + x2) / 2
        
        second_derivatives.append((x_mid, second_derivative))
    
    second_derivatives_df = pd.DataFrame(second_derivatives, columns=["x", "y"])
    
    return second_derivatives_df

velocity_df = estimate_derivatives(angle_df)
velocity_df['y'] = velocity_df['y'].rolling(window=240).mean()

acceleration_df = estimate_second_derivative(velocity_df)
acceleration_df['y'] = acceleration_df['y'].rolling(window=80).mean()


moment_of_inertia = 0.006767

power_time_data = []
power_rpm_data = []

interp_func = interp1d(velocity_df['x'], velocity_df['y'], kind='linear', fill_value="extrapolate")

for timestamp, acceleration in acceleration_df.values:
    velocity_interp = interp_func(timestamp)

    new_y_value = moment_of_inertia * acceleration * velocity_interp
    
    power_time_data.append((timestamp, new_y_value))

power_df = pd.DataFrame(power_time_data, columns=["x", "y"])
power_df['y'] = power_df['y'].rolling(window=160).mean()


plt.figure(figsize=(10, 8))

plt.plot(angle_df['timestamp'], angle_df['total_angle'], label='Total Angle(rad) vs Time(s)', color='blue', marker='o')
plt.plot(velocity_df['x'], velocity_df['y'], label='Velocity(rad/s) vs Time(s)', color='red', marker='o')
plt.plot(acceleration_df['x'], acceleration_df['y'], label='Acceleration(rad/s^2) vs Time(s)', color='green', marker='o')
plt.plot(power_df['x'], power_df['y'], label='Power(watts) vs Time(s)', color='purple', marker='o')

plt.xlabel('Time(sec)')
plt.ylabel('Angle/Velocity/Acceleration/Power')
plt.title('Angle(rad)/Velocity(rad/s)/Acceleration(rad/s^2)/Power(watts) vs Time')
plt.legend()

plt.grid(True)
plt.show()
