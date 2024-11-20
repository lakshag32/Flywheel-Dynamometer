import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("run4.csv") #creates dataframe with floats

# Plot the x and y columns
plt.plot(df['Timestamp'], df['RPM'], marker='o', linestyle='-', color='b')  # Customize the style as needed
plt.xlabel('Timestamp')
plt.ylabel('RPM')
plt.title('RPM vs Time Plot')

# # Show the plot
plt.show()