import matplotlib.pyplot as plt
import numpy as np

# Data: Number of users and corresponding total running time for different dropout rates
users = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])

# Total running time for different dropout values (in ms)
dropout_0 = np.array([1, 1.2, 1.4, 1.8, 2.1, 2.4, 2.7, 3.1, 3.5])
dropout_10 = np.array([1.1, 1.3, 1.6, 1.9, 2.3, 2.7, 3.2, 3.6, 4.1])
dropout_20 = np.array([1.2, 1.5, 1.8, 2.2, 2.6, 3.1, 3.6, 4.0, 4.4])
dropout_30 = np.array([1.3, 1.6, 2.0, 2.3, 2.8, 3.3, 3.8, 4.2, 4.7])

# Plotting
plt.figure(figsize=(10, 6))

# Plot lines for different dropout rates with markers
plt.plot(users, dropout_0, 'bo-', label="dropout = 0%", markersize=6)  # Blue circles
plt.plot(users, dropout_10, 'ro-', label="dropout = 10%", markersize=6)  # Red circles
plt.plot(users, dropout_20, 'ks-', label="dropout = 20%", markersize=6)  # Black squares
plt.plot(users, dropout_30, 'm^-', label="dropout = 30%", markersize=6)  # Magenta triangles

# Add labels and title
plt.xlabel('Number of users', fontsize=12)
plt.ylabel('Total running time (ms)', fontsize=12)
# plt.title('Effect of Dropout on Total Running Time', fontsize=14)

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
