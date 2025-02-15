import matplotlib.pyplot as plt
import numpy as np

# Define the number of epochs
epochs = np.arange(10, 100, 10)

# Simulate model accuracy values for each method
baseline = np.array([70, 72, 75, 78, 80, 82, 85, 87, 89])
ours = np.array([72, 74, 77, 80, 83, 85, 88, 90, 92])
pefl = np.array([71, 73, 76, 79, 82, 84, 87, 89, 91])
lus = np.array([69, 71, 74, 77, 80, 82, 85, 88, 90])
kumars = np.array([70, 72, 75, 78, 81, 83, 86, 88, 90])

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting each model's accuracy over epochs
plt.plot(epochs, baseline, label="Baseline", color='gray', linewidth=2)
plt.plot(epochs, ours, label="Ours", color='red', linewidth=2)
plt.plot(epochs, pefl, label="PEFL", color='#8B4513', linewidth=2)
plt.plot(epochs, lus, label="Lu's", color='#800080', linewidth=2)
plt.plot(epochs, kumars, label="Kumar's", color='#FF69B4', linewidth=2)

# Adding labels and title
# plt.title("Model Accuracy Over Epochs", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Model Accuracy (%)", fontsize=12)

# Adding a legend
plt.legend()

# Displaying the plot
plt.grid(True)
plt.show()
