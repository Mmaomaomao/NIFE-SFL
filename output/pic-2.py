import matplotlib.pyplot as plt
import numpy as np

# Data
grads_per_user = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]  # Number of gradients per user (in x 10^5)
non_verification_cost = [400, 600, 700, 850, 1000, 1150, 1300, 1450, 1600]  # Non-verification cost
total_cost = [450, 650, 750, 900, 1050, 1200, 1350, 1500, 1650]  # Total cost

# Create a bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.3  # Set width of the bars
index = np.arange(len(grads_per_user))  # the label locations

# Plot the bars for Non-Verification cost and Total cost
bar1 = ax.bar(index - bar_width/2, non_verification_cost, bar_width, label='Non-Verification cost', color='skyblue')
bar2 = ax.bar(index + bar_width/2, total_cost, bar_width, label='Total cost', color='salmon')

# Labeling
ax.set_xlabel('Number of gradients per user (x 10^5)', fontsize=12)
ax.set_ylabel('Total running time (ms)', fontsize=12)
# ax.set_title('Total Running Time vs Gradients per User', fontsize=14)

# X-axis ticks and labels
ax.set_xticks(index)
ax.set_xticklabels(grads_per_user, fontsize=12)

# Adding legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
