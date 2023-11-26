import numpy as np
import matplotlib.pyplot as plt

# Read the data from the text file
file_path = 'output/EntireVAROS.txt'
data = np.loadtxt(file_path)

# Separate the columns into individual arrays
array1 = data[:, 0]
array2 = data[:, 1]
array3 = data[:, 2]
array4 = data[:, 3]

# Create four subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# Plot each array in a separate subplot
axs[0].plot(array1, label='LightGlue Rotation')
axs[1].plot(array2, label='LightGlue Translation')
axs[2].plot(array3, label='ORB+BF Rotation')
axs[3].plot(array4, label='ORB+BF Translation')

# Set Y-axis limits to 0 to 360 for all subplots
for ax in axs:
    ax.set_ylim([0, 365])

# Add labels and legend
axs[0].set_ylabel('Angular Error')
axs[1].set_ylabel('Angular Error')
axs[2].set_ylabel('Angular Error')
axs[3].set_ylabel('Angular Error')

axs[3].set_xlabel('Frame')

for ax in axs:
    ax.legend()

plt.tight_layout()
plt.show()