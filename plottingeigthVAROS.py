import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches


# Read the data from the text file
file_path = 'output/EntireVAROS_everyfourth.txt'
data = np.loadtxt(file_path)
baseline_path = "output/baselines_between_every_8.txt"
baselines = np.loadtxt(baseline_path)

# Separate the columns into individual arrays
array1 = data[:, 0]
array2 = data[:, 1]
array3 = data[:, 2]
array4 = data[:, 3]
x_array = np.arange(0, 4715, 8)

#remove first element, not relevant
array1 = array1[1:]
array2 = array2[1:]
array3 = array3[1:]
array4 = array4[1:]
x_array = x_array[1:]

# Create four subplots
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# Plot each array in a separate subplot
axs[0].plot(x_array, array1, label='LightGlue Rotation')
axs[1].plot(x_array, array2,  label='LightGlue Translation')
axs[2].plot(x_array, array3, label='ORB+BF Rotation')
axs[3].plot(x_array, array4, label='ORB+BF Translation')


# Set Y-axis limits to 0 to 360 for all subplots
for ax in axs:
    ax.set_ylim([0, 365])



# Format Y-axis labels as degrees
for ax in axs:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-axis to integer values
    ax.set_yticklabels([f'{int(y)}°' for y in ax.get_yticks()])




# Add labels and legend
axs[0].set_ylabel('Angular Error')
axs[1].set_ylabel('Angular Error')
axs[2].set_ylabel('Angular Error')
axs[3].set_ylabel('Angular Error')

axs[3].set_xlabel('Timestamp')

for ax in axs:
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')




axs_sec = [ax.twinx() for ax in axs]

axs_sec[0].plot(x_array, baselines, label='Length of baseline', linestyle='--', color='orange')
axs_sec[1].plot(x_array,baselines,  label='Length of baseline', linestyle='--', color='orange')
axs_sec[2].plot(x_array,baselines,  label='Length of baseline', linestyle='--', color='orange')
axs_sec[3].plot(x_array,baselines, label='Length of baseline', linestyle='--', color='orange')

# Adjust the legend for the second set of axes
for ax in axs_sec:
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels)

# Set Y-axis limits to 0 to 2 for all subplots
for ax in axs_sec:
    ax.set_ylim([0, 1.5])

# Format Y-axis labels as meters
for ax in axs_sec:
    ax.set_yticklabels([f'{y} m' for y in ax.get_yticks()])


plt.tight_layout()

# Add black boxes around sequences where y-value is 360

for i in range(len(array1) - 1):
    if int(array1[i]) == -1:
        print(i)
        rect = patches.Rectangle((x_array[i], 0), 1, 365, linewidth=1, edgecolor='black', facecolor='black', zorder = 2)
        axs[0].add_patch(rect)

for i in range(len(array1) - 1):
    if int(array2[i]) == -1:
        print(i)
        rect = patches.Rectangle((x_array[i], 0), 1, 365, linewidth=1, edgecolor='black', facecolor='black', zorder = 2)
        axs[1].add_patch(rect)

for i in range(len(array1) - 1):
    if int(array3[i]) == -1:
        print(i)
        rect = patches.Rectangle((x_array[i], 0), 1, 365, linewidth=1, edgecolor='black', facecolor='black', zorder = 2)
        axs[2].add_patch(rect)

for i in range(len(array1) - 1):
    if int(array4[i]) == -1:
        print(i)
        rect = patches.Rectangle((x_array[i], 0), 1, 365, linewidth=1, edgecolor='black', facecolor='black', zorder = 2)
        axs[3].add_patch(rect)



plt.show()