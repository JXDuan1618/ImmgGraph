import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

# Read the new CSV file
file_path = r'/data/data_for_running/Figure2/merged_with_clusters.csv'
df = pd.read_csv(file_path)

# Select the required columns; assume the immune cell count column is 'Immune Cells Count' and the cluster column is 'new_cluster'
immune_cells_column = 'Immune Cells Count'
cluster_column = 'new_cluster'

# Group by new_cluster and get immune cell counts for each group
groups = df.groupby(cluster_column)[immune_cells_column].apply(list)

#Set figure size
plt.figure(figsize=(8, 8))

# Set box width and spacing
width = 1.0  # Box width
gap = 0.5    # Box spacing

#  Set the x-axis position for each box
positions = [1 + i * (width + gap) for i in range(len(groups))]

# Draw the boxplot
box = plt.boxplot(
    [group for group in groups],
    patch_artist=True,
    positions=positions,      # Set box positions
    widths=width,             # Set box width
    flierprops=dict(marker='*', color='red', markersize=1),
    showmeans=False,          # Do not show the mean
    meanline=False,
    medianprops=dict(color='blue', linewidth=3),
    whiskerprops=dict(linewidth=5),
    capprops=dict(linewidth=5),
    whis=100
)
# Print the type and contents of box for inspection
print(type(box))  # Check the type of box
print(box)        # Print the box contents to inspect keys and elements
# Configure box style
colors = ['#8583A9', '#CA8BA8', '#A0BDD5', '#EFC57F']  #Adjust colors as needed
for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
    patch.set_edgecolor('black')
    patch.set_facecolor(color)
    patch.set_linewidth(5)
    box['medians'][i].set_color('black')
    box['medians'][i].set_linewidth(5)

# Plot jittered scatter points
for i, group in enumerate(groups):
    y = np.array(group)
    x = np.full_like(y, fill_value=positions[i], dtype=float)  # Set the x-axis position for each box
    #  To avoid points aligning in a straight line, add a small random jitter
    x = x + np.random.uniform(-0.2, 0.2, size=len(x))  #Randomly jitter around the box position
    plt.plot(x, y, '.', color='black', markersize=10)

# Axes and grid settings
plt.ylabel("Immune Cells", fontsize=55, family='Arial', labelpad=30)
plt.grid(False)  # Do not show grid
plt.yticks(fontsize=55, family='Arial')
plt.tick_params(axis='y', which='both', direction='out', length=10, width=3)

# Set spine line widths
for spine in plt.gca().spines.values():
    spine.set_linewidth(4)

# Set x-axis limits and keep spacing
plt.xlim(min(positions) - 1, max(positions) + 1)  # Adjust x-axis limits based on box positions and width

plt.xticks([])  # Hide x-axis ticks

# Save the figure
out_path = r'/results/boxplot_1.png'
plt.savefig(out_path, dpi=600, bbox_inches="tight")
#plt.show()

print(f"Boxplot saved to: {out_path}")
