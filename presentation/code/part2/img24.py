import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from itertools import product

# Define the Minkowski Difference function
def minkowski_difference(A, B):
    """Computes the Minkowski difference of two polygons A ⊖ B"""
    B_flipped = [(-x, -y) for x, y in B]  # Negate B for the difference
    minkowski_diff = [(ax + bx, ay + by) for (ax, ay), (bx, by) in product(A, B_flipped)]
    return Polygon(minkowski_diff).convex_hull.exterior.coords

# Define function to plot polygons in a subplot
def plot_polygons(ax, polygons, colors, title):
    """Plots multiple polygons in a given subplot"""
    ax.set_facecolor("#191919")  # Dark background

    # Set axis color to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='both', colors='white')

    for poly, color in zip(polygons, colors):
        poly = np.array(poly)
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2)
    
    ax.set_title(title, color='white')
    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(0, color='white', linewidth=1)
    ax.grid(False)

# Define example polygons
pol_A = [(0, 0), (2, 1), (1, 3), (-1, 2), (-2, 1)]
pol_B = [(1.5, 1), (2.5, 1), (2.5, 3), (1.5, 3)]

# Compute Minkowski difference
minkowski_diff = minkowski_difference(pol_A, pol_B)

# Create a figure with two subplots (stacked vertically)
fig, axs = plt.subplots(2, 1, figsize=(6, 12))
fig.patch.set_facecolor("#191919")  # Set background color for the entire figure

# Plot original polygons on the top subplot
plot_polygons(axs[0], [pol_A + [pol_A[0]], pol_B + [pol_B[0]]], ['#5983b0', '#ec2ed7'], "Original Polygons")

# Plot Minkowski difference on the bottom subplot
plot_polygons(axs[1], [minkowski_diff], ['#81d41a'], "Minkowski Difference")

plt.show()
