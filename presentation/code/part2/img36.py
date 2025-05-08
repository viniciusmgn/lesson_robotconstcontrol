import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib import gridspec


# Configuration
bg_color = '#191919'
color_shape1 = '#81d41a'
color_shape2 = '#5983b0'
n_frames = 300

# Define shapes
square = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
triangle = np.array([[0.0, -0.6], [0.6, 0.6], [-0.6, 0.6]])

# Create time vector
t = np.linspace(0, 1, n_frames)

# Motion functions
def motion_square(t):
    angle = np.pi * t
    trans = np.array([2 * t - 1.5, 0.5+0.2 * np.sin(2 * np.pi * t)])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (R @ square.T).T + trans

def motion_triangle(t):
    angle = -np.pi * t
    trans = np.array([1.5 - 2 * t, -0.9-0.2 * np.sin(2 * np.pi * t)])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (R @ triangle.T).T + trans

# Distance computation
def compute_distance(shape1, shape2):
    dists = np.linalg.norm(shape1[:, None, :] - shape2[None, :, :], axis=2)
    return 0.5*(np.min(dists)**2)

# Prepare figure
fig = plt.figure(figsize=(8, 6), facecolor=bg_color)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax_shapes = fig.add_subplot(gs[0])
ax_dist = fig.add_subplot(gs[1])

for ax in [ax_shapes, ax_dist]:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#191919')

ax_dist.spines[:].set_color('white')

ax_shapes.set_xlim(-3, 3)
ax_shapes.set_ylim(-2, 2)
ax_shapes.set_aspect('equal')
ax_shapes.set_xticks([])
ax_shapes.set_yticks([])

ax_dist.set_xlim(0, n_frames - 1)
ax_dist.set_ylim(0, 2)
ax_dist.set_xlabel('Frame', color='white')
ax_dist.set_ylabel('S2S-HSD', color='white')

line_dist, = ax_dist.plot([], [], color='#ec2ed7')
poly1 = Polygon(motion_square(0), color=color_shape1)
poly2 = Polygon(motion_triangle(0), color=color_shape2)
ax_shapes.add_patch(poly1)
ax_shapes.add_patch(poly2)

distances = []

def init():
    line_dist.set_data([], [])
    return poly1, poly2, line_dist

def update(frame):
    shape1 = motion_square(t[frame])
    shape2 = motion_triangle(t[frame])
    poly1.set_xy(shape1)
    poly2.set_xy(shape2)
    d = compute_distance(shape1, shape2)
    distances.append(d)
    line_dist.set_data(np.arange(len(distances)), distances)
    return poly1, poly2, line_dist

ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init, blit=True, repeat=False
)

ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image36.gif", writer='pillow', fps=20)
