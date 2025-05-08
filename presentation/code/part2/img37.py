import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

# Set the colors
bg_color = "#191919"
square1_color = "#81d41a"
square2_color = "#5983b0"
dist_color = "#ec2ed7"
deriv_color = "#ffb66c"

def create_square(center, angle=0.0, size=1.0, radius=0.2, resolution=40):

    if radius >= size / 2:
        raise ValueError("Corner radius must be smaller than half the square size.")
    
    h = size / 2
    r = radius
    arc = lambda start, end: np.array([
        [r * np.cos(t), r * np.sin(t)] for t in np.linspace(start, end, resolution)
    ])
    


    # Start at top-left, go clockwise
    points = []

    # Top-left arc (from left to top)
    points += [np.array([r+h, h*t]) for t in np.linspace(-1, 1, resolution)]
    points += (arc(0, np.pi/2) + [h, h]).tolist()
    points += [np.array([-t*h, r+h]) for t in np.linspace(-1, 1, resolution)]
    points += (arc(np.pi/2, np.pi) + [-h, h]).tolist()
    points += [np.array([-(r+h), -h*t]) for t in np.linspace(-1, 1, resolution)]
    points += (arc(np.pi, 3*np.pi/2) + [-h, -h]).tolist()
    points += [np.array([t*h, -(r+h)]) for t in np.linspace(-1, 1, resolution)]
    points += (arc(3*np.pi/2, 2*np.pi) + [h, -h]).tolist()
    

    # plt.plot([p[0] for p in points],[p[1] for p in points])
    # plt.show()
    
    
    points = np.array(points)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = points @ rotation.T
    return rotated + center

def closest_points(poly1, poly2):
    min_dist = np.inf
    cp1 = cp2 = None
    for p1 in poly1:
        for p2 in poly2:
            dist = np.linalg.norm(p1 - p2)
            if dist < min_dist:
                min_dist = dist
                cp1, cp2 = p1, p2
    return cp1, cp2, 0.5*(min_dist**2)

# Animation setup
n_frames = 300
angle_speed = 2 * np.pi / (2*n_frames)
center1 = np.array([-1.5, 0])
center2 = np.array([1.5, 0])

distances = []
derivatives = []

fig, axes = plt.subplots(3, 1, figsize=(6, 9), dpi=100)
fig.patch.set_facecolor(bg_color)
plt.subplots_adjust(hspace=0.4)
for ax in axes:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors="white", labelcolor="white")


def update(frame):
    for ax in axes:
        ax.clear()
        ax.set_facecolor(bg_color)
        ax.tick_params(colors="white", labelcolor="white")

    angle = angle_speed * frame
    

    square1 = create_square(center1, angle)
    square2 = create_square(center2, 0.0)

    cp1, cp2, dist = closest_points(square1, square2)
    distances.append(dist)
    deriv = (dist - distances[-2]) if frame > 0 else 0.0
    derivatives.append(deriv)

    # Plot squares
    axes[0].add_patch(Polygon(square1, closed=True, color=square1_color))
    axes[0].add_patch(Polygon(square2, closed=True, color=square2_color))
    axes[0].plot(*zip(cp1, cp2), color="white", linestyle="--")
    axes[0].scatter(*cp1, color="white", s=20)
    axes[0].scatter(*cp2, color="white", s=20)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-1, 1)
    axes[0].spines[:].set_color('#191919')


    axes[1].plot(distances, color=dist_color)
    axes[1].set_xlim(0, n_frames)
    axes[1].set_ylim(0.8, 1.6)
    axes[1].set_title("S2S-HSD", color="white")
    axes[1].set_facecolor(bg_color)
    axes[1].tick_params(colors="white", labelcolor="white")
    axes[1].spines[:].set_color('white')

    axes[2].plot(derivatives, color=deriv_color)
    axes[2].set_xlim(0, n_frames)
    axes[2].set_ylim(-0.03, 0.03)
    axes[2].set_title("Derivative of the S2S-HSD", color="white")
    axes[2].spines[:].set_color('white')

ani = FuncAnimation(fig, update, frames=n_frames, interval=100)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image37.gif", writer=PillowWriter(fps=20))
