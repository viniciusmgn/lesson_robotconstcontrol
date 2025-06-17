import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

# Set the colors
bg_color = "white"
square1_color = "#118811"
square2_color = "#111188"
dist_color = "#AA0000"
deriv_color = "#440000"

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

def create_point_cloud(center = [0,0], variance=1.0, no_points=20):
    
    points = []
    
    for i in range(no_points-5):
        x = 2*variance*np.random.uniform()-1
        y = 2*variance*np.random.uniform()-1
        points.append( np.array([x,y]) + np.array(center))
        
    for i in range(5):
        points.append( np.array([-1+0.02*np.random.uniform(),-1+2*i/5+0.2*np.random.uniform()]) + np.array(center))
        
    return points
    
    
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
center2 = np.array([1.5, 0.5])

distances = []
derivatives = []

fig, axes = plt.subplots(3, 1, figsize=(6, 9), dpi=100)
fig.patch.set_facecolor(bg_color)
plt.subplots_adjust(hspace=0.4)
for ax in axes:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors="black", labelcolor="black")

square2 = create_point_cloud(center2, 0.6, 30)
def update(frame):
    for ax in axes:
        ax.clear()
        ax.set_facecolor(bg_color)
        ax.tick_params(colors="black", labelcolor="black")

    angle = angle_speed * frame
    

    square1 = create_square(center1, angle)
    

    cp1, cp2, dist = closest_points(square1, square2)
    distances.append(dist)
    deriv = (dist - distances[-2]) if frame > 0 else 0.0
    derivatives.append(deriv)

    # Plot squares
    axes[0].add_patch(Polygon(square1, closed=True, color=square1_color))
    axes[0].scatter([p[0] for p in square2], [p[1] for p in square2], color=square2_color, s=25)
    axes[0].plot(*zip(cp1, cp2), color="black", linestyle="--")
    axes[0].scatter(*cp1, color="black", s=20)
    axes[0].scatter(*cp2, color="black", s=20)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-2.5, 2.5)
    axes[0].set_ylim(-1, 1)
    axes[0].spines[:].set_color('white')


    axes[1].plot(distances, color=dist_color)
    axes[1].set_xlim(0, n_frames)
    axes[1].set_ylim(0.5, 1.2)
    axes[1].set_title("Distance", color="black")
    axes[1].set_facecolor(bg_color)
    axes[1].tick_params(colors="black", labelcolor="black")
    axes[1].spines[:].set_color('black')

    axes[2].plot(derivatives, color=deriv_color)
    axes[2].set_xlim(0, n_frames)
    axes[2].set_ylim(-0.012, 0.012)
    axes[2].set_title("Derivative of distance", color="black")
    axes[2].spines[:].set_color('black')

ani = FuncAnimation(fig, update, frames=n_frames, interval=100)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image38_B.gif", writer=PillowWriter(fps=20))
