import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Environment setup

np.random.seed(103)

num_obstacles = 15
obstacle_radius_range = (0.6, 0.9)
xlim, ylim = (0, 10), (0, 10)
p_connect_goal = 0.2

obstacles = []
for _ in range(num_obstacles):
    while True:
        center = np.random.uniform([xlim[0]+0.3, ylim[0]+0.3], [xlim[1]-0.3, ylim[1]-0.3])
        radius = np.random.uniform(*obstacle_radius_range)
        if (xlim[0] + radius < center[0] < xlim[1] - radius) and (ylim[0] + radius < center[1] < ylim[1] - radius):
            obstacles.append((center, radius))
            break

def is_collision(p1, p2):
    num_points = 50
    points = np.linspace(p1, p2, num_points)
    for point in points:
        for center, radius in obstacles:
            if np.linalg.norm(point - center) <= radius:
                return True
    return False

def generate_valid_point(pt=None):
    while True:
        point = np.random.uniform([xlim[0]+0.3, ylim[0]+0.3], [xlim[1]-0.3, ylim[1]-0.3])
        collision = any(np.linalg.norm(point - center) <= radius + 0.2 for center, radius in obstacles)
        if not collision:
            if pt is None or np.linalg.norm(pt - point) > 8:
                return point

start = generate_valid_point()
goal = generate_valid_point(start)
print("Start and Goal found")

path = [start]
frames = []

# Build path (random walk)
found_goal = False
while not found_goal and len(path) < 500:
    if np.random.rand() < p_connect_goal:
        candidate = goal
    else:
        angle = np.random.uniform(0, 2 * np.pi)
        step_size = 1.5
        direction = np.array([np.cos(angle), np.sin(angle)])
        candidate = path[-1] + direction * step_size
        candidate = np.clip(candidate, [xlim[0], ylim[0]], [xlim[1], ylim[1]])

    if not is_collision(path[-1], candidate):
        path.append(candidate.copy())
        frames.append(np.array(path))
        if np.linalg.norm(candidate - goal) < 0.5:
            path.append(goal.copy())
            frames.append(np.array(path))
            found_goal = True

# Post-processing: shortcutting with visualization
i = 0
while i < len(path) - 2:
    shortcut_made = False
    for j in range(len(path) - 1, i + 1, -1):
        if not is_collision(path[i], path[j]):
            # Frame showing the shortcut attempt (highlight the yellow line)
            temp_path = np.array(path)
            shortcut_line = np.array([path[i], path[j]])
            frames.append((temp_path, shortcut_line.copy()))
            # Apply shortcut
            path = path[:i + 1] + path[j:]
            frames.append(np.array(path))  # Frame showing the updated path
            shortcut_made = True
            break
    if not shortcut_made:
        i += 1

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('#191919')
ax.set_facecolor('#191919')
ax.set_xlim([xlim[0]-0.3, xlim[1]+0.3])
ax.set_ylim([ylim[0]-0.3, ylim[1]+0.3])
ax.axis('off')

def update(frame):
    ax.clear()
    ax.set_facecolor('#191919')
    ax.set_xlim([xlim[0]-0.3, xlim[1]+0.3])
    ax.set_ylim([ylim[0]-0.3, ylim[1]+0.3])
    ax.axis('off')

    # Draw obstacles
    for center, radius in obstacles:
        C = [[center[0] + radius * np.cos(theta) for theta in np.linspace(0, 2*np.pi, 300)],
             [center[1] + radius * np.sin(theta) for theta in np.linspace(0, 2*np.pi, 300)]]
        ax.plot(C[0], C[1], color='#084594', linewidth=2, zorder=12)
        circle = plt.Circle(center, radius, color='#5983b0')
        ax.add_patch(circle)

    # Start and Goal
    ax.plot(*start, 'o', markersize=10, color='#81d41a')
    ax.plot(*goal, 'o', markersize=10, color='#ec9ba4')

    # Frame content
    if isinstance(frame, tuple):
        path_points, shortcut_line = frame
        if len(path_points) > 1:
            ax.plot(path_points[:, 0], path_points[:, 1], color='white')
        ax.plot(shortcut_line[:, 0], shortcut_line[:, 1], color='yellow', linewidth=2, linestyle='--', zorder=15)
    else:
        if len(frame) > 1:
            ax.plot(frame[:, 0], frame[:, 1], color='white')
            ax.plot(frame[-1, 0],frame[-1, 1], 'o', markersize=10, color='#ffb66c')


ani = FuncAnimation(fig, update, frames=frames, repeat=False)
output_path = "/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image45.gif"
ani.save(output_path, writer=PillowWriter(fps=5))

print(f"Saved animation to {output_path}")
