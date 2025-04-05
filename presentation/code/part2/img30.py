import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import uaibot as ub

# === Parameters ===
BOX1 = {
    "center": [0, 0],
    "angle": 0,
    "width": 3,
    "height": 1.5,
    "color": "#81d41a"
}

BOX2 = {
    "center": [0, 3],
    "angle": 40,
    "width": 4,
    "height": 1,
    "color": "#5983b0"
}

BG_COLOR = "#191919"
INITIAL_POINT = np.array([2.0, 2.0])
MARGIN = 1.0

# === Projection function ===
def projection_box(center, angle, width, height, point):
    box = ub.Box(
        htm=ub.Utils.trn([center[0], center[1], 0]) * ub.Utils.rotz(angle * np.pi / 180),
        width=width, depth=height, height=0.1
    )
    prj, _ = box.projection([point[0], point[1], 0])
    return [prj[0, 0], prj[1, 0]]

# === Compute rectangle corners (world coordinates) ===
def compute_corners(center, angle_deg, width, height):
    cx, cy = center
    angle_rad = np.radians(angle_deg)
    corners = np.array([
        [-width/2, -height/2],
        [ width/2, -height/2],
        [ width/2,  height/2],
        [-width/2,  height/2]
    ])
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated = (R @ corners.T).T + center
    return rotated

# === Get global axis limits ===
def compute_axis_limits(*boxes):
    all_points = []
    for box in boxes:
        corners = compute_corners(box["center"], box["angle"], box["width"], box["height"])
        all_points.append(corners)
    all_points = np.vstack(all_points)
    xmin, ymin = all_points.min(axis=0)
    xmax, ymax = all_points.max(axis=0)
    return (xmin - MARGIN, xmax + MARGIN, ymin - MARGIN, ymax + MARGIN)

# === Draw box ===
def draw_box(ax, center, angle, width, height, color):
    trans = Affine2D().rotate_deg_around(center[0], center[1], angle) + ax.transData
    rect = Rectangle((center[0] - width / 2, center[1] - height / 2),
                     width, height,
                     linewidth=2,
                     edgecolor=color,
                     facecolor='none',
                     transform=trans)
    ax.add_patch(rect)

# === Manual frame-by-frame rendering and saving ===
def render_frames(output_dir, num_frames):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    draw_box(ax, **BOX1)
    draw_box(ax, **BOX2)

    xmin, xmax, ymin, ymax = compute_axis_limits(BOX1, BOX2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    
    step_line, = ax.plot([], [], '--', linewidth=2, color='magenta')
    point_marker, = ax.plot([], [], 'o', markersize=8, color='#ffb66c')
    path_line, = ax.plot([], [], 'w-', linewidth=1, alpha=0.7)

    current_point = INITIAL_POINT.copy()
    path = [current_point.copy()]

    for frame in range(num_frames):
        box = BOX1 if frame % 2 == 0 else BOX2
        projected = projection_box(
            box["center"], box["angle"], box["width"], box["height"], current_point
        )

        # Update visuals
        point_marker.set_data([projected[0]], [projected[1]])
        step_line.set_data([current_point[0], projected[0]], [current_point[1], projected[1]])


        xs, ys = zip(*path) if len(path) > 1 else ([path[0][0]], [path[0][1]])
        path_line.set_data(xs, ys)

        # Save frame
        fig.canvas.draw()
        fig.savefig(f"{output_dir}/image30_{frame}.svg", dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())

        # Update state
        path.append(projected.copy())
        current_point = projected

    plt.close(fig)

# === Run ===
render_frames("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2", num_frames=8)
