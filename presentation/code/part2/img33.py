import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
output_dir = "bvh_with_tree"
os.makedirs(output_dir, exist_ok=True)

bg_color = '#191919'
list_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080'
]

# Hardcoded points
points = np.array([
    [0.37454012, 0.95071431],
    [0.73199394, 0.59865848],
    [0.15601864, 0.15599452],
    [0.05808361, 0.86617615],
    [0.60111501, 0.70807258],
    [0.02058449, 0.96990985],
    [0.83244264, 0.21233911],
    [0.30424224, 0.52475643]
])

bounding_boxes = []  # (min_x, max_x, min_y, max_y, color_idx, point_ids)
tree_structure = []  # (parent_id, child_id)
frame_counter = 0
box_id_counter = 0

def draw_all_boxes(ax):
    for i, (min_x, max_x, min_y, max_y, color_idx, _) in enumerate(bounding_boxes):
        color = list_colors[color_idx % len(list_colors)]
        ax.add_patch(
            plt.Rectangle((min_x - 0.02, min_y - 0.02), (max_x - min_x) + 0.04, (max_y - min_y) + 0.04,
                          fill=False, edgecolor=color, linewidth=2)
        )



def draw_tree(ax_tree, current_id, x=0.5, y=0.85, x_span=0.5, level=0, parent_pos=None, label_text=""):
    node = bounding_boxes[current_id]
    color = list_colors[node[4] % len(list_colors)]
    point_ids = node[5]

    # Format point indices as a string
    label_str = ",".join(str(pid) for pid in point_ids)

    # Estimate box size based on label length
    base_char_width = 0.035
    width = max(0.08, len(label_str) * base_char_width)
    height = 0.06

    # Draw node rectangle with fixed fill, colored border
    ax_tree.add_patch(plt.Rectangle((x - width / 2, y - height / 2), width, height,
                                    facecolor='#191919', edgecolor=color, linewidth=2))
    ax_tree.text(x, y, label_str, ha='center', va='center', fontsize=14, color='white')

    # Draw arrow from parent to this node
    if parent_pos is not None:
        ax_tree.annotate("",
            xy=(x, y + height / 2),
            xytext=parent_pos,
            arrowprops=dict(arrowstyle="->", color="white", lw=1))

        # Compute midpoint and direction
        mid_x = (x + parent_pos[0]) / 2
        mid_y = (y + height / 2 + parent_pos[1]) / 2
        arrow_dx = x - parent_pos[0]
        label_align = 'right' if arrow_dx < 0 else 'left'
        label_offset = -0.02 if arrow_dx < 0 else 0.02

        # Draw arrow label
        ax_tree.text(mid_x + label_offset, mid_y,
                    label_text, ha=label_align, va='center', fontsize=10, color='white')
            
    # Draw leaf children (individual points) as text-only
    if len(point_ids) <= 2:
        offset = -0.15
        for i, pid in enumerate(point_ids):
            dx = (i - (len(point_ids) - 1) / 2) * x_span * 0.5
            x_child = x + dx
            y_child = y + offset

            # Arrow to point
            ax_tree.annotate("",
                xy=(x_child, y_child + 0.01),
                xytext=(x, y - height / 2),
                arrowprops=dict(arrowstyle="->", color="white", lw=1))

            # Compute horizontal alignment and label position
            arrow_dx = x_child - x
            label_align = 'right' if arrow_dx < 0 else 'left'
            label_offset = -0.02 if arrow_dx < 0 else 0.02

            # Draw text label next to the arrow (e.g., "leaf")
            ax_tree.text((x + x_child) / 2 + label_offset,
                        (y + y_child) / 2,
                        "leaf", ha=label_align, va='center', fontsize=10, color='white')

            # White text, no box
            ax_tree.text(x_child, y_child-0.02, str(pid), ha='center', va='center', fontsize=14, color='white')
        return

    # Continue with child boxes
    children = [cid for pid, cid in tree_structure if pid == current_id]
    if not children:
        return

    n = len(children)
    
    labels = ['1','left','down','up','right','down','up','8']
    for i, child_id in enumerate(children):
        new_x = x + (i - (n - 1) / 2) * x_span
        new_y = y - 0.15
        draw_tree(ax_tree, child_id, x=new_x, y=new_y, x_span=x_span / 1.5, level=level + 1, parent_pos=(x, y - height / 2), label_text = labels[child_id])

def build_bvh(points_subset, depth, point_ids, parent_id=None):
    global frame_counter, box_id_counter

    if len(points_subset) == 0:
        return

    # Compute bounding box
    min_x, min_y = np.min(points_subset, axis=0)
    max_x, max_y = np.max(points_subset, axis=0)

    box_id = box_id_counter
    bounding_boxes.append((min_x, max_x, min_y, max_y, box_id, point_ids))
    if parent_id is not None:
        tree_structure.append((parent_id, box_id))
    box_id_counter += 1

    # Create plot
    fig, (ax_main, ax_tree) = plt.subplots(1, 2, figsize=(12, 6), facecolor=bg_color,
                                           gridspec_kw={'width_ratios': [2, 1]})
    fig.subplots_adjust(wspace=0.3)

    # Main 2D plot
    ax_main.set_facecolor(bg_color)
    ax_main.scatter(points[:, 0], points[:, 1], c='white', s=40)
    for i, (x, y) in enumerate(points):
        ax_main.text(x + 0.01, y + 0.01, str(i), fontsize=14, color='white')
    draw_all_boxes(ax_main)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')


    # Tree visualization
    ax_tree.set_facecolor(bg_color)
    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.axis('off')

    draw_tree(ax_tree, 0)  # always draw from root

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2", f"image33_{frame_counter:03d}.svg"), dpi=150, facecolor=bg_color)
    plt.close(fig)
    frame_counter += 1

    if len(points_subset) <= 2:
        return

    # Alternate split axis
    axis = depth % 2
    sorted_points = points_subset[points_subset[:, axis].argsort()]
    sorted_ids = [x for _, x in sorted(zip(points_subset[:, axis], point_ids), key=lambda x: x[0])]
    mid = len(sorted_points) // 2

    left, right = sorted_points[:mid], sorted_points[mid:]
    left_ids, right_ids = sorted_ids[:mid], sorted_ids[mid:]

    build_bvh(left, depth + 1, left_ids, parent_id=box_id)
    build_bvh(right, depth + 1, right_ids, parent_id=box_id)

# Run BVH build
point_ids = list(range(len(points)))
build_bvh(points, depth=0, point_ids=point_ids)
