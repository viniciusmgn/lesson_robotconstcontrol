import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
plt.rcParams['text.usetex'] = True

# Define the curve (e.g., sine wave)
t = np.linspace(-2 * np.pi, 2 * np.pi, 400)
x_curve = np.cos(t)+0.3*np.sin(2*t)
y_curve = np.sin(t)-0.1*np.cos(3*t)
curve_points = np.vstack((x_curve, y_curve)).T

# Define point p
p = np.array([2, 2])

# Find projection (closest point on the curve)
dists = np.linalg.norm(curve_points - p, axis=1)
closest_idx = np.argmin(dists)
proj = curve_points[closest_idx]
t_proj = t[closest_idx]

# Compute vectors
tangent_vec = np.array([-np.sin(t_proj)+0.6*np.cos(2*t_proj), np.cos(t_proj)+0.3*np.sin(3*t_proj)])
tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
normal_vec = proj - p
normal_vec = normal_vec / np.linalg.norm(normal_vec)

# Colors
bg_color = "#191919"
curve_color = "#5983b0"
point_color = "#81d41a"
point_color_dark = "#ec9ba4"
line_color = "white"
normal_color = "#ffb66c"
tangent_color = "#ec2ed7"

x_scale = 4.5
y_scale = 3

# Generate and save each frame
for step in range(5):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8*y_scale/x_scale)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.set_position([0, 0, 1, 1]) 

    ax.plot(x_curve, y_curve, color=curve_color)
    ax.text(x_curve[-1]+0.3, y_curve[-1], r'$\mathcal{C}$', color='white', fontsize=25)
    ax.scatter(p[0], p[1], color=point_color, s=150, zorder=10)
    ax.text(p[0] + 0.1, p[1], r'$p$', color='white', fontsize=25)
    
    ax.scatter(proj[0]+2, proj[1], color='#191919', s=150, zorder=10)

    if step >= 1:
        ax.scatter(p[0], p[1], color=point_color, s=150)
        ax.text(proj[0]-0.7, proj[1]-0.1, r'$\Pi_{\mathcal{C}}(p)$', color='white', fontsize=25)
        ax.scatter(proj[0], proj[1], color=point_color_dark, s=150, zorder=10)
        ax.plot([p[0], proj[0]], [p[1], proj[1]], linestyle='dotted', color=line_color, zorder=-1)
        
        

    if step >= 2:
        arrow_normal = p + 0.8 * normal_vec
        ax.arrow(p[0], p[1], arrow_normal[0] - p[0], arrow_normal[1] - p[1],
                 color=normal_color, head_width=0.1, length_includes_head=True)
        ax.text(( arrow_normal[0]+0.1) , (arrow_normal[1]-0.1) ,
                r'$N_{\mathcal{C}}(p)$', color='white', fontsize=25)

    if step >= 3:
        arrow_tangent = proj - 0.8 * tangent_vec
        ax.arrow(proj[0], proj[1], arrow_tangent[0] - proj[0], arrow_tangent[1] - proj[1],
                 color=tangent_color, head_width=0.1, length_includes_head=True)
        ax.text((arrow_tangent[0]+0.1) , (arrow_tangent[1]-0.1) ,
                r'$T_{\mathcal{C}}(p)$', color='white', fontsize=25)
        
        
        # Normalize to make sure they are unit vectors
        n = normal_vec / np.linalg.norm(normal_vec)
        t = tangent_vec / np.linalg.norm(tangent_vec)

        # Define the square size
        scale = 0.2

        # Define square corners
        corner0 = proj
        corner1 = proj - scale * t
        corner2 = corner1 - scale * n
        corner3 = proj - scale * n

        square = np.array([corner0, corner1, corner2, corner3, corner0])  # closed loop

        # Center dot
        center = proj - 0.5 * scale * (n + t)


        ax.set_facecolor("#191919")

        # Draw square
        ax.plot(square[:, 0], square[:, 1], color='white')

        # Draw dot
        ax.plot(center[0], center[1], 'o', color='white')

    if step >= 4:
        arrow_tangent = p - 0.8 * tangent_vec
        ax.arrow(p[0], p[1], arrow_tangent[0] - p[0], arrow_tangent[1] - p[1],
                 color=tangent_color, head_width=0.1, length_includes_head=True)
        ax.text((arrow_tangent[0]+0.1) , (arrow_tangent[1]-0.1) ,
                r'$T_{\mathcal{C}}(p)$', color='white', fontsize=25)
        
        
        # Normalize to make sure they are unit vectors
        n = normal_vec / np.linalg.norm(normal_vec)
        t = tangent_vec / np.linalg.norm(tangent_vec)

        # Define the square size
        scale = 0.2

        # Define square corners
        corner0 = p
        corner1 = p - scale * t
        corner2 = corner1 + scale * n
        corner3 = p + scale * n

        square = np.array([corner0, corner1, corner2, corner3, corner0])  # closed loop

        # Center dot
        center = p + 0.5 * scale * (n - t)


        ax.set_facecolor("#191919")

        # Draw square
        ax.plot(square[:, 0], square[:, 1], color='white')

        # Draw dot
        ax.plot(center[0], center[1], 'o', color='white')
        
    ax.set_xlim(-x_scale, x_scale)
    ax.set_ylim(-y_scale, y_scale)
    ax.axis('off')
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save as SVG
    fig.savefig(f"/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image1_Slide{step + 1}.svg", format='svg',  bbox_inches=None, pad_inches=0)
    plt.close(fig)
