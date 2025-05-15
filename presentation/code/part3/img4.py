import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

list_alpha = [0.8,3,25]
list_names = ['image4_A','image4_B','image4_C']

for ss in range(3):

    t = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    x_curve = np.cos(t)+0.3*np.sin(2*t)
    y_curve = np.sin(t)-0.1*np.cos(3*t)
    curve_points = np.vstack((x_curve, y_curve)).T

    # Dummy get_data function (replace with your own logic)
    def get_data(p):
        
        
        dists = np.linalg.norm(curve_points - p, axis=1)
        closest_idx = np.argmin(dists)
        min_dist = np.linalg.norm(curve_points[closest_idx] - p)
        proj = curve_points[closest_idx]
        t_proj = t[closest_idx]

        # Compute vectors
        tangent_vec = np.array([-np.sin(t_proj)+0.6*np.cos(2*t_proj), np.cos(t_proj)+0.3*np.sin(3*t_proj)])
        tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
        normal_vec = proj - p
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        G = (2/np.pi)*np.arctan(list_alpha[ss]*min_dist)
        H = np.sqrt(1.0-0.999*G**2)
        
        vel_vect = G*normal_vec+H*tangent_vec
        
        next_p = p + 0.1 * vel_vect
        
        return proj, 1.2*G*normal_vec, 1.2*H*tangent_vec, 1.2*vel_vect, next_p, G, H

    # Set up figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(1.5*8, 1.5*4.5)
    ax.set_position([0, 0, 1, 1])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#191919")
    ax.set_xlim(-6, 2.5)
    ax.set_ylim(-2.8, 2.8)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot static curve
    ax.plot(x_curve, y_curve, color="#5983b0")
    ax.text(0,0, r'$\mathcal{P}$', color='white', fontsize=30)

    # Prepare animation elements
    point_p = ax.scatter([], [], color="#81d41a", s=150, zorder=30)
    point_proj = ax.scatter([], [], color="#ec9ba4", s=150)
    traj_line, = ax.plot([], [], color='white', linewidth=1)
    dynamic_text = ax.text(-5.5, 2.0, "", color="white", fontsize=20)

    arrows = []

    # Initial point
    p_hist = []
    p_current = np.array([-5.5, 1.0])

    # Animation update function
    def update(frame):
        global p_current, arrows

        # Clear previous arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()

        # Get new data
        proj, normal_vec, tangent_vec, vel_vect, next_p, G, H = get_data(p_current)

        # Update scatter points
        point_p.set_offsets([p_current])
        point_proj.set_offsets([proj])

        # Update trajectory
        p_hist.append(p_current.copy())
        traj = np.array(p_hist)
        traj_line.set_data(traj[:, 0], traj[:, 1])
        dynamic_text.set_text(rf"G = "+str(round(G,4))+", H = "+str(round(H,4)))
        

        # Helper to draw arrows
        def draw_arrow(start, vec, color):
            arrow = ax.arrow(start[0], start[1], vec[0], vec[1], color=color,
                            head_width=0.15, length_includes_head=True)
            arrows.append(arrow)

        draw_arrow(p_current, normal_vec, "#ffb66c")
        draw_arrow(p_current, tangent_vec, "#ec2ed7")
        draw_arrow(p_current, vel_vect, "cyan")

        p_current = next_p
        return [point_p, point_proj, traj_line] + arrows

    # Create and save animation
    ani = animation.FuncAnimation(fig, update, frames=120, blit=False, interval=100)
    ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/"+list_names[ss]+".gif", writer='pillow', fps=10, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0})
