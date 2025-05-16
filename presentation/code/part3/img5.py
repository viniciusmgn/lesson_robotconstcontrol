import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub
import matplotlib.patches as patches

qF = np.matrix([1.5,0]).T
# === Curve and dynamics ===
def curve_C(t):
    x = np.sin(t)
    y = np.cos(t)
    return np.stack([x, y], axis=0)

def next_config(q):
    
    param_kp = 0.5
    param_eta = 0.3
    
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    f = 2*param_kp*(q-qF)
    
    E = np.linalg.norm(q)-1-0.05
    
    A = np.matrix(q.T/np.linalg.norm(q))
    b = np.matrix([-param_eta*(E)])
    
    u = ub.Utils.solve_qp(H,f,A,b)
    
    return q+0.05*u

# === Setup figure ===
fig, ax = plt.subplots()
fig.patch.set_facecolor('#191919')
ax.set_facecolor('#191919')
ax.axis('equal')
#ax.axis('off')
ax.set_xlim(-2, 2)
ax.set_ylim(-1.8, 1.8)
ax.tick_params(axis='both', colors='white')              # Tick marks and numbers
ax.xaxis.label.set_color('white')                        # X-axis label
ax.yaxis.label.set_color('white')                        # Y-axis label
ax.spines['bottom'].set_color('white')                   # Bottom border
ax.spines['top'].set_color('white')                      # Top border
ax.spines['left'].set_color('white')                     # Left border
ax.spines['right'].set_color('white')                    # Right border
ax.scatter([qF[0,0]],[qF[1,0]], color='magenta',s=40)
circle = patches.Circle((0.0,0.0), radius=1.0, color='#5983b0')  # Solid fill with your color
ax.add_patch(circle)

# Static curve
t_vals = np.linspace(0, 2 * np.pi, 500)
C = curve_C(t_vals)
ax.plot(C[0], C[1], color='#084594', linewidth=2)

# Moving point and trajectory
point, = ax.plot([], [], 'o', color='#81d41a', markersize=10)
traj_line, = ax.plot([], [], color='white', linewidth=2, alpha=0.7)

# Global state
q = np.matrix([-1.5, 0.2]).T
traj_x, traj_y = [], []

def init():
    return point, traj_line

def update(frame):
    global q, traj_x, traj_y

    q = next_config(q)
    traj_x.append(q[0,0])
    traj_y.append(q[1,0])
    traj_line.set_data(traj_x, traj_y)
    point.set_data([q[0,0]], [q[1,0]])
    
    return point, traj_line

# Animation without blit (blit=True can fail silently in headless mode)
num_frames = 250
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

# Save the animation
writer = PillowWriter(fps=20)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image5.gif", writer=writer)
