import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub
import matplotlib.patches as patches

qF = np.matrix([1.5,0]).T
# === Curve and dynamics ===
def curve_C(t, center, radius):
    x = center[0] + radius * np.sin(t)
    y = center[1] + radius * np.cos(t)
    return np.stack([x, y], axis=0)

def next_config(q):
    
    centers = [np.matrix([-0.5,0.7]).T, np.matrix([-0.5,-0.7]).T , np.matrix([0.0,0.0]).T]
    radius = [0.5,0.5,0.5]
    
    param_kp = 0.5
    param_eta = 0.3
    
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    f = 2*param_kp*(q-qF)
    
    A = np.matrix(np.zeros((0,2)))
    b = np.matrix(np.zeros((0,1)))
    
    for i in range(len(centers)):
        E = np.linalg.norm(q-centers[i])-radius[i]-0.1
        A = np.vstack( (A, (q-centers[i]).T/np.linalg.norm(q-centers[i]) ) )
        b = np.vstack( (b, -param_eta*(E)) )
    
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
circle = patches.Circle((-0.5,0.7), radius=0.5, color='#5983b0') 
ax.add_patch(circle)
circle = patches.Circle((-0.5,-0.7), radius=0.5, color='#5983b0') 
ax.add_patch(circle)


# Static curve
t_vals = np.linspace(0, 2 * np.pi, 500)
C = curve_C(t_vals,[-0.5,0.7],0.5)
ax.plot(C[0], C[1], color='#084594', linewidth=2)
C = curve_C(t_vals,[-0.5,-0.7],0.5)
ax.plot(C[0], C[1], color='#084594', linewidth=2)


circle = patches.Circle((0.0,0.0), radius=0.5, color='#5983b0', zorder=10) 
ax.add_patch(circle)
C = curve_C(t_vals,[0,0],0.5)
ax.plot(C[0], C[1], color='#084594', linewidth=2, zorder=12)

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
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image6.gif", writer=writer)
