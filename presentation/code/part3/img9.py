import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub
import matplotlib.patches as patches
from PIL import Image

centers = [np.matrix([-0.2,0.5]).T, np.matrix([1.0,0]).T]
radius = [0.3,0.4]
    
qF = np.matrix([1.5,0.5]).T
# === Curve and dynamics ===
def curve_C(t, center, radius):
    x = center[0] + radius * np.sin(t)
    y = center[1] + radius * np.cos(t)
    return np.stack([x, y], axis=0)

def next_state(q, qdot):
    

    param_kp = 0.5
    param_eta = 0.5
    
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    f = 2*( 2*param_kp*qdot + (param_kp**2)*(q-qF))
    
    A = np.matrix(np.zeros((0,2)))
    b = np.matrix(np.zeros((0,1)))
    
    for i in range(len(centers)):
        dist = np.linalg.norm(q-centers[i])
        gradE = (q-centers[i]).T/dist
        E = dist-radius[i]-0.1
        dotE = gradE*qdot
        gamma = qdot.T*( np.eye(2) -gradE.T*gradE)*qdot/dist
        
        A = np.vstack( (A, gradE ) )
        b = np.vstack( (b, -2*param_eta*dotE-(param_eta**2)*E-gamma) )
    
    u = ub.Utils.solve_qp(H,f,A,b)
    
    return q+0.05*qdot, qdot + 0.05*u

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


# Static curve
t_vals = np.linspace(0, 2 * np.pi, 500)

for i in range(len(centers)):
    C = curve_C(t_vals,[centers[i][0,0],centers[i][1,0]],radius[i])
    ax.plot(C[0], C[1], color='#084594', linewidth=2)
    circle = patches.Circle((centers[i][0,0],centers[i][1,0]), radius=radius[i], color='#5983b0', zorder=10) 
    ax.add_patch(circle)



# Moving point and trajectory
point, = ax.plot([], [], 'o', color='#81d41a', markersize=10)
traj_line, = ax.plot([], [], color='white', linewidth=2, alpha=0.7)

# Global state
q = np.matrix([-1.5, 0.2]).T
qdot = np.matrix([0, 0]).T
traj_x, traj_y = [], []

def init():
    return point, traj_line

def update(frame):
    global q, qdot, traj_x, traj_y

    q, qdot = next_state(q, qdot)
    q, qdot = next_state(q, qdot)
    q, qdot = next_state(q, qdot)
    
    traj_x.append(q[0,0])
    traj_y.append(q[1,0])
    traj_line.set_data(traj_x, traj_y)
    point.set_data([q[0,0]], [q[1,0]])
    
    return point, traj_line

# Animation without blit (blit=True can fail silently in headless mode)
num_frames = 150
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

# Save the animation
writer = PillowWriter(fps=20)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image9.gif", writer=writer)

with Image.open("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image9.gif") as im:
    im.seek(0)  # Go to the first frame
    im.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image9_static.png")