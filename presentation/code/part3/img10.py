import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub
import matplotlib.patches as patches
from PIL import Image

qF = np.matrix([1.5,0]).T

q_path_wp = [[-1.5, 0.2],[-1.5, 0.5],[-1.25,0.6],[-1.25,1.0],[-0.5,1.4],[0.5,0.6],[1.5,0]]
len_wp = len(q_path_wp)

fun_path = ub.Utils.interpolate(q_path_wp, is_closed = False)

n_max = 1000

q_path = fun_path([i/1000 for i in range(n_max)])

# === Curve and dynamics ===
def curve_C(t, center, radius):
    x = center[0] + radius * np.sin(t)
    y = center[1] + radius * np.cos(t)
    return np.stack([x, y], axis=0)

def next_state(q, qdot):
    
    centers = [np.matrix([-0.5,0.7]).T, np.matrix([-0.5,-0.7]).T , np.matrix([0.0,0.0]).T, np.matrix([0.5,0.5]).T]
    radius = [0.5,0.5,0.5,0.3]
    
    param_kv = 1.0
    param_eta = 0.8
    
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    psi, _, index = ub.Robot.vector_field(q,q_path, alpha=1.5, const_vel=0.5, is_closed = False)
    
    psi_next, _, _ = ub.Robot.vector_field(q+qdot*0.05,q_path, alpha=1.5, const_vel=0.5, is_closed = False)
    psi_ant, _, _ = ub.Robot.vector_field(q-qdot*0.05,q_path, alpha=1.5, const_vel=0.5, is_closed = False)
    ff = (psi_next-psi_ant)/0.1
    
    f = -2*(-param_kv*(qdot-psi)+ff)
    
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
    
    return q+0.05*qdot, qdot+0.05*u

# === Setup figure ===
fig, ax = plt.subplots()
fig.patch.set_facecolor('#191919')
ax.set_facecolor('#191919')
ax.axis('equal')
#ax.axis('off')
ax.set_xlim(-2, 2)
ax.set_ylim(-1.8, 2.2)
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

circle = patches.Circle((0.5,0.5), radius=0.3, color='#ec9ba4', zorder=15) 
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
C = curve_C(t_vals,[0.5,0.5],0.3)
ax.plot(C[0], C[1], color='red', linewidth=2, zorder=13)

# Moving point and trajectory
point, = ax.plot([], [], 'o', color='#81d41a', markersize=10)
traj_line, = ax.plot([], [], color='white', linewidth=2, alpha=0.7, zorder = 100)

# Global state
q = np.matrix([-1.5, 0.2]).T
qdot = np.matrix([1.0, -1.0]).T
traj_x, traj_y = [], []

ax.plot([qp[0,0] for qp in q_path], [qp[1,0] for qp in q_path], color='#81d41a', linewidth=2, zorder=25)

def init():
    return point, traj_line

def update(frame):
    global q, qdot, traj_x, traj_y

    q, qdot = next_state(q, qdot)
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
writer = PillowWriter(fps=15)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image10.gif", writer=writer)

with Image.open("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image10.gif") as im:
    im.seek(0)  # Go to the first frame
    im.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image10_static.png")