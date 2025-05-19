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

dt = 0.01

# === Curve and dynamics ===
def curve_C(t, center, radius):
    x = center[0] + radius * np.sin(t)
    y = center[1] + radius * np.cos(t)
    return np.stack([x, y], axis=0)

def next_config(t,q):
    
    #print("Time inner "+str(t))
    
    centers = [np.matrix([-0.5,0.7]).T, np.matrix([-0.5,-0.7]).T , np.matrix([0.0,0.0]).T]
    radius = [0.5,0.5,0.5]
    
    param_kp = 0.5
    param_eta = 0.6
    
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    psi, _, index = ub.Robot.vector_field(q,q_path, alpha=1.5, const_vel=0.5, is_closed = False)
    f = -2*psi
    
    A = np.matrix(np.zeros((0,2)))
    b = np.matrix(np.zeros((0,1)))
    
    for i in range(len(centers)):
        E = np.linalg.norm(q-centers[i])-radius[i]-0.05
        A = np.vstack( (A, (q-centers[i]).T/np.linalg.norm(q-centers[i]) ) )
        b = np.vstack( (b, -param_eta*(E)) )
  

    center_mov = np.matrix([0.5,0.5+0.3*np.sin(2*t)]).T
    radius_mov = 0.3
    ff = -np.matrix([0,0.6*np.cos(2*t)])*(q-center_mov)/np.linalg.norm(q-center_mov)
    ff = ff[0,0]
    E = np.linalg.norm(q-center_mov)-radius_mov-0.05
    A = np.vstack( (A, (q-center_mov).T/np.linalg.norm(q-center_mov)) )
    b = np.vstack( (b, -param_eta*(E)-ff) )
    

                        
    u = ub.Utils.solve_qp(H,f,A,b)

    dotE = (q-center_mov).T/np.linalg.norm(q-center_mov)*u + ff 
    
    E_next = np.linalg.norm(q+dt*u-np.matrix([0.5,0.5+0.3*np.sin(2*(t+dt))]).T)-radius_mov-0.05
    dot_E_est = (E_next-E)/dt
    
    #print("E = "+str(round(E,3))+", dotE = "+str(round(dotE[0,0],3))+", dotE_est = "+str(round(dot_E_est,3))+" dotE_min = "+str(round(-param_eta*(E),3)))
        
    return q+dt*u

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

circle_mov = patches.Circle((0.5,0.5), radius=0.3, color='#ec9ba4', zorder=15) 
ax.add_patch(circle_mov)


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
circle_mov_bound = ax.plot(C[0], C[1], color='red', linewidth=2, zorder=13)[0]

# Moving point and trajectory
point, = ax.plot([], [], 'o', color='#81d41a', markersize=10)
traj_line, = ax.plot([], [], color='white', linewidth=2, alpha=0.7, zorder = 100)

# Global state
q = np.matrix([-1.5, 0.2]).T
traj_x, traj_y = [], []
traj_circ = []

ax.plot([qp[0,0] for qp in q_path], [qp[1,0] for qp in q_path], color='#81d41a', linewidth=2, zorder=25)

def init():
    return point, traj_line, circle_mov, circle_mov_bound

def update(frame):
    global q, traj_x, traj_y, traj_circ 

    t = 8*frame*dt
    
    #print("Time outer = "+str(t))
    
    for i in range(8):
        q = next_config(t+i*dt,q)
    
    traj_x.append(q[0,0])
    traj_y.append(q[1,0])
    traj_line.set_data(traj_x, traj_y)
    point.set_data([q[0,0]], [q[1,0]])
    
    
    
    circle_mov.set_center((0.5, 0.5+0.3*np.sin(2*t)))
    circle_mov.set_radius(0.3)
    traj_circ = curve_C(t_vals,[0.5, 0.5+0.3*np.sin(2*t)],0.3)
    circle_mov_bound.set_data(traj_circ[0] ,traj_circ[1])
    
    return point, traj_line, circle_mov, circle_mov_bound

# Animation without blit (blit=True can fail silently in headless mode)
num_frames = 300
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

# Save the animation
writer = PillowWriter(fps=15)
ani.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image8.gif", writer=writer)

with Image.open("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image8.gif") as im:
    im.seek(0)  # Go to the first frame
    im.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image8_static.png")