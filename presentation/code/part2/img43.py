import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub


# Configuration
num_frames = 300
square_size = 0.4
bg_color = '#191919'
color_A = '#81d41a'
color_points = '#5983b0'
dist_color = '#ffb66c'
smooth_dist_color = 'cyan'

# Static point cloud
np.random.seed(0)
N = 50
point_cloud = np.random.randn(N, 2) * 0.15
cloud_center = np.mean(point_cloud, axis=0)

fake_time = 0
hist_dist = []
hist_dist_smooth = []
A = np.matrix([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
b = np.matrix([square_size/2,square_size/2,square_size/2,square_size/2,square_size/2,square_size/2]).T
sq_obj = ub.ConvexPolytope(A=A,b=b,htm=ub.Utils.trn([0,0,0]),color='red')
pc_obj = ub.PointCloud(points=[ np.matrix([a[0],0,a[1]]).T for a in point_cloud])

# Placeholder distance functions
def fun_dist(c, theta):
    global fake_time
    global sq_obj
    global pc_obj
    global hist_dist
        
    cm = [c[0], 0, c[1]]

    
    sq_obj.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cm)*ub.Utils.roty(theta)) 

    
    fake_time+=0.01
    
    _, _, d, _ = sq_obj.compute_dist(pc_obj)
    
    hist_dist.append(0.5*(d**2))
    
    return 0.5*(d**2)

def fun_dist_smooth(c, theta):
    global fake_time
    global sq_obj
    global pc_obj
    global hist_dist_smooth
        
    cm = [c[0], 0, c[1]]

    
    sq_obj.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cm)*ub.Utils.roty(theta)) 

    
    fake_time+=0.01
    
    _, _, d, _ = sq_obj.compute_dist(pc_obj, h=0.1, eps=0.05)
    
    hist_dist_smooth.append(5*0.5*(d**2))

    return 5*0.5*(d**2)

def fun_dist_der(square_A, square_B):
    return 0.1 * np.sin(np.sum(square_A) + np.sum(square_B))

def fun_dist_smooth_der(square_A, square_B):
    return 0.1 * np.cos(np.sum(square_A) + np.sum(square_B))

def create_square(center, angle):
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (square_size / 2)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return (R @ base.T).T + center

# Set up figure
fig, axes = plt.subplots(3, 1, figsize=(6, 9), dpi=100)
fig.patch.set_facecolor(bg_color)
for ax in axes:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors='white')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

ax0, ax1, ax2 = axes
ax0.axis('off')
ax0.axis('equal')
ax0.set_xlim(-2.9, 2.9)
ax0.set_ylim(-2.9, 2.9)

scatter = ax0.scatter(point_cloud[:, 0], point_cloud[:, 1], color=color_points, s=10)
square_patch = Polygon([[0, 0]], closed=True, color=color_A)
ax0.add_patch(square_patch)

ax1.spines[:].set_color('white')
ax2.spines[:].set_color('white')

# Data storage
dist_vals, smooth_dist_vals = [], []
dist_der_vals, smooth_dist_der_vals = [], []
x_vals = []

line_dist, = ax1.plot([], [], color=dist_color, label='Euclidean')
line_smooth_dist, = ax1.plot([], [], color=smooth_dist_color, label='Smooth')
line_dist_der, = ax2.plot([], [], color=dist_color, label='Euclidean')
line_smooth_dist_der, = ax2.plot([], [], color=smooth_dist_color, label='Smooth')

ax1.legend(facecolor=bg_color, edgecolor='white', labelcolor='white')
ax2.legend(facecolor=bg_color, edgecolor='white', labelcolor='white')

plt.tight_layout(pad=3.0)

def update(frame):
    t = frame / num_frames * 2 * np.pi
    R_outer = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    orbit_radius = 1.0
    orbit_center = cloud_center + R_outer @ np.array([orbit_radius, 0.0])/(1+0.15*t)
    square = create_square(orbit_center, angle=2 * t)

    square_patch.set_xy(square)

    d = fun_dist(orbit_center, 2 * t)
    ds = fun_dist_smooth(orbit_center, 2 * t)
    if len(hist_dist)>2:
        dd = hist_dist[-1]-hist_dist[-2] 
        dds =  hist_dist_smooth[-1]-hist_dist_smooth[-2]
    else:
        dd = 0
        dds = 0

    dist_vals.append(d)
    smooth_dist_vals.append(ds)
    dist_der_vals.append(dd)
    smooth_dist_der_vals.append(dds)
    x_vals.append(frame)
    
    ax1.set_title("S2S-HSD", color="white")
    line_dist.set_data(x_vals, dist_vals)
    line_smooth_dist.set_data(x_vals, smooth_dist_vals)
    ax1.set_xlim(0, num_frames)
    ax1.set_ylim(-0.02, 0.2)

    ax2.set_title("Derivative of S2S-HSD", color="white")
    line_dist_der.set_data(x_vals, dist_der_vals)
    line_smooth_dist_der.set_data(x_vals, smooth_dist_der_vals)
    ax2.set_xlim(0, num_frames)
    ax2.set_ylim(-0.01, 0.01)
    


    
    
    return square_patch, line_dist, line_smooth_dist, line_dist_der, line_smooth_dist_der

ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

# Save as GIF
output_path = "/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image43.gif"
ani.save(output_path, writer=PillowWriter(fps=20))


ub.Robot.create_franka_emika_3