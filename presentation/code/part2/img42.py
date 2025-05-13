import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation, PillowWriter
import uaibot as ub

# Configuration
num_frames = 200
square_size = 0.4
bg_color = '#191919'
color_A = '#81d41a'
color_B = '#5983b0'
dist_color = '#ffb66c'
smooth_dist_color = 'cyan'





A = np.matrix([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
b = np.matrix([square_size/2,square_size/2,square_size/2,square_size/2,square_size/2,square_size/2]).T
sq_objA = ub.ConvexPolytope(A=A,b=b,htm=ub.Utils.trn([0,0,0]),color='red')
sq_objB = ub.ConvexPolytope(A=A,b=b,htm=ub.Utils.trn([0,0,0]),color='blue')

sim = ub.Simulation([sq_objA, sq_objB])
fake_time = 0
hist_dist = []
hist_dist_smooth=[]

def create_constraints(vertices: np.ndarray):
    """
    Given 4x2 array of square vertices (in order), return matrices A and b
    such that the interior of the square satisfies A @ p <= b.
    """
    A = []
    b = []
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])  # outward normal
        normal = normal / np.linalg.norm(normal)

        # Ensure the normal points inward
        center = np.mean(vertices, axis=0)
        if (normal @ (center - p1)) < 0:
            normal = -normal

        A.append(np.append(normal, 0))
        b.append(normal @ p1)

    A.append(np.array([0, 0, 1]))
    b.append(-1)
    A.append(np.array([0, 0, -1]))
    b.append(-1)
    
    A = np.array(A)
    b = np.array(b).reshape(-1, 1)
    return np.matrix(A), np.matrix(b)

# Placeholder distance functions (to be replaced later)
def fun_dist(cA, cB, thetaA, thetaB):

    global fake_time
    global sq_objA
    global sq_objB
    global hist_dist
        
    cAm = [cA[0], 0, cA[1]]
    cBm = [cB[0], 0, cB[1]]
    
    sq_objA.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cAm)*ub.Utils.roty(thetaA)) 
    sq_objB.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cBm)*ub.Utils.roty(thetaB)) 
    
    fake_time+=0.01
    
    _, _, d, _ = sq_objA.compute_dist(sq_objB)
    
    hist_dist.append(0.5*(d**2))
    
    return 0.5*(d**2)

# def fun_dist_smooth(square_A, square_B):
#     A_A, b_A = create_constraints(square_A)
#     A_B, b_B = create_constraints(square_B)
    
#     objA = ub.ConvexPolytope(A=-A_A,b=-b_A)
#     objB = ub.ConvexPolytope(A=-A_B,b=-b_B)
    
#     _, _, d, _ = objA.compute_dist(objB, h=0.3,eps=0.1, no_iter_max=2000, tol=1e-6)
    
#     return 200*0.5*(d**2)

def fun_dist_smooth(cA, cB, thetaA, thetaB):

    global fake_time
    global sq_objA
    global sq_objB
    global hist_dist_smooth
    
    cAm = [cA[0], 0, cA[1]]
    cBm = [cB[0], 0, cB[1]]
    
    # objA = ub.Box(htm = ub.Utils.trn(cAm)*ub.Utils.roty(thetaA),width=square_size,depth=square_size,height=square_size)
    # objB = ub.Box(htm = ub.Utils.trn(cBm)*ub.Utils.roty(thetaB),width=square_size,depth=square_size,height=square_size)
    
    sq_objA.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cAm)*ub.Utils.roty(thetaA)) 
    sq_objB.add_ani_frame(time=fake_time,htm = ub.Utils.trn(cBm)*ub.Utils.roty(thetaB)) 
    
    fake_time+=0.01
    
    _, _, d, _ = sq_objA.compute_dist(sq_objB, h=0.2,eps=0.05, no_iter_max=2000, tol=1e-6)
    
    hist_dist_smooth.append(8.0*0.5*(d**2))
    
    return 8.0*0.5*(d**2)

def fun_dist_der(square_A, square_B):
    return 0.1 * np.sin(np.sum(square_A) + np.sum(square_B))

def fun_dist_smooth_der(square_A, square_B):
    return 0.1 * np.cos(np.sum(square_A) + np.sum(square_B))

def create_square(center, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * (square_size / 2)
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
ax0.set_xlim(-1.2, 1.2)
ax0.set_ylim(-0.8, 0.8)

square_A_patch = Polygon([[0, 0]], closed=True, color=color_A)
square_B_patch = Polygon([[0, 0]], closed=True, color=color_B)
ax0.add_patch(square_A_patch)
ax0.add_patch(square_B_patch)

# Tracking
dist_vals, smooth_dist_vals = [], []
dist_der_vals, smooth_dist_der_vals = [], []
x_vals = []

line_dist, = ax1.plot([], [], color=dist_color, label='Euclidean')
line_smooth_dist, = ax1.plot([], [], color=smooth_dist_color, label='Smooth')
line_dist_der, = ax2.plot([], [], color=dist_color, label='Euclidean')
line_smooth_dist_der, = ax2.plot([], [], color=smooth_dist_color, label='Smooth')

ax1.legend(facecolor=bg_color, edgecolor='white', labelcolor='white')
ax2.legend(facecolor=bg_color, edgecolor='white', labelcolor='white')

ax1.spines[:].set_color('white')
ax2.spines[:].set_color('white')

plt.tight_layout(pad=3.0)

def update(frame):
    t = frame / num_frames * 2 * np.pi
    center_A = np.array([-0.8, 0])
    center_B = np.array([0.4*np.sin(t), 0])
    angle_A = 2*t
    angle_B = 0

    square_A = create_square(center_A, angle_A)
    square_B = create_square(center_B, angle_B)

    square_A_patch.set_xy(square_A)
    square_B_patch.set_xy(square_B)



    d = fun_dist(center_A, center_B, angle_A, angle_B)
    ds = fun_dist_smooth(center_A, center_B, angle_A, angle_B)
    
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

    line_dist.set_data(x_vals, dist_vals)
    line_smooth_dist.set_data(x_vals, smooth_dist_vals)

    ax1.set_title("S2S-HSD", color="white")
    ax1.set_xlim(0, num_frames)
    ax1.set_ylim(-0.1, 0.35)

    line_dist_der.set_data(x_vals, dist_der_vals)
    line_smooth_dist_der.set_data(x_vals, smooth_dist_der_vals)
    ax2.set_title("Derivative of S2S-HSD", color="white")
    ax2.set_xlim(0, num_frames)
    ax2.set_ylim(-0.02, 0.02)

    return square_A_patch, square_B_patch, line_dist, line_smooth_dist, line_dist_der, line_smooth_dist_der

ani = FuncAnimation(fig, update, frames=num_frames, blit=True)


# Save as gif
output_path = "/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image42.gif"
ani.save(output_path, writer=PillowWriter(fps=20))


