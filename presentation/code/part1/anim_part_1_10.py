from uaibot import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

width=550
height=500

robot = Robot.create_abb_crb()
robot = Robot.create_kuka_kr5()

htm_des = Utils.trn([0.2,0.3,-0.1])*robot.fkm()*Utils.rotx(np.pi/2)
htm_des = Utils.trn([0.1,0.15,-0.1])*robot.fkm()*Utils.rotx(np.pi/2)

frame_des = Frame(size=0.1, htm=htm_des)
light0 = PointLight(name="light0", color="white", intensity=14, htm=Utils.trn([ 0,0,5.]))
light1 = PointLight(name="light1", color="white", intensity=14, htm=Utils.trn([ 5.,0,0]))
light2 = PointLight(name="light2", color="white", intensity=14, htm=Utils.trn([ -5.,0,0]))
light3 = PointLight(name="light3", color="white", intensity=14, htm=Utils.trn([ 0,5.,0]))
light4 = PointLight(name="light4", color="white", intensity=14, htm=Utils.trn([ 0,-5.,0]))

sim = Simulation([robot, frame_des, light0, light1, light2, light3, light4], load_screen_color="#191919", background_color="#191919", width=500, height=500,
                 camera_type="orthographic", show_grid = False, show_world_frame=False)

dt=0.01
K=0.5
tmax=25
Nmax = round(tmax/dt)


hist_r = []
hist_t = []

for i in range(Nmax):
    t = i*dt
    
    r, jac_r = robot.task_function(q=robot.q, htm_des = htm_des)
 
 
    r_new = np.matrix(r)
    r_new[0,0] = r[0,0]/(1e-6+np.sqrt(abs(r[0,0])))
    r_new[1,0] = r[1,0]/(1e-6+np.sqrt(abs(r[1,0])))
    r_new[2,0] = r[2,0]/(1e-6+np.sqrt(abs(r[2,0])))
    r_new[3,0] = r[3,0]/(1e-6+np.sqrt(abs(r[3,0])))
    r_new[4,0] = r[4,0]/(1e-6+np.sqrt(abs(r[4,0])))
    r_new[5,0] = r[5,0]/(1e-6+np.sqrt(abs(r[5,0])))   
       

    qdot = Utils.dp_inv_solve(jac_r,-K*r_new)
    
    robot.add_ani_frame(t, q=robot.q+qdot*dt)
    
    r[0,0] = 1000*abs(r[0,0])
    r[1,0] = 1000*abs(r[1,0])
    r[2,0] = 1000*abs(r[2,0])
    r[3,0] = (180/np.pi)*np.arccos(1-r[3,0])
    r[4,0] = (180/np.pi)*np.arccos(1-r[4,0])
    r[5,0] = (180/np.pi)*np.arccos(1-r[5,0])
    
    hist_r.append(r)
    hist_t.append(t)
    
  

r_values = np.hstack(hist_r).T  

t_values = np.array(hist_t)

# Define the background color
bg_color = "#191919"

# Create the figure and subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.patch.set_facecolor(bg_color)

# Plot the first three entries of r vs t

axs[0].plot(t_values, r_values[:, 0], label='x', linewidth=2)
axs[0].plot(t_values, r_values[:, 1], label='y', linewidth=2)
axs[0].plot(t_values, r_values[:, 2], label='z', linewidth=2)

axs[0].set_ylabel("Position error (mm)")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# Plot the last three entries of r vs t
axs[1].plot(t_values, r_values[:, 3], label='x', linewidth=2)
axs[1].plot(t_values, r_values[:, 4], label='y', linewidth=2)
axs[1].plot(t_values, r_values[:, 5], label='z', linewidth=2)

axs[1].set_xlabel("Time (t)")
axs[1].set_ylabel("Orientation error (deg)")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)

# Set colors for text, ticks, and labels
for ax in axs:
    ax.set_facecolor(bg_color)
    ax.tick_params(axis="both", colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(facecolor=bg_color, edgecolor="white", labelcolor="white")
    

plt.show()
sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/","part_1_10")