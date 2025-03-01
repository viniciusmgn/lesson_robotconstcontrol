import uaibot as ub
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import quadprog




# URL of the .npz file
url = "https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/PointCloud/data_wall_with_hole.npy"

# Download the file
wallpoints = np.load(BytesIO(requests.get(url).content))
pc = ub.PointCloud(points = wallpoints, size=0.02, color='cyan')


point_link = ub.Ball(color='red',radius=0.02)
point_obj = ub.Ball(color='blue',radius=0.02)
robot = ub.Robot.create_franka_emika_3()
htm_des = ub.Utils.trn([0.45,0.34,0.7])*ub.Utils.rotx(-np.pi/2)
frame_des = ub.Frame(htm_des, size=0.1)
#sim = ub.Simulation.create_sim_grid([pc, robot, point_link, point_obj, frame_des])
sim = ub.Simulation.create_sim_lesson([pc, robot, point_link, point_obj, frame_des])
sim.set_parameters(width=500,height=500,load_screen_color='#191919',background_color='#191919')




def funF(r):
    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f
    
t = 0
dt = 0.02
eta = 0.3
K = 2.0

q_min = robot.joint_limit[:,0]
q_max = robot.joint_limit[:,1]

for i in range(round(35/dt)):
    
    qr = robot.q
    
    dr_obj =robot.compute_dist(obj = pc, q = qr, eps=0.003, h=0.003)
    
    A_obj = dr_obj.jac_dist_mat
    b_obj = dr_obj.dist_vect
    
    dr_auto =robot.compute_dist_auto(q = qr, eps=0.02, h=0.05)    

    A_auto = dr_auto.jac_dist_mat
    b_auto = dr_auto.dist_vect
        
    A_joint = np.matrix(np.vstack(  (np.identity(7), -np.identity(7))  ))
    b_joint = np.matrix(np.vstack(  (qr-q_min , q_max - qr)  )) 
    
    A =    np.matrix(np.vstack( (A_obj, A_auto, A_joint) ) )
    b =    -eta * np.matrix(np.vstack( (b_obj, b_auto, b_joint) ) )
    
    r, Jr = robot.task_function(htm_des, q=qr, mode='python')
    
    H = 2*(Jr.transpose() * Jr + 0.01 *np.identity(7))
    f = Jr.transpose() * K * funF(r)
    
    qdot = ub.Utils.solve_qp(H, f, A, b)

    qr += qdot*dt
    
    print("r = "+str(r.transpose())+", dr = "+str(dr_obj.get_closest_item().distance))
    
    if not np.all(b < 0):
        print("Error!")
    

    robot.add_ani_frame(t,qr)
    robot.update_col_object(t)
    point_link.add_ani_frame(t, htm = ub.Utils.trn(dr_obj.get_closest_item().point_link))
    point_obj.add_ani_frame(t, htm = ub.Utils.trn(dr_obj.get_closest_item().point_object ))    
                
    t+=dt
    

sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/intro/","intro_3")