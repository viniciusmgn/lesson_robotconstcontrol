


import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt
from helper_anim_part_3_ex2 import *

    
###################################   
#Parameters control
param_eta_static = 0.2
param_eta_human = 0.3
param_eta_tray = 0.15
param_eta_auto = 0.15
param_eps = 0.005
param_k = 0.3
param_kr = 1.0
param_kox = 0.3
param_max_dotq = 1.5
param_tol_error = 0.007   
param_decay_alpha = 0.9
param_min_alpha = 0.02

#Parameters distance computation and safety
param_no_iter_max_dist = 3000
param_max_dist_static = 0.6
param_max_dist_human = 2.5
param_joint_delta = 2*np.pi/180
param_use_smooth_dist = True



if param_use_smooth_dist:
    #When using smooth distances, all 'delta' 
    #should be smaller, because smooth distances
    #are smaller than Euclidean when objects are close
    param_h_dist = 0.05
    param_eps_dist = 0.02
    param_tol_dist = 1e-6
    param_obs_delta = 0.01
    param_human_delta = 0.3
    param_tray_delta = 0.01
    param_auto_delta = 0.001
else:
    param_h_dist = 1e-9
    param_eps_dist = 1e-9
    param_tol_dist = 1e-5
    param_obs_delta = 0.05
    param_human_delta = 0.4
    param_tray_delta = 0.05
    param_auto_delta = 0.015
    

#Parameters of the human movement
param_human_T = 15
param_spd_human_1 = 0.3 
param_spd_human_2 = 0.2 

#Parameters simulation
dt=0.02 
param_t_max = 250



##########################################################################


#Definition of the control functions

def fun_G(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
      

    return out


#The control function
def control_fun(_q, _robot, _htm_tg, _obstacles, _humans, _tray, _vel_human, _holding_tray, _not_delivering_tray):
    

    #Compute the "base task function"
    r0, jac_r0 = _robot.task_function(q = _q, htm_tg = _htm_tg)
    

    
    #Compute the new task function from the base one 
    r = np.matrix(r0)
    jac_r = np.matrix(jac_r0)
    rp = np.linalg.norm(r[0:3,:])**2
    r[3:6,:] += param_kr*(rp)
    jac_rp = 2*r[0:3,:].T*jac_r[0:3,:]
    jac_r[3,:] += param_kr*jac_rp
    jac_r[4,:] += param_kr*jac_rp
    jac_r[5,:] += param_kr*jac_rp
    
    
    #Assembly the H, f matrices
    no_joint = np.shape(_robot.q)[0]
    
    H = jac_r.T * jac_r + param_eps * np.identity(no_joint)
    f = -jac_r.T * fun_G(r, param_k)
 
    #Compute the end-effector Jacobian, this will be used in 
    #some places
    jac, htm = _robot.jac_geo(_q)
    s_e = htm[0:3,-1]
    jac_v = jac[0:3,:]
    jac_w = jac[3:6,:]   
       
    # Start assembling the matrices A and b
    #The vector b will be written as b_basic+b_ff
    #b_ff will include the feedforward terms
    #this is because we may have to change b_ff many times
    A = np.matrix(np.zeros((0,no_joint)))
    b_basic = np.matrix(np.zeros((0,1)))
    

    
    #### CONSTRAINT 1: JOINT LIMIT ####

    #Create the CBF constraint for (manipulator) joint limits
    I_ext = np.hstack( (np.zeros((no_joint-3,3)), np.identity(no_joint-3)) )
     
    A = np.vstack((A, I_ext  ))
    b_basic = np.vstack((b_basic, -param_eta_static*(_q[3:]-_robot.joint_limit[3:,0]-param_joint_delta) ))  
    A = np.vstack((A, -I_ext))
    b_basic = np.vstack((b_basic, -param_eta_static*(_robot.joint_limit[3:,1]-_q[3:]-param_joint_delta) )) 

    #### CONSTRAINT 2: VELOCITY LIMIT ####
    
    #Implement velocity limits. We will only implement velocity kimits
    #for the joint velocities
    A = np.vstack((A, I_ext))
    b_basic = np.vstack((b_basic, -param_max_dotq*np.matrix(np.ones((no_joint-3,1))) ))  
    A = np.vstack((A, -I_ext))
    b_basic = np.vstack((b_basic, -param_max_dotq*np.matrix(np.ones((no_joint-3,1))) ))   
        
    
    #### CONSTRAINT 3: TRAY PARALLEL TO THE GROUND ####
    
    #If the robot is holding the tray, the constraint for the orientation
    #of the axis should be a hard constraint, that is
    # (d/dt) r_{ox}(q) = -K_{ox}*ox should hold true
               
    if _holding_tray:
        y_d = _htm_tg[0:3,1]
        z_d = _htm_tg[0:3,2]
        x_e = htm[0:3,0]
        
        r_rotx = np.vstack((y_d.T*x_e,z_d.T*x_e))
        b_rotx = -param_kox*r_rotx
        jac_r_rotx = -np.vstack(( (y_d.T*ub.Utils.S(x_e))*jac[3:6,:], (z_d.T*ub.Utils.S(x_e))*jac[3:6,:]))
        
        #Implement an equality constraint
        
        A_eq = jac_r_rotx
        b_eq = np.matrix(b_rotx)
    else:
        A_eq = None
        b_eq = None
    
            
        
    #### CONSTRAINT 4: OBSTACLE AVOIDANCE WITH STATIC OBSTACLES ####
    
    #Implement obstacle avoidance with static obstacles
    tray_col_obj = _tray.list_of_objects[0]
    
     
    for obs in _obstacles:
        ds = _robot.compute_dist(q = _q, obj = obs, h=param_h_dist, eps=param_eps_dist, 
                                tol=param_tol_dist, no_iter_max=param_no_iter_max_dist, 
                                max_dist=param_max_dist_static)
        
        #If we are not delivering the tray in the table
        #we have to consider the full collision. If not, we disregard collision
        #with the last five collision objects in the kinematic chain
        
        if _not_delivering_tray:
            A = np.vstack((A, ds.jac_dist_mat))
            b_basic = np.vstack((b_basic, -param_eta_static*(ds.dist_vect-param_obs_delta)))
        else:
            A = np.vstack((A, ds.jac_dist_mat[0:-5,:]))
            b_basic = np.vstack((b_basic, -param_eta_static*(ds.dist_vect[0:-5,:]-param_obs_delta)))  
           
        #If we are holding the tray, we also have to check this collision  
        if _holding_tray and _not_delivering_tray:
            for obs in _obstacles:
                #Only make computation if the AABB dist is less than the maximum distance
                #to consider
                
                if ub.Utils.compute_aabbdist(obs, tray_col_obj) < param_max_dist_static:
                    p_tray, p_obs, dist, _ = tray_col_obj.compute_dist(obs, h=param_h_dist, 
                                    eps=param_eps_dist, tol=param_tol_dist,no_iter_max=param_no_iter_max_dist)

                    jac_dist = ((p_tray - p_obs).T * jac_v + np.cross((p_tray - s_e ).T, (p_tray - p_obs).T)  * jac_w)/dist
                    A = np.vstack((A, jac_dist))
                    b_basic = np.vstack((b_basic, -param_eta_static*(dist-param_obs_delta)))

    
    
    #### CONSTRAINT 5: COLLISION AVOIDANCE BETWEEN THE TRAY AND THE MANIPULATOR ####
    
    #Implement the collision avoidance between the tray and the body of the robot
    #Disregard the last link (because this one is colliding with the tray)
    
    if _holding_tray and _not_delivering_tray:
        ds = _robot.compute_dist(q = _q, obj = tray_col_obj, h=param_h_dist, eps=param_eps_dist, 
                                    tol=param_tol_dist, no_iter_max=param_no_iter_max_dist, 
                                    max_dist=param_max_dist_static)
        

        for ind1 in range(len(_robot.links)-3):
            for ind2 in range(len(_robot.links[ind1].col_objects)):
                try:
                    #We need to put the "try" because some of the indexes may
                    #not have been computed
                    
                    item = ds.get_item(ind1,ind2)
                    p_tray = item.point_object
                    p_robot = item.point_link
                    
                    jac_dist_1 = item.jac_distance
                    jac_dist_2 = ((p_tray - p_robot).T * jac_v + np.cross((p_tray - s_e ).T, (p_tray - p_robot).T)  * jac_w)/item.distance

                    A = np.vstack((A, jac_dist_1+jac_dist_2))
                    b_basic = np.vstack((b_basic, -param_eta_tray*(item.distance-param_tray_delta)))
                    
                except:
                    pass            
            
 
    #### CONSTRAINT 6: COLLISION AVOIDANCE BETWEEN THE MANIPULATOR AND ITSELF ####
    
    ds = robot.compute_dist_auto(q=_q, h=param_h_dist, eps=param_eps_dist, 
                                    tol=param_tol_dist, no_iter_max=param_no_iter_max_dist)
    
    
    A = np.vstack((A, ds.jac_dist_mat))
    b_basic = np.vstack((b_basic, -param_eta_auto*(ds.dist_vect-param_auto_delta)))
                    
    #### CONSTRAINT 7: COLLISION AVOIDANCE WITH MOVING OBSTACLES (HUMANS) ####
        
    #Implement collision avoidance with moving obstacles (humans)
    
    #Initialize b_ff
    b_ff = np.matrix(np.zeros(np.shape(b_basic)))
    
    for j in range(len(_humans)):
        
        human_col_obj = _humans[j].list_of_objects[-1]
        
        ds = _robot.compute_dist(q = _q, obj = human_col_obj, h=param_h_dist, 
                                 eps=param_eps_dist, tol=param_tol_dist, no_iter_max=param_no_iter_max_dist, 
                                 max_dist=param_max_dist_human)
        
        for ind1 in range(len(_robot.links)):
            for ind2 in range(len(_robot.links[ind1].col_objects)):
                try:
                    #We need to put the "try" because some of the indexes may
                    #not have been computed
                    
                    item = ds.get_item(ind1,ind2)
                    p_human = item.point_object
                    p_robot = item.point_link
                    dv = (p_human-p_robot)/item.distance
                    ff = dv.T *_vel_human[j]
                    
                    A = np.vstack((A, item.jac_distance))
                    b_basic = np.vstack((b_basic, -param_eta_human*(item.distance-param_human_delta)))
                    b_ff = np.vstack((b_ff, -ff))
                except:
                    pass

        
        #If we are holding the tray, we also have to check this collision
        if _holding_tray and _not_delivering_tray:
            
            #Only make computation if the AABB dist is less than the maximum distance
            #to consider
            
            if ub.Utils.compute_aabbdist(tray_col_obj, human_col_obj) < param_max_dist_static:
                            
                p_tray, p_human, dist, _ = tray_col_obj.compute_dist(human_col_obj, 
                            h=param_h_dist, eps=param_eps_dist, tol=param_tol_dist, 
                            no_iter_max=param_no_iter_max_dist)
                
                
                jac_dist = ((p_tray - p_human).T * jac_v + np.cross((p_tray - s_e).T, (p_tray - p_human).T)  * jac_w)/dist
                
                
                dv = (p_human-p_tray)/dist
                ff = dv.T *_vel_human[j]
                
                A = np.vstack((A, jac_dist))
                b_basic = np.vstack((b_basic, -param_eta_human*(dist-param_human_delta)))
                b_ff = np.vstack((b_ff, -ff))        
            
        
    ##### FINALLY TRY TO SOLVE #############
    #Try to compute the controller
    #If it fails, reduce the feedforward term
    alpha = 1.0
    
    while cont:
    
        try:
            u = ub.Utils.solve_qp(H, f, A, b_basic+alpha*b_ff, A_eq, b_eq)
            return u, np.linalg.norm(r)
        except:
            alpha = param_decay_alpha*alpha
            
            if alpha < param_min_alpha:
                return np.matrix(0*_robot.q), np.linalg.norm(r)


    
#####################################################

#Model the task

htm_tg = [htm_tg_table_0, htm_tg_table_1, htm_tg_table_0, htm_tg_table_2, htm_tg_table_3, htm_tg_table_2, 
          htm_tg_table_4, htm_tg_table_5, htm_tg_table_4, htm_tg_table_6, htm_tg_table_7,htm_tg_table_0]
holding_tray = [False, False, True, True, True, False, False , False, True, True, True, False]
not_delivering_tray = [True,False,False,True, False, False, True, False, False, True, False, True]


#Initialize some variables
q = np.matrix(robot.q)
q[8]+=np.pi/2

mode = 0
cont = True
y_h1 = 0.5
x_h2 = -0.5
t = 0

vel_human = [np.matrix([0.,0.,0.]).T, np.matrix([0.,0.,0.]).T]
obj_to_hold = tray_1

hist_u = []
hist_error = []
hist_mode = []
hist_t = []
hist_q = []


    
while cont:
    
    
    #Implement human movement
    if t % param_human_T < param_human_T/4 or t % param_human_T > 3*param_human_T/4:
        y_h1 += param_spd_human_1*dt
        x_h2 += param_spd_human_2*dt
        rotz_h1 = ub.Utils.rotz(0)
        rotz_h2 = ub.Utils.rotz(-np.pi/2)
        
        vel_human[0] = np.matrix([0.,param_spd_human_1, 0.]).T
        vel_human[1] = np.matrix([param_spd_human_2,0., 0.]).T
    else:
        y_h1 -= param_spd_human_1*dt
        x_h2 -= param_spd_human_2*dt
        rotz_h1 = ub.Utils.rotz(-np.pi)
        rotz_h2 = ub.Utils.rotz(np.pi/2)

        vel_human[0] = np.matrix([0.,-param_spd_human_1, 0.]).T
        vel_human[1] = np.matrix([-param_spd_human_2,0., 0.]).T
                
    set_human_pose(t,human_john_connor, ub.Utils.trn([.01,y_h1,0])*rotz_h1,t)
    set_human_pose(t,human_kyle_reese, ub.Utils.trn([x_h2,0,0])*rotz_h2,t)
    
    
    ###    
    
    
        
    dotq, error = control_fun(q, robot, htm_tg[mode], all_obstacles, all_humans,  
                              obj_to_hold, vel_human, holding_tray[mode], not_delivering_tray[mode])
    

    
    hist_u.append(dotq)
    hist_error.append(error)
    hist_mode.append(mode)
    hist_t.append(t)
    hist_q.append(np.matrix(q))

    
    t += dt
    q+=dotq*dt

    
    robot.add_ani_frame(time = t, q = q)
    robot.update_col_object(time=t)

    
    print("Mode "+str(mode)+" = "+str(round(error,3))+", t = "+str(round(t,2)))
    if error <= param_tol_error:
        
        try:
            mode+= 1
            
            if mode>0 and not holding_tray[mode-1] and holding_tray[mode]:
                robot.attach_object(obj_to_hold)
                
            if mode>0 and holding_tray[mode-1] and not holding_tray[mode]:
                robot.detach_object(obj_to_hold)
                obj_to_hold = tray_2
        except:
            cont = False    
         

    
    cont = cont and t < param_t_max 
 

 
###############################################################
# Plot graphs
        
# Plot joint positions
fig = plt.figure(facecolor='#191919') 
for i in range(10):
    ax = plt.subplot(5, 2, i + 1, facecolor='#191919')
    
    ax.plot(hist_t, [q[i, 0] for q in hist_q], color='#81d41a', zorder=10)
    
    if i>=3:
        ax.plot(hist_t, [robot.joint_limit[i, 0] + param_joint_delta for _ in hist_q], color='red')
        ax.plot(hist_t, [robot.joint_limit[i, 1] - param_joint_delta for _ in hist_q], color='red')
    
    if i==0:
        ax.set_title("x", color='white')
    if i==1:
        ax.set_title("y", color='white')
    if i==2:
        ax.set_title("theta", color='white')
    if i>=3:    
        ax.set_title("Manip joint " + str(i -2), color='white')
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.grid(True, color='white', linestyle=':', alpha=0.3)

plt.tight_layout()
fig.subplots_adjust(hspace=0.4)

# Plot joint velocities
fig = plt.figure(facecolor='#191919')   
for i in range(10):
    ax = plt.subplot(5, 2, i + 1, facecolor='#191919')
    
    ax.plot(hist_t, [u[i, 0] for u in hist_u], color='#81d41a', zorder=10)
    
    if i>=3:
        ax.plot(hist_t, [-param_max_dotq for _ in hist_u], color='red')
        ax.plot(hist_t, [param_max_dotq for _ in hist_u], color='red')
    
    
    if i==0:
        ax.set_title("velocity x", color='white')
    if i==1:
        ax.set_title("velocity y", color='white')
    if i==2:
        ax.set_title("velocity theta", color='white')
    if i>=3:    
        ax.set_title("velocity joint " + str(i -2), color='white')
    
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.grid(True, color='white', linestyle=':', alpha=0.3)

plt.tight_layout()
fig.subplots_adjust(hspace=0.4)

plt.show()

   

sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, pixel_ratio=0.9)
sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/","part_3_4_b")
# %%
