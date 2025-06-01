import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt
from presentation.code.part3.helper_anim_part_3_ex2 import *

    
###################################   
#Parameters control
param_eta_static = 0.2
param_eta_human = 0.3 
param_eps = 0.005
param_k = 0.3
param_kr = 1.0
param_kox = 0.3
param_max_qdot = 1.5
param_obs_delta = 0.01
param_joint_delta = 2*np.pi/180
param_human_delta = 0.3

#Parameters distance computation
param_h_dist = 0.05
param_eps_dist = 0.02
param_tol_dist = 1e-5
param_no_iter_max_dist = 2000
param_max_dist_static = 0.6
param_max_dist_human = 2.5

#Parameters of the human movement
param_human_T = 15
param_spd_human_1 = 0.3 
param_spd_human_2 = 0.2 

#Parameters simulation
dt=0.02 
param_t_max = 200

def fun_G(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
      

    return out


def control_fun(_q, _robot, _htm_tg, _obstacles, _humans, _tray, _vel_human, _holding_tray, _not_delivering_tray, _alpha=1.0):
    

    #Compute the "base task function"
    r0, jac_r0 = _robot.task_function(_q, _htm_tg)
    
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
    no_joint = np.shape(robot.q)[0]
    
    H = jac_r.T * jac_r + param_eps * np.identity(no_joint)
    f = -jac_r.T * fun_G(r, param_k)
 
    #Compute the end-effector Jacobian, this will be used in 
    #some places
    jac, htm = _robot.jac_geo(_q)
    s_e = htm[0:3,-1]
    jac_v = jac[0:3,:]
    jac_w = jac[3:6,:]   
       
    # Start assembling the matrices A and b

    A = np.matrix(np.zeros((0,no_joint)))
    b = np.matrix(np.zeros((0,1)))

    #Create the CBF constraint for (manipulator) joint limits
    I_ext = np.hstack( (np.zeros((no_joint-3,3)), np.identity(no_joint-3)) )
     
    A = np.vstack((A, I_ext  ))
    b = np.vstack((b, -param_eta_static*(_q[3:]-robot.joint_limit[3:,0]-param_joint_delta) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_eta_static*(robot.joint_limit[3:,1]-_q[3:]-param_joint_delta) )) 

    #Implement velocity limits. We will only implement velocity kimits
    #for the joint velocities
    
    A = np.vstack((A, I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint-3,1))) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint-3,1))) ))   
        
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
        
        #We implement the constraint A_eq *u = b_eq as
        #A_eq * u <= b_eq, -Ae_q * u >= -b_eq (with a margin)
        A = np.vstack((A, jac_r_rotx, -jac_r_rotx))
        b = np.vstack((b, b_rotx-0.001, -b_rotx-0.001))
            
        
    #Implement obstacle avoidance with static obstacles
    tray_col_obj = _tray.list_of_objects[0]
    
     
    for obs in _obstacles:
        ds = robot.compute_dist(q = _q, obj = obs, h=param_h_dist, eps=param_eps_dist, 
                                tol=param_tol_dist, no_iter_max=param_no_iter_max_dist, 
                                max_dist=param_max_dist_static)
        
        #If we are not delivering the tray in the table
        #we have to consider the full collision. If not, we disregard collision
        #with the last five collision objects in the kinematic chain
        
        if _not_delivering_tray:
            A = np.vstack((A, ds.jac_dist_mat))
            b = np.vstack((b, -param_eta_static*(ds.dist_vect-param_obs_delta)))
        else:
            A = np.vstack((A, ds.jac_dist_mat[0:-5,:]))
            b = np.vstack((b, -param_eta_static*(ds.dist_vect[0:-5,:]-param_obs_delta)))  
           
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
                    b = np.vstack((b, -param_eta_static*(dist-param_obs_delta)))

            
    #Implement obstacle avoidance with moving obstacles (humans)
    
    b_ff = np.matrix(np.zeros(np.shape(b)))
    
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
                    b = np.vstack((b, -param_eta_human*(item.distance-param_human_delta)))
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
                b = np.vstack((b, -param_eta_human*(dist-param_human_delta)))
                b_ff = np.vstack((b_ff, -ff))        
            
        

    #Try to compute the controller
    _alpha = 1.0
    
    while cont:
    
        try:
            u = ub.Utils.solve_qp(H, f, A, b+_alpha*b_ff)
            return u, np.linalg.norm(r)
        except:
            print("Failed... try again")
            _alpha = 0.9*_alpha


    
    

 
 
 
 
q = np.matrix(robot.q)

mode = 0

htm_tg = [htm_tg_table_0, htm_tg_table_1, htm_tg_table_0, htm_tg_table_2, htm_tg_table_3, htm_tg_table_2, htm_tg_table_4, htm_tg_table_5, htm_tg_table_4, htm_tg_table_6, htm_tg_table_7,htm_tg_table_0]
holding_tray = [False, False, True, True, True, False, False , False, True, True, True, False]
consider_collision = [True,False,False,True, False, False, True, False, False, True, False, True]

obj_to_hold = tray_1
cont = True
i = 0


y_h1 = 0.5
x_h2 = -0.5

t = 0

hist_u = []

dotq = np.matrix(0*robot.q)

hist_dp = []
hist_ds = []
hist_error = []
vel_human = [np.matrix([0.,0.,0.]).T, np.matrix([0.,0.,0.]).T]


while cont:
    
    
    #Human movements
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
                              obj_to_hold, vel_human, holding_tray[mode], consider_collision[mode])
    

    
    hist_u.append(dotq)
    hist_error.append(error)
    
    t += dt
    
    q+=dotq*dt

    
    robot.add_ani_frame(time = t, q = q)
    robot.update_col_object(time=t)

    
    print("Mode "+str(mode)+" = "+str(round(error,3))+", "+str(i))
    if error <= 0.007:
        
        try:
            mode+= 1
            
            if mode>0 and not holding_tray[mode-1] and holding_tray[mode]:
                robot.attach_object(obj_to_hold)
                
            if mode>0 and holding_tray[mode-1] and not holding_tray[mode]:
                robot.detach_object(obj_to_hold)
                obj_to_hold = tray_2
        except:
            cont = False    
         

    
    cont = cont and t < param_t_max #6000
    
plt.figure()    
for i in range(0,3):
    plt.plot([j for j in range(len(hist_u))], [u[i,0] for u in hist_u])    


plt.figure()    
plt.plot([j for j in range(len(hist_error))], hist_error)  
    
        
plt.show()

sim.save()