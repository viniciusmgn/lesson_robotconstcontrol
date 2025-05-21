import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt
import cvxpy as cp

def solve_qc_qp(H: np.matrix, f: np.matrix, A: np.matrix, b: np.matrix, gamma: float, u_0: np.matrix = None):
    n = H.shape[0]
    m = A.shape[0]
    
    u = cp.Variable((n, 1))
    
    objective = 0.5 * cp.quad_form(u, H) + f.T @ u
    
    constraints = []
    for i in range(m):
        ai = A[i, :].reshape((1, n))
        bi = b[i, 0]
        constraint = ai @ u - 0.5 * gamma * cp.sum_squares(u) >= bi
        constraints.append(constraint)
    
    prob = cp.Problem(cp.Minimize(objective), constraints)

    # Warm start if u_0 is provided
    if u_0 is not None:
        u.value = np.asarray(u_0).reshape((n, 1))  # Convert np.matrix to (n,1) array

    prob.solve(solver=cp.ECOS, warm_start=True)  # Warm start enabled
    
    if u.value is None:
        raise ValueError("Problem is infeasible or solver failed.")
    
    return np.matrix(u.value)

#Parameters
param_eta = 0.4*4
param_eps = 0.01
param_k = 0.5*2
param_max_qdot = 1.5

param_h = 0.01
param_eps_obs = 0.02

param_obs_delta = 0.002 #0.05
param_joint_delta = 2*np.pi/180


#####################
    
robot = ub.Robot.create_kuka_lbr_iiwa(htm=ub.Utils.rotz(np.pi/2))



#Create objects

texture_steel = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)

platform1 = ub.Box(htm=ub.Utils.trn([0.3,0,0.45]),width=0.2,depth=3.0,height=0.1,mesh_material=material_steel)
platform2 = ub.Box(htm=ub.Utils.trn([-0.8+0.1,0,0.25]),width=0.8,depth=1.0,height=0.5,mesh_material=material_steel)
platform3 = ub.Box(htm=ub.Utils.trn([-1.2+0.3+0.1,-0.5+0.15,0.7]),width=0.6,depth=0.3,height=0.4,mesh_material=material_steel)
platform4 = ub.Box(htm=ub.Utils.trn([-1.2+0.3+0.1,0.5-0.15,0.7]),width=0.6,depth=0.3,height=0.4,mesh_material=material_steel)
platform5 = ub.Box(htm=ub.Utils.trn([-0.8+0.1,0,0.9+0.15]),width=0.8,depth=1.0,height=0.21,mesh_material=material_steel)
platform6 = ub.Box(htm=ub.Utils.trn([-0.5+0.1,0,0.55]),width=0.1,depth=0.4,height=0.1,mesh_material=material_steel)
platform7 = ub.Box(htm=ub.Utils.trn([-0.48,0.225,0.7]),width=0.1,depth=0.05,height=0.2,mesh_material=material_steel)



lever = ub.Cylinder(htm=ub.Utils.trn([-1.2+0.3+0.1+0.3-0.1,0.5-0.15,0.77])*ub.Utils.roty(np.pi/2), radius=0.02, height =0.3, color='magenta')


disk = ub.Cylinder(htm=ub.Utils.trn([0.3,0,0.5+0.02]),height=0.04,radius=0.05,color='yellow')

sim = ub.Simulation.create_sim_factory([robot, platform1, platform2, platform3, platform4, platform5, platform6, platform7, lever, disk])

all_obstacles = [platform2, platform3, platform4, platform5, platform6, platform7]


#######################

def fun_F(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
       
    # out[3:6,0] = 0.2*out[3:6,0] 
    return out

def compute_F_and_grad(_b, _A, _r):
    b_flat = np.asarray(_b).flatten()  
    b_min = np.min(b_flat)
    
    b_pow_inv_h = (b_flat/b_min) ** (-1.0 / _r)             
    S = np.sum(b_pow_inv_h)                        
    F = b_min * (S ** (-_r))
                               
    
    b_pow_inv_h_plus1 = (b_flat/b_min) ** (-1.0 / _r - 1.0)  

    weight = b_pow_inv_h_plus1/( (F/b_min)**(-1.0/_r-1.0) ) 
    grad_F = weight * _A            
    
    return np.matrix(F), grad_F    

def compute_control(_q, _htm_tg, is_handling_disk, care_obstacles):
    

    
    A = np.matrix(np.zeros((0,7)))
    b = np.matrix(np.zeros((0,1)))
    
    #Create the CBF constraints for all obstacles:
    

    

        
    
    #If the disk is in the end-effector, it should also be handled
    b_raw = np.matrix([0])
    grad_norm = 1.0
    if is_handling_disk and care_obstacles:

        #Call the forward kinematic and differential for the end-effector,
        #since this is where the disk is attaches
        jac, htm = robot.jac_geo(q=_q)
        
        s_e = htm[0:3,-1]
        jac_v = jac[0:3,:]
        jac_w = jac[3:6,:]
        
        print("---------")
        
        for obs in all_obstacles:
            point_disk, point_obs, dist, _ = disk.compute_dist(obs, h=param_h, eps=param_eps_obs, no_iter_max=1000, tol=1e-6)


            jac_dist = (point_disk - point_obs).T * jac_v + np.cross((point_disk - s_e ).T, (point_disk - point_obs).T)  * jac_w
            A = np.vstack((A, jac_dist))
            b = np.vstack((b, dist))
        

    if care_obstacles:
        for obs in all_obstacles:
            ds = robot.compute_dist(q = _q, obj = obs,  h=param_h, eps=param_eps_obs,  no_iter_max=1000, tol=1e-6)
            A = np.vstack((A, ds.jac_dist_mat))
            b = np.vstack((b, ds.dist_vect))
            
                 
                    
        b_raw, A = compute_F_and_grad(b, A, 0.2)
        
        b = -param_eta*(b_raw-param_obs_delta)
        
        grad_norm = np.linalg.norm(A)
        
        print("F = "+str(b_raw))
        print("grad_F = "+str(A))
            
    #Create the CBF constraint for joint limits
    A = np.vstack((A, np.identity(7)))
    b = np.vstack((b, -param_eta*(_q-robot.joint_limit[:,0]-param_joint_delta) ))  
    A = np.vstack((A, -np.identity(7)))
    b = np.vstack((b, -param_eta*(robot.joint_limit[:,1]-_q-param_joint_delta) ))  
    
        
    
    #Implement velocity limits
    A = np.vstack((A, np.identity(7)))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((7,1))) ))  
    A = np.vstack((A, -np.identity(7)))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((7,1))) ))      
    
    
    #Create the objective function
    r, jac_r = robot.task_function(q=_q, htm_tg=_htm_tg)
    
    #Heuristic
    if np.linalg.norm(r[0:3,:])>=0.1:
        r = r[0:3,:]
        jac_r = jac_r[0:3,:]
    
    H = 2*(jac_r.T * jac_r + param_eps* np.identity(7))
    f = -2*jac_r.T*fun_F(r, param_k)
    
    #Compute the control input
    
    try:
        u = ub.Utils.solve_qp(H, f, A, b)
    except:
        u = 0*np.matrix(_q)
    # except:
    #     print("Error!")
    #     u = 0*np.matrix(_q)

    
    # print("---------------")
    # print((_q-robot.joint_limit[:,0]).T)
    # print((robot.joint_limit[:,1]-_q).T)
    
    # uc = ub.Utils.solve_qp(H, f, A_save, b_save)
    
    # print("lim_d = "+str( (uc +param_eta*(_q-robot.joint_limit[:,0]-param_joint_delta)).T ) )
    # print("lim_u = "+str( (uc +param_eta*(robot.joint_limit[:,1]-_q-param_joint_delta)).T ) )
    # print("u = "+str(u.T))
    # print("uc = "+str(uc.T))
    
    return u, r, b_raw[0,0]
        
    

#######################

dt=0.005


# htm_tg_0 = ub.Utils.trn([0.3,0,0.5+0.04+0.05])*ub.Utils.roty(np.pi)
# htm_tg_1 = np.matrix([0.3,0,0.5+0.04]).T*ub.Utils.roty(np.pi)
# htm_tg_2 = np.matrix([0.3,0,0.5+0.04+0.05]).T*ub.Utils.roty(np.pi)


# htm_tg_3 = ub.Utils.trn([-0.65,0,0.57+0.03])*ub.Utils.roty(np.pi)
# htm_tg_4 = np.matrix([-0.65,0,0.57]).T*ub.Utils.roty(np.pi)
# htm_tg_5 = np.matrix([-0.65,0,0.57+0.03]).T*ub.Utils.roty(np.pi)

# htm_tg_6 = ub.Utils.trn([-1.2+0.3+0.1+0.3+0.3-0.15,0.5-0.15,0.77])*ub.Utils.roty(-np.pi/2)
# htm_tg_3 = ub.Utils.trn([0.3,0.2,0.5+0.03])*ub.Utils.roty(np.pi)


htm_tg_0 = ub.Utils.trn([0.3,0,0.5+0.04+0.1])*ub.Utils.roty(np.pi)
htm_tg_1 = ub.Utils.trn([0.3,0,0.5+0.04])*ub.Utils.roty(np.pi)
htm_tg_2 = ub.Utils.trn([0.3,0,0.5+0.04+0.1])*ub.Utils.roty(np.pi)


htm_tg_3 = ub.Utils.trn([-0.65,0,0.57+0.06])*ub.Utils.roty(np.pi)
htm_tg_4 = ub.Utils.trn([-0.65,0,0.57])*ub.Utils.roty(np.pi)
htm_tg_5 = ub.Utils.trn([-0.65,0,0.57+0.06])*ub.Utils.roty(np.pi)

htm_tg_6 = ub.Utils.trn([-1.2+0.3+0.1+0.3+0.3-0.15,0.5-0.15,0.77])*ub.Utils.roty(-np.pi/2)

htm_tg_7 = ub.Utils.trn([0.3,0.2,0.5+0.04+0.1])*ub.Utils.roty(np.pi)

sim.add(ub.Frame(htm_tg_0, size=0.1))
sim.add(ub.Frame(htm_tg_3, size=0.1))
sim.add(ub.Frame(htm_tg_6, size=0.1))
sim.add(ub.Frame(htm_tg_7, size=0.1))

q = np.matrix(robot.q)

mode = 0
cont = True
i=0

hist_q = []
hist_u = []
hist_b = []

t = 0
while cont:
    
    i+=1
    
    cont = i < 5000
    
    
    
    if mode==0:
        #Going slightly above the disk
        u, r, b_raw = compute_control(q, htm_tg_0, False, True)
        
        print("Mode 0, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 1

    if mode==1:
        #Downward movement
        u, r, b_raw = compute_control(q, htm_tg_1, False, False)
        
        print("Mode 1, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 2
            robot.attach_object(disk)

    if mode==2:
        #Upward movement
        u, r, b_raw = compute_control(q, htm_tg_2, False, False)
        
        print("Mode 2, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 3


                        
    if mode==3:
        #Move object to be delivered
        u, r, b_raw = compute_control(q, htm_tg_3, True, True)
    
        print("Mode 3, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 4
            

            



    if mode==4:
        #Move down
        u, r, b_raw = compute_control(q, htm_tg_4, True, False)
        
        print("Mode 4, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 5
            robot.detach_object(disk)
            
    if mode==5:
        #Move up
        u, r, b_raw = compute_control(q, htm_tg_5, False, False)
        
        print("Mode 5, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 6
            

                        
    if mode==6:
        #Go press the button
        u, r, b_raw = compute_control(q, htm_tg_6, False, True)
        
        print("Mode 6, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.005:
            mode = 7

    if mode==7:
        u, r, b_raw = compute_control(q, htm_tg_3, False, True)
        
        print("Mode 7, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 8
            

    if mode==8:
        u, r, b_raw = compute_control(q, htm_tg_4, False, False)
        
        print("Mode 8, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            mode = 9
            robot.attach_object(disk)
            
    if mode==9:
        u, r, b_raw = compute_control(q, htm_tg_5, False, False)
        
        print("Mode 9, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            mode = 10    
            

    if mode==10:
        u, r, b_raw = compute_control(q, htm_tg_7, True, True)
        
        print("Mode 10, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            cont = False 
                 
    hist_q.append(np.matrix(q))
    hist_u.append(np.matrix(u))
    hist_b.append(b_raw)
                                       
    q = robot.q+u*dt
    t+= dt
    
    # if b_raw < 0.005:
    #     dt=0.001
    # else:
    #     dt=0.005
        
        
    robot.add_ani_frame(time = t, q = q)
    
    
plt.figure()

plt.scatter([i for i in range(len(hist_b))],hist_b)


plt.figure() 
for i in range(7):
    
    plt.subplot(4,2,i+1)
    plt.plot([q[i,0] for q in hist_q])
    
    param_joint_delta = 5*np.pi/180
    plt.plot([robot.joint_limit[i,0]+param_joint_delta for j in range(len(hist_q))])
    plt.plot([robot.joint_limit[i,1]-param_joint_delta for j in range(len(hist_q))])

    # plt.figure()
    # plt.plot([u[i,0] for u in hist_q])
 
plt.figure()   
for i in range(7):
    
    plt.subplot(4,2,i+1)
    plt.plot([u[i,0] for u in hist_u])
    
        
plt.show()

sim.save()