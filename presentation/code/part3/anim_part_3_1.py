import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

robot = ub.Robot.create_kuka_lbr_iiwa(htm=ub.Utils.rotz(np.pi/2))



#Create objects

texture_steel = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)

platform1 = ub.Box(htm=ub.Utils.trn([0.3,0,0.25]),width=0.2,depth=3.0,height=0.5,mesh_material=material_steel)
platform2 = ub.Box(htm=ub.Utils.trn([-0.8+0.1,0,0.25]),width=0.8,depth=1.0,height=0.5,mesh_material=material_steel)
platform3 = ub.Box(htm=ub.Utils.trn([-1.2+0.3+0.1,-0.5+0.15,0.7]),width=0.6,depth=0.3,height=0.4,mesh_material=material_steel)
platform4 = ub.Box(htm=ub.Utils.trn([-1.2+0.3+0.1,0.5-0.15,0.7]),width=0.6,depth=0.3,height=0.4,mesh_material=material_steel)
platform5 = ub.Box(htm=ub.Utils.trn([-0.8+0.1,0,0.9+0.1]),width=0.8,depth=1.0,height=0.2,mesh_material=material_steel)
platform6 = ub.Box(htm=ub.Utils.trn([-0.5+0.1,0,0.53]),width=0.1,depth=0.4,height=0.06,mesh_material=material_steel)

lever = ub.Cylinder(htm=ub.Utils.trn([-1.2+0.3+0.1+0.3-0.1,0.5-0.15,0.77])*ub.Utils.roty(np.pi/2), radius=0.02, height =0.3, color='magenta')


disk = ub.Cylinder(htm=ub.Utils.trn([0.3,0,0.5+0.02]),height=0.04,radius=0.05,color='yellow')

sim = ub.Simulation.create_sim_factory([robot, platform1, platform2, platform3, platform4, platform5, platform6, lever, disk])

all_obstacles = [platform1, platform2, platform3, platform4, platform5, platform6]


#######################

def fun_F(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
       
    # out[3:6,0] = 0.2*out[3:6,0] 
    return out

def compute_control(_q, _htm_tg):
    
    param_eta = 0.4*3
    param_obs_delta = 0.025 #0.05
    param_joint_delta = 2*np.pi/180
    param_eps = 0.001
    param_k = 0.5*2
    param_max_qdot = 1.5
    
    A = np.matrix(np.zeros((0,7)))
    b = np.matrix(np.zeros((0,1)))
    
    #Create the CBF constraints for all obstacles:
    for obs in all_obstacles:
        ds = robot.compute_dist(q = _q, obj = obs)
        A = np.vstack((A, ds.jac_dist_mat))
        b = np.vstack((b, -param_eta*(ds.dist_vect-param_obs_delta)))
        
    A_save = np.matrix(A)
    b_save = np.matrix(b)
    
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
    if np.linalg.norm(r[0:3,:])>=0.3:
        r = r[0:3,:]
        jac_r = jac_r[0:3,:]
    
    H = 2*(jac_r.T * jac_r + param_eps* np.identity(7))
    f = -2*jac_r.T*fun_F(r, param_k)
    
    #Compute the control input
    u = ub.Utils.solve_qp(H, f, A, b)
    
    # print("---------------")
    # print((_q-robot.joint_limit[:,0]).T)
    # print((robot.joint_limit[:,1]-_q).T)
    
    # uc = ub.Utils.solve_qp(H, f, A_save, b_save)
    
    # print("lim_d = "+str( (uc +param_eta*(_q-robot.joint_limit[:,0]-param_joint_delta)).T ) )
    # print("lim_u = "+str( (uc +param_eta*(robot.joint_limit[:,1]-_q-param_joint_delta)).T ) )
    # print("u = "+str(u.T))
    # print("uc = "+str(uc.T))
    
    return u, r
        
    

#######################

dt=0.005

htm_tg_0 = ub.Utils.trn([0.3,0,0.5+0.04])*ub.Utils.roty(np.pi)
htm_tg_1 = ub.Utils.trn([-0.65,0,0.57])*ub.Utils.roty(np.pi)
htm_tg_2 = ub.Utils.trn([-1.2+0.3+0.1+0.3+0.3-0.15,0.5-0.15,0.77])*ub.Utils.roty(-np.pi/2)
htm_tg_3 = ub.Utils.trn([0.3,0.2,0.5+0.03])*ub.Utils.roty(np.pi)

sim.add(ub.Frame(htm_tg_0, size=0.1))
sim.add(ub.Frame(htm_tg_1, size=0.1))
sim.add(ub.Frame(htm_tg_2, size=0.1))
sim.add(ub.Frame(htm_tg_3, size=0.1))

q = np.matrix(robot.q)

mode = 0
cont = True
i=0

hist_q = []
hist_u = []

while cont:
    
    i+=1
    
    cont = i < 10000
    
    
    
    if mode==0:
        u, r = compute_control(q, htm_tg_0)
        
        print("Mode 0, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 1
            robot.attach_object(disk)

    if mode==1:
        u, r = compute_control(q, htm_tg_1)
        
        print("Mode 1, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 2
            robot.detach_object(disk)

    if mode==2:
        u, r = compute_control(q, htm_tg_2)
        
        print("Mode 2, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.005:

            mode = 3

    if mode==3:
        u, r = compute_control(q, htm_tg_1)
        
        print("Mode 3, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            mode = 4
            robot.attach_object(disk)

    if mode==4:
        u, r = compute_control(q, htm_tg_3)
        
        print("Mode 4, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            cont = False
            robot.detach_object(disk)
     
     
    hist_q.append(np.matrix(q))
    hist_u.append(np.matrix(u))
                                       
    q = robot.q+u*dt
    robot.add_ani_frame(time = i*dt, q = q)
    
    
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