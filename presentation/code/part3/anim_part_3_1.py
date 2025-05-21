import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

#Parameters
param_eta = 0.4*3
param_eps = 0.001
param_k = 0.5*2
param_max_qdot = 1.5
param_obs_delta = 0.025 
param_joint_delta = 2*np.pi/180
dt=0.005
param_iter_max = 10000
robot = ub.Robot.create_kuka_lbr_iiwa(htm=ub.Utils.rotz(np.pi/2))

htm_tg_0 = ub.Utils.trn([0.3, 0.0, 0.64])*ub.Utils.roty(np.pi)
htm_tg_1 = ub.Utils.trn([0.3, 0.0, 0.54])*ub.Utils.roty(np.pi)
htm_tg_2 = ub.Utils.trn([0.3, 0.0, 0.64])*ub.Utils.roty(np.pi)
htm_tg_3 = ub.Utils.trn([-0.65, 0.0, 0.63])*ub.Utils.roty(np.pi)
htm_tg_4 = ub.Utils.trn([-0.65, 0.0, 0.57])*ub.Utils.roty(np.pi)
htm_tg_5 = ub.Utils.trn([-0.65, 0.0, 0.63])*ub.Utils.roty(np.pi)
htm_tg_6 = ub.Utils.trn([-0.35, 0.35, 0.77])*ub.Utils.roty(-np.pi/2)
htm_tg_7 = ub.Utils.trn([0.30,  0.20, 0.64])*ub.Utils.roty(np.pi)




#####################
#Create environment

texture_steel = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

texture_gold = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/gold_metal.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])


material_steel= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_steel)
material_gold= ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], texture_map=texture_gold)

platform1 = ub.Box(htm=ub.Utils.trn([0.3, 0.0, 0.25]),width=0.2,depth=3.0,height=0.5,mesh_material=material_steel)
platform2 = ub.Box(htm=ub.Utils.trn([-0.7, 0.0, 0.25]),width=0.8,depth=1.0,height=0.5,mesh_material=material_steel)
platform3 = ub.Box(htm=ub.Utils.trn([-0.8, -0.35, 0.725]),width=0.6,depth=0.3,height=0.45,mesh_material=material_steel)
platform4 = ub.Box(htm=ub.Utils.trn([-0.8, 0.35, 0.725]),width=0.6,depth=0.3,height=0.45,mesh_material=material_steel)
platform5 = ub.Box(htm=ub.Utils.trn([-0.7, 0.0, 1.05]),width=0.8,depth=1.0,height=0.2,mesh_material=material_steel)
platform6 = ub.Box(htm=ub.Utils.trn([-0.4, 0.0, 0.55]),width=0.1,depth=0.4,height=0.1,mesh_material=material_steel)
platform7 = ub.Box(htm=ub.Utils.trn([-0.48,0.225,0.7]),width=0.1,depth=0.05,height=0.2,mesh_material=material_steel)
button = ub.Cylinder(htm=ub.Utils.trn([-0.6, 0.35, 0.77])*ub.Utils.roty(np.pi/2), radius=0.02, height =0.3, color='magenta')
disk = ub.Cylinder(htm=ub.Utils.trn([0.3,0,0.52]),height=0.04,radius=0.05,mesh_material=material_gold)



all_obstacles = [platform1, platform2, platform3, platform4, platform5, platform6, platform7]


if True:
    
    all_points = []
    for obs in all_obstacles:
        all_points+=[np.matrix(p).T for p in obs.to_point_cloud(disc=0.04).points.T]
        
    all_obstacles = [ub.PointCloud(points=all_points, color='cyan', size=0.02)]
    
sim = ub.Simulation.create_sim_factory([robot, button, disk])

sim.add(ub.Frame(htm_tg_0, size=0.1))
sim.add(ub.Frame(htm_tg_3, size=0.1))
sim.add(ub.Frame(htm_tg_6, size=0.1))
sim.add(ub.Frame(htm_tg_7, size=0.1))




for obs in all_obstacles:
    sim.add(obs)


#######################

def fun_F(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
       
    return out

def compute_control(_q, _htm_tg, is_handling_disk, care_obstacles):
    
    no_joint = np.shape(robot.q)[0]
    
    A = np.matrix(np.zeros((0,no_joint)))
    b = np.matrix(np.zeros((0,1)))
    
    #Create the CBF constraints for all obstacles:
    if care_obstacles:
        
        #Between the robot and the objects
        for obs in all_obstacles:
            ds = robot.compute_dist(q = _q, obj = obs)
            A = np.vstack((A, ds.jac_dist_mat))
            b = np.vstack((b, -param_eta*(ds.dist_vect-param_obs_delta)))
        
        #If the disk is in the end-effector, it should also be handled
        if is_handling_disk:

            #Call the forward kinematic and differential for the end-effector,
            #since this is where the disk is attaches
            jac, htm = robot.jac_geo(q=_q)
            
            s_e = htm[0:3,-1]
            jac_v = jac[0:3,:]
            jac_w = jac[3:6,:]
            
            for obs in all_obstacles:
                point_disk, point_obs, dist, _ = disk.compute_dist(obs)

                jac_dist = (point_disk - point_obs).T * jac_v + np.cross((point_disk - s_e ).T, (point_disk - point_obs).T)  * jac_w
                A = np.vstack((A, jac_dist))
                b = np.vstack((b, -param_eta*(dist-param_obs_delta)))
                    
    #Create the CBF constraint for joint limits
    A = np.vstack((A, np.identity(no_joint)))
    b = np.vstack((b, -param_eta*(_q-robot.joint_limit[:,0]-param_joint_delta) ))  
    A = np.vstack((A, -np.identity(no_joint)))
    b = np.vstack((b, -param_eta*(robot.joint_limit[:,1]-_q-param_joint_delta) ))  
    
    #Implement velocity limits
    A = np.vstack((A, np.identity(no_joint)))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint,1))) ))  
    A = np.vstack((A, -np.identity(no_joint)))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint,1))) ))      
    
    
    #Create the objective function
    r, jac_r = robot.task_function(q=_q, htm_tg=_htm_tg)
        
    H = 2*(jac_r.T * jac_r + param_eps* np.identity(no_joint))
    f = -2*jac_r.T*fun_F(r, param_k)
    
    #Compute the control input
    u = ub.Utils.solve_qp(H, f, A, b)

    return u, r
        
#######################




q = np.matrix(robot.q)

mode = 0
cont = True
i=0

hist_q = []
hist_u = []
hist_t = []

while cont:
    
    i+=1
    cont = i < param_iter_max
    
    if mode==0:
        #Going slightly above the disk
        u, r = compute_control(q, htm_tg_0, False, True)
        
        print("Mode 0, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 1

    if mode==1:
        #Downward movement
        u, r = compute_control(q, htm_tg_1, False, False)
        
        print("Mode 1, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 2
            robot.attach_object(disk)

    if mode==2:
        #Upward movement
        u, r = compute_control(q, htm_tg_2, False, False)
        
        print("Mode 2, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01:
            mode = 3
 
    if mode==3:
        #Move object to be delivered
        u, r = compute_control(q, htm_tg_3, True, True)
    
        print("Mode 3, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 4
            
    if mode==4:
        #Move down
        u, r = compute_control(q, htm_tg_4, True, False)
        
        print("Mode 4, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 5
            robot.detach_object(disk)
            
    if mode==5:
        #Move up
        u, r = compute_control(q, htm_tg_5, False, False)
        
        print("Mode 5, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 6
                         
    if mode==6:
        #Go press the button
        u, r = compute_control(q, htm_tg_6, False, True)
        
        print("Mode 6, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.005:
            mode = 7

    if mode==7:
        #Got the object again
        u, r = compute_control(q, htm_tg_3, False, True)
        
        print("Mode 7, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            mode = 8
            

    if mode==8:
        #Go down
        u, r = compute_control(q, htm_tg_4, False, False)
        
        print("Mode 8, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            mode = 9
            robot.attach_object(disk)
            
    if mode==9:
        #Go up
        u, r = compute_control(q, htm_tg_5, False, False)
        
        print("Mode 9, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.015: 
            mode = 10    
            

    if mode==10:
        #Move to final pose
        u, r = compute_control(q, htm_tg_7, True, True)
        
        print("Mode 10, error = "+str(round(np.linalg.norm(r),3)))
        if np.linalg.norm(r)<=0.01: 
            cont = False 
                 
    hist_q.append(np.matrix(q))
    hist_u.append(np.matrix(u))
                                       
    q = robot.q+u*dt
    
    hist_t.append(i*dt)
    robot.add_ani_frame(time = i*dt, q = q)
    
    
plt.figure() 
for i in range(7):
    
    plt.subplot(4,2,i+1)
    plt.plot(hist_t, [q[i,0] for q in hist_q])
    plt.plot(hist_t, [robot.joint_limit[i,0]+param_joint_delta for j in range(len(hist_q))])
    plt.plot(hist_t, [robot.joint_limit[i,1]-param_joint_delta for j in range(len(hist_q))])
    plt.title("Joint "+str(i+1))
    plt.tight_layout()
    # plt.figure()
    # plt.plot([u[i,0] for u in hist_q])
 
plt.figure()   
for i in range(7):
    
    plt.subplot(4,2,i+1)
    plt.plot(hist_t, [u[i,0] for u in hist_u])
    plt.plot(hist_t, [-param_max_qdot for j in range(len(hist_u))])
    plt.plot(hist_t, [param_max_qdot for j in range(len(hist_u))])
    plt.title("Velocity "+str(i+1))
    plt.tight_layout()
        
plt.show()

sim.save()