import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt


def add_mobile_base(robot):

    ridgeback_3d_model = ub.Model3D(
    'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Other/ridgeback_mobile.obj',
    0.0007,
    ub.Utils.trn([0,0,-0.2])*ub.Utils.rotz(0)*ub.Utils.rotx(np.pi/2), mesh_material=ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#606060"))


    links_new = []
    
    links_new.append(ub.Link(0,  0      , 0, -np.pi/2, 0, 1, []))
    links_new.append(ub.Link(1,  -np.pi/2, 0,  np.pi/2, 0, 1, []))
    #links_new.append(ub.Link(2,  0, 0.21, 0, 0, 0, robot.list_object_3d_base+[ridgeback_3d_model]))
    links_new.append(ub.Link(2,  0, 0.21, 0, 0, 0, robot.list_object_3d_base))
    
    #links_new[-1].attach_col_object(ub.Box(width=0.7,depth=0.6,height=0.2, color='blue', opacity=0.7), ub.Utils.trn([0,0,-0.1]))
    links_new[-1].attach_col_object(ub.Cylinder(radius=0.38,height=0.2, color='blue', opacity=0.7), ub.Utils.trn([0,0,-0.1]))
    
        
    i = 3
    for link in robot.links:
        links_new.append(ub.Link(i, link.theta, link.d, link.alpha, link.a, link.joint_type, link.list_model_3d))
        
        for j in range(len(link.col_objects)): 
            links_new[i].attach_col_object(link.col_objects[j][0], link.col_objects[j][1])
            
        i+=1
        
    q0 = np.vstack((0,0,0, robot.q0))
    inf_bound = np.matrix([-1000,1000])
    joint_limits = np.vstack( (inf_bound, inf_bound, inf_bound, robot.joint_limit))
    
    return ub.Robot("base"+robot.name, links_new, [], robot.htm, robot.htm_base_0*ub.Utils.roty(np.pi/2), robot.htm_n_eef, q0, robot.eef_frame_visible, joint_limits)
        
        


texture_wood = ub.Texture(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Textures/wood_1.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

material_wood= ub.MeshMaterial(metalness=0, clearcoat=0, roughness=1.0, flat_shading=True, normal_scale=[0, 0], texture_map=texture_wood)

glass_material = ub.MeshMaterial(
    opacity=1.0,                   # Should be 1 if using transmission
    transmission=0.98,             # High transmission for glass-like effect
    roughness=0.2,                # Very smooth
    metalness=0.0,                 # Must be zero for dielectric (glass)
    ior=1.52,                      # Typical glass index of refraction
    reflectivity=0.7,              # Strong reflections
    env_map_intensity=1.2,         # Amplify environment reflection
    refraction_ratio=0.985,        # Subtle, usually close to 1
    clearcoat=0.0,                 # Glass does not need clearcoat
    clearcoat_roughness=0.0,
    specular_intensity=1.0,        # Boost highlights for clean reflections
    color='white',                 # True clear glass — subtle tint if desired
    emissive_intensity=0.0,        # Glass usually doesn't emit
    flat_shading=False,            # Always smooth for glass
    normal_scale=[0.5, 0.5],       # Normal strength — reduce if overpowered
    side="DoubleSide"              # Double-sided avoids weird visibility issues
)

    
table_top_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5-0.4,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5-0.4,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
table_top_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
platform = ub.Box(htm=ub.Utils.trn([-2,2.5,0.4]), width=1.8,depth=0.6,height=0.8, mesh_material=material_wood)

all_obstacles=[table_top_1,table_bot_1,table_top_2,table_bot_2,platform]

tray_1 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=glass_material),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='red')]
, htm = ub.Utils.trn([-1.9,2.3,0.81]))

tray_2 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=glass_material),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='black')]
, htm = ub.Utils.trn([-2.2,2.3,0.81]))

robot = add_mobile_base(ub.Robot.create_franka_emika_3())

q0 = np.matrix(robot.q)
q0[0,0] = -2.0
q0[1,0] = 1.7
q0[2,0] = 0
robot.add_ani_frame(0,q=q0)


sim = ub.Simulation.create_sim_mountain([robot, table_top_1, table_bot_1, table_top_2, table_bot_2, platform, tray_1, tray_2])
sim.set_parameters(show_grid=False)

htm_tg_table_0 = ub.Utils.trn([-1.9,2.0-0.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_1 = ub.Utils.trn([-1.9,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_2 = ub.Utils.trn([1.55-0.15,0.95,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)
htm_tg_table_3 = ub.Utils.trn([1.85,0.95,0.81+0.09])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)

htm_tg_table_4 = ub.Utils.trn([-2.2,2.0-0.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_5 = ub.Utils.trn([-2.2,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_6 = ub.Utils.trn([-0.6+0.15,-1.5-0.4,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)
htm_tg_table_7 = ub.Utils.trn([-0.9,-1.5-0.4,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)


sim.add(ub.Frame(htm_tg_table_0, size=0.2))
sim.add(ub.Frame(htm_tg_table_1, size=0.2))
sim.add(ub.Frame(htm_tg_table_2, size=0.2))
sim.add(ub.Frame(htm_tg_table_3, size=0.2))
sim.add(ub.Frame(htm_tg_table_4, size=0.2))
sim.add(ub.Frame(htm_tg_table_5, size=0.2))
sim.add(ub.Frame(htm_tg_table_6, size=0.2))
sim.add(ub.Frame(htm_tg_table_7, size=0.2))

sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=True, show_grid=False, camera_type='perspective', camera_start_pose=[1.0,1.0,5.5,0,0,0,0.8])

#Functions for creating an human and animate it moving around
def create_human(torso_color):
    objects = []

    # Head
    head = ub.Ball(htm=ub.Utils.trn([0 ,   0 ,   1.3500]),radius=0.15,color="peachpuff")
    objects.append(head)

    # Torso
    torso = ub.Cylinder(htm=ub.Utils.trn([0, 0, 1.0]),radius=0.22,height=0.5,color=torso_color)
    objects.append(torso)

    # Left  Arm
    left_lower_arm = ub.Cylinder(htm=ub.Utils.trn([0.29, 0, 0.90]) ,radius=0.07,height=0.6,color="peachpuff")
    objects.append(left_lower_arm)

    # Right  Arm
    right_lower_arm = ub.Cylinder(htm=ub.Utils.trn([-0.29, 0, 0.90]) ,radius=0.07,height=0.6,color="peachpuff")
    objects.append(right_lower_arm)

    # Left  Leg
    left_lower_leg = ub.Cylinder(htm=ub.Utils.trn([0.10, 0, 0.40]),radius=0.08,height=0.8,color="green")
    objects.append(left_lower_leg)

    # Right  Leg
    right_lower_leg = ub.Cylinder(htm=ub.Utils.trn([-0.10, 0, 0.40]),radius=0.08,height=0.8,color="green")
    objects.append(right_lower_leg)
    
    # Eye left
    eye_left = ub.Ball(htm=ub.Utils.trn([-0.06 ,   0.12 ,   1.400]),radius=0.02,color="black")
    objects.append(eye_left)

    # Eye right
    eye_right = ub.Ball(htm=ub.Utils.trn([0.06 ,   0.12 ,   1.400]),radius=0.02,color="black")
    objects.append(eye_right)
    
    #Bounding cylinder
    bcyl = ub.Cylinder(htm=ub.Utils.trn([0, 0, 0.9]),radius=0.4,height=1.8,color=torso_color, opacity=0.3)
    objects.append(bcyl)
        
    return ub.Group(objects)

def set_human_pose(_t, _human, _htm, walk_cycle):
    
 
    _human.add_ani_frame(time = _t, htm = _htm)
    
    theta = 0.15*np.sin(2*np.pi*walk_cycle)
    _human.list_of_objects[2].add_ani_frame(time =_t, htm = _human.list_of_objects[2].htm * ub.Utils.trn([0,0,0.3]) * ub.Utils.rotx(theta) * ub.Utils.trn([0,0,-0.3]))
    _human.list_of_objects[3].add_ani_frame(time =_t, htm = _human.list_of_objects[3].htm * ub.Utils.trn([0,0,0.3]) * ub.Utils.rotx(-theta) * ub.Utils.trn([0,0,-0.3]))    
    _human.list_of_objects[4].add_ani_frame(time =_t, htm = _human.list_of_objects[4].htm * ub.Utils.trn([0,0,0.4]) * ub.Utils.rotx(-theta) * ub.Utils.trn([0,0,-0.4]))
    _human.list_of_objects[5].add_ani_frame(time =_t, htm = _human.list_of_objects[5].htm * ub.Utils.trn([0,0,0.4]) * ub.Utils.rotx(theta) * ub.Utils.trn([0,0,-0.4])) 
    
human_1 = create_human('blue')
human_2 = create_human('red')
sim.add([human_1, human_2])

##
# for link in robot.links:
#     for obj in link.col_objects:
#         sim.add(obj[0])

##
    
sim.save()

#Compute the htm and Jacobians for the end-effector and all the
#DHi frames, considering the DOFs of the base (x,y,theta)

param_zb = 0.21

    
    
###################################   
#Parameters
param_eta = 0.2
param_eps = 0.005
param_k = 0.3
param_max_qdot = 1.5
param_obs_delta = 0.10 #0.15 #0.02 0.10
param_joint_delta = 2*np.pi/180
dt=0.01 #dt=0.005
param_iter_max = 10000
param_use_pc = False

def fun_G(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
      

    return out


def control_fun(_q, _robot, _htm_tg, _obstacles, _humans, _tray, hist_dist_human, _vel_human, _holding_tray, _care_obstacles, _old_ds):
    

    no_joint = np.shape(robot.q)[0]
    r, jac_r = _robot.task_function(_q, _htm_tg)

    if np.linalg.norm(r[0:3])>0.5:
        r = r[0:3]
        jac_r = jac_r[0:3,:]
    
    A = np.matrix(np.zeros((0,no_joint)))
    b = np.matrix(np.zeros((0,1)))

    #If the robot is holding the tray, the constraint for the orientation
    #of the axis should be a hard constraint, that is
    # (d/dt) r_{rot_x}(q) = G(r_{rot,x}(q)) should hold true
    
    jac, htm = _robot.jac_geo(_q)
    s_e = htm[0:3,-1]
    jac_v = jac[0:3,:]
    jac_w = jac[3:6,:]
                        
    if _holding_tray:
        y_d = _htm_tg[0:3,1]
        z_d = _htm_tg[0:3,2]
        
        x_e = htm[0:3,0]
        
        r_rotx = np.vstack((y_d.T*x_e,z_d.T*x_e))
        b_rotx = -param_k*r_rotx
        jac_r_rotx = -np.vstack(( (y_d.T*ub.Utils.S(x_e))*jac[3:6,:], (z_d.T*ub.Utils.S(x_e))*jac[3:6,:]))
        
        A = np.vstack((A, jac_r_rotx, -jac_r_rotx))
        b = np.vstack((b, b_rotx-0.001, -b_rotx-0.001))
            
    #Implement the anti-stuck constraint
    # p = _q[0:2,-1]
    # jac_eef, htm_eef = robot.jac_geo(q=_q)
    # s = htm_eef[0:2,-1]
    
    theta = _q[2,0]
    
    # deltap = s-p
    # e_theta = np.matrix([np.cos(theta), np.sin(theta)])
    # e_theta_R = np.matrix([-np.sin(theta), np.cos(theta)])
    
    # E = e_theta*deltap-0.05
    # mat_1 = jac_eef[0:2,:]
    # mat_2 = np.hstack((np.identity(2), np.zeros((2,no_joint-2))))
    # jac_E = e_theta*(mat_1-mat_2)
    # jac_E[0,2]+= e_theta_R*deltap
    
    # print("E "+str(round(E[0,0],3)))
    
    # A = np.vstack((A, jac_E))
    # b = np.vstack((b, -param_eta*E))
    
    # p = _q[0:2,-1]
    # s = htm[0:2,-1]
    # deltap = 0.8-np.linalg.norm(s-p)
    # mat_1 = jac[0:2,:]
    # mat_2 = np.hstack((np.identity(2), np.zeros((2,no_joint-2))))
    # jac_deltap = -(s-p).T*(mat_1-mat_2)/np.linalg.norm(s-p)
    
    # A = np.vstack((A, jac_deltap))
    # b = np.vstack((b, -param_eta*deltap))
    # print("dp = "+str(round(deltap,3)))    
    
    

        
    # print("E "+str(round(E[0,0],3)))
    

        
    H = jac_r.T * jac_r + param_eps * np.identity(no_joint)
    f = -jac_r.T * fun_G(r, param_k)

    
    #Add the equality constraint for the non-holonomic constraint
    #The equality u_x*sin(theta)-u_y*cos(theta) = 0 is written as
    # u_x*sin(theta)-u_y*cos(theta) >= 0
    #-u_x*sin(theta)+u_y*cos(theta) >=0
    # A_nhol = np.matrix(np.zeros((1,no_joint)))
    
    # A_nhol[0, 0] = np.sin(theta)
    # A_nhol[0, 1] = -np.cos(theta)
    
    # A = np.vstack((A, A_nhol, -A_nhol))
    # b = np.vstack((b, -0.0001,-0.0001))
    
    
    #Avoid humans

    #Between the robot and the objects
    print("Obs")
    
    hist_dd=np.matrix(np.zeros((0,1)))
    
    hist_ds = []
    k=0
    
    for obs in _obstacles:
        # if len(_old_ds)==0:
        #     ds = robot.compute_dist(q = _q, obj = obs, h=0.05, eps=0.02, tol=1e-5,no_iter_max=2000)
        # else:
        #     ds = robot.compute_dist(q = _q, obj = obs, h=0.05, eps=0.02, tol=1e-5,no_iter_max=2000)
        #     k =k+1
        
        ds = robot.compute_dist(q = _q, obj = obs, max_dist=0.6)
        
        hist_ds.append(ds)
        #ds = robot.compute_dist(q = _q, obj = obs)
        
        hist_dd = np.vstack((   hist_dd,   np.matrix(np.linalg.norm(ds.jac_dist_mat[:,:], axis=1)).T    )) 
        
        
        # aa = np.min( np.linalg.norm(ds.jac_dist_mat[:,0:3], axis=1)  )
        # print(aa)
        
        # if aa<0.01:
        #     uu=0
        
        if _care_obstacles:
            A = np.vstack((A, ds.jac_dist_mat))
            b = np.vstack((b, -param_eta*(ds.dist_vect-param_obs_delta)))
        else:
            A = np.vstack((A, ds.jac_dist_mat[0:-5,:]))
            b = np.vstack((b, -param_eta*(ds.dist_vect[0:-5,:]-param_obs_delta)))  
           
             
        if _holding_tray and _care_obstacles:
            for obs in _obstacles:
                point_disk, point_obs, dist, _ = _tray.list_of_objects[0].compute_dist(obs)

                jac_dist = ((point_disk - point_obs).T * jac_v + np.cross((point_disk - s_e ).T, (point_disk - point_obs).T)  * jac_w)/dist
                A = np.vstack((A, jac_dist))
                b = np.vstack((b, -param_eta*(dist-param_obs_delta)))

            
    # dist_stack = np.matrix(np.zeros((0,1)))
    j = 0
    
    for j in range(len(_humans)):
        ds = _robot.compute_dist(q = _q, obj = _humans[j].list_of_objects[-1], max_dist=3.0)
        
        for ind1 in range(len(_robot.links)):
            for ind2 in range(len(_robot.links[ind1].col_objects)):
                try:
                    item = ds.get_item(ind1,ind2)
                    p1 = item.point_object
                    p2 = item.point_link
                    dv = ((p1-p2)/np.linalg.norm(p1-p2)).T
                    ff = dv[:,0:2]*_vel_human[j]
                    
                    A = np.vstack((A, item.jac_distance))
                    b = np.vstack((b, -1.5*param_eta*(item.distance-0.30)-ff)) #param_obs_delta-0.30
                except:
                    pass
                #print(-2*param_eta*(item.distance-param_obs_delta-0.40)-ff)
        
        if _holding_tray and _care_obstacles:
            point_tray, point_hmn, dist, _ = _tray.list_of_objects[0].compute_dist(_humans[j].list_of_objects[-1])
            jac_dist = ((point_tray - point_hmn).T * jac_v + np.cross((point_tray - s_e ).T, (point_tray - point_hmn).T)  * jac_w)/dist
            
            
            dv = ((point_hmn-point_tray)/np.linalg.norm(point_hmn-point_tray)).T
            ff = dv[:,0:2]*_vel_human[j]
            
            A = np.vstack((A, jac_dist))
            b = np.vstack((b, -1.5*param_eta*(dist-0.30)-ff))        
            
        
        
        #dist_stack = np.vstack( (dist_stack, ds.dist_vect) )
        
        

        
    # hist_dist_human.append(dist_stack)
                

    
    # print(f.T)

        
 
    #Create the CBF constraint for joint limits
    I_ext = np.hstack( (np.zeros((no_joint-3,3)), np.identity(no_joint-3)) )
     
    A = np.vstack((A, I_ext  ))
    b = np.vstack((b, -param_eta*(_q[3:]-robot.joint_limit[3:,0]-param_joint_delta) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_eta*(robot.joint_limit[3:,1]-_q[3:]-param_joint_delta) )) 
    
    # print("Joints!")
    # print( (_q[3:]-robot.joint_limit[3:,0]-param_joint_delta).T)
    # print( (robot.joint_limit[3:,1]-_q[3:]-param_joint_delta).T)
    
    #Implement velocity limits
    A = np.vstack((A, I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint-3,1))) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint-3,1))) ))   
       
    try:
        u = ub.Utils.solve_qp(H, f, A, b)
    except:
        u = 0*_robot.q


    
    return u, np.linalg.norm(r), hist_dd, hist_ds

 
 
 
 
q = np.matrix(robot.q)

mode = 0

htm_tg = [htm_tg_table_0, htm_tg_table_1, htm_tg_table_0, htm_tg_table_2, htm_tg_table_3, htm_tg_table_2, htm_tg_table_4, htm_tg_table_5, htm_tg_table_4, htm_tg_table_6, htm_tg_table_7,htm_tg_table_0]
holding_tray = [False, False, True, True, True, False, False , False, True, True, True, False]
consider_collision = [True,False,False,True, False, False, True, False, False, True, False, True]

obj_to_hold = tray_1
cont = True
i = 0

dt = 0.02
human_T = 15
y_h1 = 0.5
x_h2 = -0.5


param_spd_human_1 = 0.3 #0.008
param_spd_human_2 = 0.2 #0.006

hist_dist_human = []

t = 0

hist_u = []

dotq = np.matrix(0*robot.q)

hist_dp = []
hist_ds = []
hist_error = []
vel_human = [np.matrix([0,0]).T, np.matrix([0,0]).T]
while cont:
    
    
    i+=1
    
    #Human movements
    if t % human_T < human_T/4 or t % human_T > 3*human_T/4:
        y_h1 += param_spd_human_1*dt
        x_h2 += param_spd_human_2*dt
        rotz_h1 = ub.Utils.rotz(0)
        rotz_h2 = ub.Utils.rotz(-np.pi/2)
        
        vel_human[0] = np.matrix([0.,param_spd_human_1]).T
        vel_human[1] = np.matrix([param_spd_human_2,0.]).T
    else:
        y_h1 -= param_spd_human_1*dt
        x_h2 -= param_spd_human_2*dt
        rotz_h1 = ub.Utils.rotz(-np.pi)
        rotz_h2 = ub.Utils.rotz(np.pi/2)

        vel_human[0] = np.matrix([0.,-param_spd_human_1]).T
        vel_human[1] = np.matrix([-param_spd_human_2,0.]).T
                
    set_human_pose(t,human_1, ub.Utils.trn([.01,y_h1,0])*rotz_h1,t)
    set_human_pose(t,human_2, ub.Utils.trn([x_h2,0,0])*rotz_h2,t)
    
    
    ###    
    
    
        
    new_dotq, error, hist_dd, hist_ds = control_fun(q, robot, htm_tg[mode], all_obstacles, [human_1, human_2],  
                              obj_to_hold, hist_dist_human, vel_human, holding_tray[mode], consider_collision[mode], hist_ds)
    
    #dotq = 0.95*dotq+0.05*new_dotq
    dotq = 0.00*dotq+1.00*new_dotq
    
    hist_u.append(dotq)
    hist_dp.append(hist_dd)
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
         

    
    cont = cont and i < 10000 #6000
    
plt.figure()    
for i in range(0,3):
    plt.plot([j for j in range(len(hist_u))], [u[i,0] for u in hist_u])    


plt.figure()    
plt.plot([j for j in range(len(hist_error))], hist_error)  
    
        
plt.show()

sim.save()