import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt

ridgeback_3d_model = ub.Model3D(
    'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Other/ridgeback_mobile.obj',
    0.0007,
    ub.Utils.trn([0,0,0.01])*ub.Utils.rotz(np.pi/2)*ub.Utils.rotx(np.pi/2), mesh_material=ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#606060"))


ridgeback = ub.RigidObject(list_model_3d=[ridgeback_3d_model])

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

    
table_top_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
table_top_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
platform = ub.Box(htm=ub.Utils.trn([-2,2.5,0.4]), width=1.8,depth=0.6,height=0.8, mesh_material=material_wood)

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

robot = ub.Robot.create_franka_emika_3(ub.Utils.trn([-2,1.7,0.21]))
ridgeback.add_ani_frame(0,htm = ub.Utils.trn([-2,1.7,0]))

sim = ub.Simulation.create_sim_mountain([ridgeback, robot, table_top_1, table_bot_1, table_top_2, table_bot_2, platform, tray_1, tray_2])
sim.set_parameters(show_grid=False)

htm_tg_table_0 = ub.Utils.trn([-1.9,2.0,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_1 = ub.Utils.trn([-1.9,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_2 = ub.Utils.trn([1.55,0.95,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)
htm_tg_table_3 = ub.Utils.trn([1.85,0.95,0.81])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)

htm_tg_table_4 = ub.Utils.trn([-2.2,2.0,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_5 = ub.Utils.trn([-2.2,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_6 = ub.Utils.trn([-0.9,-1.5,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)
htm_tg_table_7 = ub.Utils.trn([-0.6,-1.5,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)


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

dt = 0.01
human_T = 20
y_h1 = 0
x_h2 = -1
for i in range(0):
    t = i*dt
    
    if t % human_T < human_T/4 or t % human_T > 3*human_T/4:
        y_h1 += 0.005
        x_h2 += 0.003
        rotz_h1 = ub.Utils.rotz(0)
        rotz_h2 = ub.Utils.rotz(-np.pi/2)
    else:
        y_h1 -= 0.005
        x_h2 -= 0.003
        rotz_h1 = ub.Utils.rotz(-np.pi)
        rotz_h2 = ub.Utils.rotz(np.pi/2)
        
    set_human_pose(t,human_1, ub.Utils.trn([.01,y_h1,0])*rotz_h1,t)
    set_human_pose(t,human_2, ub.Utils.trn([x_h2,0,0])*rotz_h2,t)
    
sim.save()

#Compute the htm and Jacobians for the end-effector and all the
#DHi frames, considering the DOFs of the base (x,y,theta)

param_zb = 0.21

def fk_whole(_q_m, _x, _y, _theta,_robot):
    
    z_0 = np.matrix([0.,0.,1.]).T
    S_z_0 = ub.Utils.S(z_0)
    I_3x2 = np.identity(3)[:,0:2]
    I_4x4 = np.identity(4)
    zero_3x2 = np.zeros((3,2))
    
    hmt_0_DH0 = ub.Utils.trn([_x,_y,param_zb])*ub.Utils.rotz(_theta)
    Q_z = hmt_0_DH0[0:3,0:3]
    p = hmt_0_DH0[0:3,-1]
    
    #We need to call the function  jac_geo with htm=np.identity(4), because otherwise it
    #will use the one stored inside the robot, that was set with .add_ani_frame( htm = ...)
    list_jac_DH0_DHi, list_htm_DH0_DHi = _robot.jac_geo(q=_q_m, htm = I_4x4, axis='dh')
    jac_DH0_eef, htm_DH0_eef = _robot.jac_geo(q=_q_m, htm = I_4x4, axis='eef')
    
    list_jac_0_DHi = []
    list_htm_0_DHi = []
    
    #Compute for the end-effector
    htm_0_eef = hmt_0_DH0 * htm_DH0_eef
    s_eef  = htm_0_eef[0:3,-1]
    
    jac_v = jac_DH0_eef[0:3,:]
    jac_w = jac_DH0_eef[3:6,:]
    
    jac_v_0_eef = np.hstack((Q_z*jac_v, I_3x2, S_z_0*(s_eef-p)))
    jac_w_0_eef = np.hstack((Q_z*jac_w, zero_3x2, z_0))
    jac_0_eef = np.vstack( (jac_v_0_eef, jac_w_0_eef))
    
    
    #Compute for the frame DH-i
    for i in range(len(list_jac_DH0_DHi)):
        
        list_htm_0_DHi.append(hmt_0_DH0 * list_htm_DH0_DHi[i])
        s_i = list_htm_0_DHi[i][0:3,-1]
        
        jac_v = list_jac_DH0_DHi[i][0:3,:]
        jac_w = list_jac_DH0_DHi[i][3:6,:]
        
        jac_v_0_DHi = np.hstack((Q_z*jac_v, I_3x2, S_z_0*(s_i-p)))
        jac_w_0_DHi = np.hstack((Q_z*jac_w, zero_3x2, z_0))
        
        list_jac_0_DHi.append(np.vstack( (jac_v_0_DHi, jac_w_0_DHi)   ))
     
  
    return htm_0_eef, jac_0_eef, list_htm_0_DHi, list_jac_0_DHi
        
        
def task_fun_whole(_q_m, _x, _y, _theta, _robot, _htm_tg):
    
    htm_0_eef, jac_0_eef, _, _  = fk_whole(_q_m, _x, _y, _theta,_robot)
    
    x_hat = htm_0_eef[0:3,0]
    y_hat = htm_0_eef[0:3,1]
    z_hat = htm_0_eef[0:3,2]
    s = htm_0_eef[0:3,3]

    x_hat_d = _htm_tg[0:3,0]
    y_hat_d = _htm_tg[0:3,1]
    z_hat_d = _htm_tg[0:3,2]
    s_d = _htm_tg[0:3,3]
    
    r = np.matrix(np.zeros((6,1)))
    r[0:3,0] = s-s_d
    r[3,0] = 1.0-x_hat_d.T*x_hat
    r[4,0] = 1.0-y_hat_d.T*y_hat
    r[5,0] = 1.0-z_hat_d.T*z_hat
    
    no_joint = np.shape(robot.q)[0]
    
    jac_r = np.matrix(np.zeros((6, no_joint+3)))
    jac_r[0:3, :] = jac_0_eef[0:3, :]
    jac_r[3, :] = x_hat_d.T * ub.Utils.S(x_hat) * jac_0_eef[3:6, :]
    jac_r[4, :] = y_hat_d.T * ub.Utils.S(y_hat) * jac_0_eef[3:6, :]
    jac_r[5, :] = z_hat_d.T * ub.Utils.S(z_hat) * jac_0_eef[3:6, :]
    
    
    return r, jac_r

def compute_dist_whole(_q_m, _x, _y, _theta, _robot, _obj):
            
                
    htm_0_eef, jac_0_eef, list_htm_0_DHi, list_jac_0_DHi =  fk_whole(_q_m, _x, _y, _theta,_robot)
    
    ds = ub.Robot.create_abb_crb().compute_dist(obj = _obj, max_dist=1.0)
    
    
    
###################################   
#Parameters
param_eta = 1.2
param_eps = 0.001
param_k = 1.0
param_max_qdot = 1.5
param_obs_delta = 0.025 
param_joint_delta = 2*np.pi/180
dt=0.005
param_iter_max = 10000
param_use_pc = False

def fun_G(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
       
    return out


def control_fun(_q_m, _x, _y, _theta, _robot, _htm_tg, _holding_tray, _consider_collision):
    
    r, jac_r = task_fun_whole(_q_m, _x, _y, _theta, _robot, _htm_tg)
    
    no_joint = np.shape(robot.q)[0]
    
    H = jac_r.T * jac_r + param_eps * np.identity(no_joint+3)
    f = -jac_r.T * fun_G(r, param_k)
    
    
    #Add the equality constraint for the non-holonomic constraint
    #The equality u_x*sin(theta)-u_y*cos(theta) = 0 is written as
    # u_x*sin(theta)-u_y*cos(theta) >= 0
    #-u_x*sin(theta)+u_y*cos(theta) >=0
    A_nhol = np.matrix(np.zeros((1,10)))
    A_nhol[0,-3] = np.sin(_theta)
    A_nhol[0,-2] = -np.cos(_theta)
    
    A = np.vstack((A_nhol, -A_nhol))
    b = np.vstack((0,0))
    
    #If the robot is holding the tray, the constraint for the orientation
    #of the axis should be a hard constraint, that is
    # (d/dt) r_{rot_x}(q) = G(r_{rot,x}(q)) should hold true
    
    if _holding_tray:
        A_r_rot_x = jac_r[3,:]
        b_r_rot_x = fun_G(np.matrix(r[3,0]), param_k)
        
        A = np.vstack((A, A_r_rot_x, -A_r_rot_x))
        b = np.vstack((b, b_r_rot_x, -b_r_rot_x))
        
 
    #Create the CBF constraint for joint limits
    I_ext = np.hstack( (np.identity(no_joint), np.zeros((no_joint,3)) ) )
     
    A = np.vstack((A, I_ext  ))
    b = np.vstack((b, -param_eta*(_q_m-robot.joint_limit[:,0]-param_joint_delta) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_eta*(robot.joint_limit[:,1]-_q_m-param_joint_delta) ))  
    
    #Implement velocity limits
    A = np.vstack((A, I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint,1))) ))  
    A = np.vstack((A, -I_ext))
    b = np.vstack((b, -param_max_qdot*np.matrix(np.ones((no_joint,1))) ))   
        
    u = ub.Utils.solve_qp(H, f, A, b)
    
    
    
    return u[0:7,:], u[7,0], u[8,0], u[9,0], np.linalg.norm(r)

 
 
 
 
x = ridgeback.htm[0,-1]
y = ridgeback.htm[1,-1]
theta = 0
q_m = np.matrix(robot.q)

mode = 0

htm_tg = [htm_tg_table_0, htm_tg_table_1, htm_tg_table_0, htm_tg_table_2, htm_tg_table_3, htm_tg_table_2, htm_tg_table_4]
holding_tray = [False,False,True,True, False, False, False]
consider_collision = [True,False,False,True, False, False, True]

cont = True
i = 0

while cont:
    t = i*dt
    
    i+=1
    
    
    qdotm, dotx, doty, dottheta, error = control_fun(q_m, x, y, theta, robot, htm_tg[mode], holding_tray[mode], consider_collision[mode])
    
    q_m+=qdotm*dt
    x+=dotx*dt
    y+=doty*dt
    theta+=dottheta*dt
    
    robot.add_ani_frame(time = t, q = q_m, htm = ub.Utils.trn([x,y,param_zb])*ub.Utils.rotz(theta))
    ridgeback.add_ani_frame(time = t, htm = ub.Utils.trn([x,y,0])*ub.Utils.rotz(theta))
    
    print("Mode "+str(mode)+" = "+str(round(error,3))+", "+str(i))
    if error <= 0.005:
        mode+= 1
        
        if mode == 2:
            robot.attach_object(tray_1)
            
        if mode == 5:
            robot.detach_object(tray_1)    
        cont = mode < 7 

    
    cont = cont and i < 3500
    
    
    

sim.save()