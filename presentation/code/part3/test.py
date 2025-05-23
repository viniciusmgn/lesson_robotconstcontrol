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

dish_plate_1 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=glass_material),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='red')]
, htm = ub.Utils.trn([-1.9,2.3,0.81]))

dish_plate_2 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=glass_material),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='black')]
, htm = ub.Utils.trn([-2.2,2.3,0.81]))

robot = ub.Robot.create_franka_emika_3(ub.Utils.trn([-2,1.7,0.21]))
ridgeback.add_ani_frame(0,htm = ub.Utils.trn([-2,1.7,0]))

sim = ub.Simulation.create_sim_mountain([ridgeback, robot, table_top_1, table_bot_1, table_top_2, table_bot_2, platform, dish_plate_1, dish_plate_2])
sim.set_parameters(show_grid=False)

htm_tg_table_1 = ub.Utils.trn([1.85,0.95,0.9])*ub.Utils.roty(np.pi/2)
htm_tg_table_2 = ub.Utils.trn([-0.9,-1.5,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)

sim.add(ub.Frame(htm_tg_table_1, size=0.2))
sim.add(ub.Frame(htm_tg_table_2, size=0.2))
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_type='perspective', camera_start_pose=[1.0,1.0,5.5,0,0,0,0.8])

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
for i in range(3000):
    t = i*dt
    
    print(t)
    if t % human_T < human_T/4 or t % human_T > 3*human_T/4:
        print("Pt = "+str(round(t,3)))
        y_h1 += 0.005
        x_h2 += 0.003
        rotz_h1 = ub.Utils.rotz(0)
        rotz_h2 = ub.Utils.rotz(-np.pi/2)
    else:
        print("Nt = "+str(round(t,3)))
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

def kin_base(_q_m, _x, _y, _theta,_robot):
    
    z_0 = np.matrix([0.,0.,1.]).T
    S_z_0 = ub.Utils.S(z_0)
    I_3x2 = np.identity(3)[:,0:2]
    zero_3x2 = np.zeros((3,2))
    
    hmt_0_DH0 = ub.Utils.trn([_x,_y,param_zb])*ub.Utils.rotz(_theta)
    Q_z = hmt_0_DH0[0:3,0:3]
    p = hmt_0_DH0[0:3,-1]
    
    list_jac_DH0_DHi, list_htm_DH0_DHi = _robot.jac_geo(q=_q_m, axis='dh')
    jac_DH0_eef, htm_DH0_eef = _robot.jac_geo(q=_q_m, axis='eef')
    
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
        
        
        
htm_0_eef, jac_0_eef, list_htm_0_DHi, list_jac_0_DHi =  kin_base(robot.q,0.,1.0,np.pi/2,robot)
