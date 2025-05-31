import numpy as np
import uaibot as ub


#Save this code as "helper_anim_part_3_2.py"

#Add the mobile base to a manipulator
def add_mobile_base(robot):

    ridgeback_3d_model = ub.Model3D(
    'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Other/ridgeback_mobile.obj',
    0.0007,
    ub.Utils.trn([0,0,-0.2])*ub.Utils.rotz(0)*ub.Utils.rotx(np.pi/2), mesh_material=ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#606060"))


    links_new = []
    
    links_new.append(ub.Link(0,  0      , 0, -np.pi/2, 0, 1, []))
    links_new.append(ub.Link(1,  -np.pi/2, 0,  np.pi/2, 0, 1, []))
    links_new.append(ub.Link(2,  0, 0.21, 0, 0, 0, robot.list_object_3d_base+[ridgeback_3d_model]))
    #links_new.append(ub.Link(2,  0, 0.21, 0, 0, 0, robot.list_object_3d_base))
    
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
        
        
#Functions for creating an human 
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
    bcyl = ub.Cylinder(htm=ub.Utils.trn([0, 0, 0.9]),radius=0.4,height=1.8,color=torso_color, opacity=0.0)
    objects.append(bcyl)
        
    return ub.Group(objects)

#Function to set human pose
def set_human_pose(_t, _human, _htm, walk_cycle):
    
    _human.add_ani_frame(time = _t, htm = _htm)
    
    theta = 0.15*np.sin(2*np.pi*walk_cycle)
    _human.list_of_objects[2].add_ani_frame(time =_t, htm = _human.list_of_objects[2].htm * ub.Utils.trn([0,0,0.3]) * ub.Utils.rotx(theta) * ub.Utils.trn([0,0,-0.3]))
    _human.list_of_objects[3].add_ani_frame(time =_t, htm = _human.list_of_objects[3].htm * ub.Utils.trn([0,0,0.3]) * ub.Utils.rotx(-theta) * ub.Utils.trn([0,0,-0.3]))    
    _human.list_of_objects[4].add_ani_frame(time =_t, htm = _human.list_of_objects[4].htm * ub.Utils.trn([0,0,0.4]) * ub.Utils.rotx(-theta) * ub.Utils.trn([0,0,-0.4]))
    _human.list_of_objects[5].add_ani_frame(time =_t, htm = _human.list_of_objects[5].htm * ub.Utils.trn([0,0,0.4]) * ub.Utils.rotx(theta) * ub.Utils.trn([0,0,-0.4])) 
 
 
#Create some materials
material_wood = ub.MeshMaterial.create_wood()
material_glass= ub.MeshMaterial.create_glass()

#Create the static obstacles    
table_top_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5-0.4,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_1 = ub.Cylinder(htm=ub.Utils.trn([-1.5,-1.5-0.4,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
table_top_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.8]), radius=0.7,height=0.02, mesh_material=material_wood)
table_bot_2 = ub.Cylinder(htm=ub.Utils.trn([2.5,1,0.4]), radius=0.1,height=0.8, mesh_material=material_wood)
platform = ub.Box(htm=ub.Utils.trn([-2,2.5,0.4]), width=1.8,depth=0.6,height=0.8, mesh_material=material_wood)

all_obstacles=[table_top_1,table_bot_1,table_top_2,table_bot_2,platform]

#Create the tray objects
tray_1 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=material_glass),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='red')]
, htm = ub.Utils.trn([-1.9,2.3,0.81]))

tray_2 = ub.Group(
    [ub.Cylinder(color='white',radius=0.13,height=0.02),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.10]), radius=0.05,height=0.18, mesh_material=material_glass),
     ub.Cylinder(htm=ub.Utils.trn([0,0,0.08]), radius=0.04,height=0.12, color='black')]
, htm = ub.Utils.trn([-2.2,2.3,0.81]))

#Create the robot
robot = add_mobile_base(ub.Robot.create_franka_emika_3())

#Create the target poses
htm_tg_table_0 = ub.Utils.trn([-1.9,2.0-0.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_1 = ub.Utils.trn([-1.9,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_2 = ub.Utils.trn([1.55-0.15,0.95,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)
htm_tg_table_3 = ub.Utils.trn([1.85,0.95,0.81+0.09])*ub.Utils.roty(np.pi/2)*ub.Utils.rotz(np.pi)

htm_tg_table_4 = ub.Utils.trn([-2.2,2.0-0.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)
htm_tg_table_5 = ub.Utils.trn([-2.2,2.15,0.81])*ub.Utils.rotx(-np.pi/2)*ub.Utils.rotz(-np.pi/2)

htm_tg_table_6 = ub.Utils.trn([-0.6+0.15,-1.5-0.4,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)
htm_tg_table_7 = ub.Utils.trn([-0.9,-1.5-0.4,0.9])*ub.Utils.roty(np.pi/2)*ub.Utils.rotx(np.pi)*ub.Utils.rotz(np.pi)

#Set the initial configuration
q0 = np.matrix([-2.0, 1.7, 0.0, 0.0, 0.0, 0.0 ,-0.0698, 0.0, 0.0, 0.0])
robot.add_ani_frame(0,q=q0)

#Create simulation
sim = ub.Simulation.create_sim_mountain([robot, table_top_1, table_bot_1, table_top_2, table_bot_2, platform, tray_1, tray_2])
sim.set_parameters(show_grid=False)
sim.add(ub.Frame(htm_tg_table_0, size=0.2))
sim.add(ub.Frame(htm_tg_table_1, size=0.2))
sim.add(ub.Frame(htm_tg_table_2, size=0.2))
sim.add(ub.Frame(htm_tg_table_3, size=0.2))
sim.add(ub.Frame(htm_tg_table_4, size=0.2))
sim.add(ub.Frame(htm_tg_table_5, size=0.2))
sim.add(ub.Frame(htm_tg_table_6, size=0.2))
sim.add(ub.Frame(htm_tg_table_7, size=0.2))

sim.set_parameters(load_screen_color="#191919", background_color="#191919", 
                   width=500, height=500, show_world_frame=True, show_grid=False, camera_type='perspective', 
                   camera_start_pose=[1.0,1.0,5.5,0,0,0,0.8], pixel_ratio=0.8)

human_john_connor = create_human('blue')
human_kyle_reese = create_human('red')
all_humans = [human_john_connor, human_kyle_reese]

sim.add(all_humans)


    