import uaibot as ub
from scipy.linalg import null_space
import numpy as np
import requests
from io import BytesIO

#Animação da junta revolta


#Animação da junta revolta


csp = [1.0,-1.0,1.0,-1.2+0.5,1.0+0.5,0.0,1.0]

csp = [1.0,-1.0,1.0,0.0,0.5,0.0,2.0]

light1 =  ub.PointLight(name="light1", color="white", intensity=8, htm=ub.Utils.trn([-1, -1, 1.5]))
light2 =  ub.PointLight(name="light2", color="white", intensity=8, htm=ub.Utils.trn([-1, 1, 1.5]))
light3 =  ub.PointLight(name="light3", color="white", intensity=8, htm=ub.Utils.trn([1, -1, 1.5]))
light4 =  ub.PointLight(name="light4", color="white", intensity=8, htm=ub.Utils.trn([1, 1, 1.5]))

sim = ub.Simulation([light1,light2,light3,light4])
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_start_pose = csp)




robot = ub.Robot.create_magician_e6(eef_frame_visible=False)

q = np.matrix(robot.q)
q[-2]+=np.pi/2
robot.add_ani_frame(0,q)

sim.add([robot])

for link in robot.links:
    for co in link.col_objects:
        co[0].mesh_material._opacity=0.7
        sim.add(co[0])
  
dt=0.01
htm_tg = ub.Utils.trn([0.6,0.2,-0.3])*robot.fkm()
for i in range(1000):      
    r, Jr = robot.task_function(htm_des=htm_tg)
    
    qdot = ub.Utils.dp_inv(Jr)*(-0.5*r)
    
    if i<100:
        qdot[3]=0
    
    robot.add_ani_frame(i*dt,robot.q+qdot*dt)
    robot.update_col_object(i*dt)




sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/","part_2_2")