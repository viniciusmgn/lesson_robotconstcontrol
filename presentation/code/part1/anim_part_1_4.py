from uaibot import *
from scipy.linalg import null_space
import numpy as np




#Animação da junta prismática

robot = Robot.create_epson_t6(htm=Utils.rotz(3.14/2), color="gray", opacity=0.3)
sim = Simulation([robot], load_screen_color="#191919", show_grid=False, show_world_frame=False, background_color="#191919",
                 camera_type="orthographic", width=500, height=400, camera_start_pose=[1,0.1,0.35,-3.14,3.14/2,-3.14/2,4])


fact = 0.25/0.3

joint_0 = Box(htm = Utils.trn([0,0,0.2]), color="blue", width=0.04, depth=0.05, height=0.6, opacity=1)
joint_cylinder_0 = Cylinder(htm = Utils.trn([0,0,0.2]), color="magenta", radius=0.05, height=fact*0.6, opacity=1)

joint_1 = Box(htm = robot.fkm(axis='dh')[0], color="blue", width=0.04, depth=0.05, height=0.4, opacity=1)
joint_cylinder_1 = Cylinder(htm = robot.fkm(axis='dh')[0], color="magenta", radius=0.04, height=fact*0.4, opacity=1)

joint_2 = Box(htm = robot.fkm(axis='dh')[2], color="blue", width=0.04, depth=0.05, height=0.3, opacity=1)
joint_cylinder_2 = Cylinder(htm = robot.fkm(axis='dh')[2]* Utils.trn([0,0,-0.1]), color="magenta", radius=0.04, height=0.3*fact, opacity=1)


light1 = PointLight(name="light1", color="white", intensity=8, htm=Utils.trn([-1, -1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=8, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=8, htm=Utils.trn([1, -1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=8, htm=Utils.trn([1, 1, 1.5]))

style = "top:" + str(0.8 * sim.height) + "px;right:" + str(0) + "px;width:" + str(
    sim.width) + "px;position:absolute;text-align:center;background-color:#191919;color:white;font-smooth:always;font-family:arial"
explanation = HTMLDiv(html_text="", style=style)

sim.add(joint_0)
sim.add(joint_cylinder_0)
sim.add(joint_1)
sim.add(joint_cylinder_1)
sim.add(joint_2)
sim.add(joint_cylinder_2)
sim.add([light1,light2,light3,light4,explanation])

dt=0.01
q0 = robot.q


for i in range(3000):
    robot.add_ani_frame(i*dt, q=q0+np.array([0.3*sin(i*dt),0,0.095+0.075*cos(i*dt)]).reshape((3,1)))
    htm2 = robot.fkm(axis='dh')[2]
    joint_0.add_ani_frame(i*dt, htm =  Utils.trn([0,0,0.2]) * Utils.rotz(0.3*sin(i*dt)) )
    joint_1.add_ani_frame(i*dt, Utils.trn([0,0,0.1]) * robot.fkm(axis='dh')[0] )
    joint_2.add_ani_frame(i*dt, htm =  htm2*Utils.trn([0,0,-0.2]) )
    
    joint_cylinder_1.add_ani_frame(i*dt, htm =  Utils.trn([0,0,0.1]) * robot.fkm(axis='dh')[0] )
    joint_cylinder_2.add_ani_frame(i*dt, htm =  htm2*Utils.trn([0,0,-0.2]) )
    message = "<div style=\'item-align:center;text-align:center;width:500px\'><div style=\'width:100%;display: inline-block;float:left;vertical-align: middle\'>q(" + str(
        round(100 * i * dt) / 100) + "s) = "+str(round(robot.q[0,0],2))+" rad, "+str(round(robot.q[1,0],2))+" rad, "+str(round(robot.q[2,0],2))+" m</div></div>"
    
    explanation.add_ani_frame(i*dt, html_text=message)


sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/","part_1_4")