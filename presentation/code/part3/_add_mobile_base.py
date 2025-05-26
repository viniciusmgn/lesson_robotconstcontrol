import numpy as np
import os
from uaibot import ub





        
robot_a = ub.Robot.create_franka_emika_3()
robot_base = add_mobile_base(robot_a,[])

sim = ub.Simulation([robot_base])
for link in robot_base.links:
    for obj in link.col_objects:
        sim.add(obj[0])
        
for i in range(100):
    robot_base.add_ani_frame(0.01*i,q=[0,0,0.01*i,0,0,0,0,0,0,0])
    robot_base.update_col_object(0.01*i)
for i in range(100):
    robot_base.add_ani_frame(1+0.01*i,q=[0,0.01*i,0,0,0,0,0,0,0,0])
    robot_base.update_col_object(1+0.01*i)



sim.save()
        