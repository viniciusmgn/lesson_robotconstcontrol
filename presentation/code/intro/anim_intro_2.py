import uaibot as ub
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import quadprog



obj1 = ub.Box(htm = ub.Utils.trn([0,0,1])*ub.Utils.rotx(np.pi/2), color='red', width=0.3,depth=0.4,height=0.8)
obj2 = ub.Box(htm = ub.Utils.trn([0,0.3,2])*ub.Utils.rotx(np.pi/2-0.2), color='blue', width=0.3,depth=0.4,height=0.4)
projball1 = ub.Ball(color='cyan', radius=0.03)
projball2 = ub.Ball(color='magenta', radius=0.03)

light = ub.PointLight(name="light", color="white", intensity=20, htm=ub.Utils.trn([ 0,0,5]))

sim = ub.Simulation.create_sim_lesson([obj1, obj2, projball1, projball2, light])
sim.set_parameters(show_world_frame = False, show_grid = False, camera_start_pose = [-1.,0.,1.5,4.,0.,1.4,0.5])
sim.set_parameters(width=500,height=500,load_screen_color='#191919',background_color='#191919')

point1 = np.matrix([0.,-0.5,2.]).transpose()

projball1.add_ani_frame(0,ub.Utils.trn(point1))
projball2.add_ani_frame(0,ub.Utils.trn([100,100,100]))

t = 1.5

pcpoints=[]
indexrange=[0]
for i in range(10):
    
    if i%2 == 0:
        point2, _ = obj2.projection(point1)
        projball2.add_ani_frame(t,ub.Utils.trn(point2))
        projball1.add_ani_frame(t,ub.Utils.trn([100,100,100]))
        
        indexrange.append(10)
        for j in range(10):
            pcpoints.append(point1+(point2-point1)*(j/10))
    else:
        point1, _ = obj1.projection(point2)
        projball1.add_ani_frame(t,ub.Utils.trn(point1))
        projball2.add_ani_frame(t,ub.Utils.trn([100,100,100]))
        
        indexrange.append(10)
        for j in range(10):
            pcpoints.append(point2+(point1-point2)*(j/10))    
    
    t+=1.5
    
pc = ub.PointCloud(points=pcpoints,size=0.02,color='white')
sim.add(pc)

for i in range(10):
    pc.add_ani_frame(1.5*i,10*i,10*(i+1))

projball2.add_ani_frame(15,ub.Utils.trn(point2))
projball1.add_ani_frame(15,ub.Utils.trn(point1))
pc.add_ani_frame(15,0,0)

sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/intro/","intro_2")