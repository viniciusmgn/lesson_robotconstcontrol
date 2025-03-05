import uaibot as ub
from scipy.linalg import null_space
import numpy as np
import requests
from io import BytesIO

#Animação da junta revolta
def fill_box_with_spheres(width, depth, height, radius):
    centers = []
    
    radius_cor = radius/(np.sqrt(2))
    fat = 0.9
    
    
    
    nx = int(np.round(fat*width/(2*radius_cor)))
    ny = int(np.round(fat*depth/(2*radius_cor)))
    nz = int(np.round(fat*height/(2*radius_cor)))
    
    dx = width/(nx+1)
    dy = depth/(ny+1)
    dz = height/(nz+1)
    
    
    centers=[]
    x = -(nx-1)*dx/2
    for i in range(nx):
        y = -(ny-1)*dy/2
        for j in range(ny):
            z = -(nz-1)*dz/2
            for k in range(nz):
                centers.append([x,y,z])
                z+=dz
                   
            y+=dy
            
        x+=dx
    
    
    return centers

#Animação da junta revolta



csp = [1.0,-1.0,1.0,-0.5,0.5,0.0,2.0]

light1 =  ub.PointLight(name="light1", color="white", intensity=8, htm=ub.Utils.trn([-1, -1, 1.5]))
light2 =  ub.PointLight(name="light2", color="white", intensity=8, htm=ub.Utils.trn([-1, 1, 1.5]))
light3 =  ub.PointLight(name="light3", color="white", intensity=8, htm=ub.Utils.trn([1, -1, 1.5]))
light4 =  ub.PointLight(name="light4", color="white", intensity=8, htm=ub.Utils.trn([1, 1, 1.5]))

sim = ub.Simulation([light1,light2,light3,light4])
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_start_pose = csp)

center = fill_box_with_spheres(1, 2, 3, 0.5)

robot = ub.Robot.create_magician_e6(eef_frame_visible=False)

q = np.matrix(robot.q)
q[-2]+=np.pi/2
robot.add_ani_frame(0,q)

sim.add([robot])

htm0 = robot.fkm(axis='dh')

i=0
for link in robot.links:
    for co in link.col_objects:
        obj = co[0]
        
        if ub.Utils.get_uaibot_type(obj)=='uaibot.Box':
            radius = np.sqrt(2)*min([obj.width,obj.depth,obj.height])/2
            centers = fill_box_with_spheres(obj.width,obj.depth,obj.height, radius)

            for c in centers:
                ball = ub.Ball(color=obj.color,radius=radius,opacity=0.7, htm=htm0[i]*co[1]*ub.Utils.trn(c))
                robot.attach_object(ball)
                sim.add(ball)
            
        if ub.Utils.get_uaibot_type(obj)=='uaibot.Cylinder': 
            radius = min([2*obj.radius,obj.height])/2
            centers = fill_box_with_spheres(2*obj.radius,2*obj.radius,1.2*obj.height, radius)

            for c in centers:
                ball = ub.Ball(color=obj.color,radius=1.1*radius,opacity=0.7, htm=htm0[i]*co[1]*ub.Utils.trn(c))
                robot.attach_object(ball)
                sim.add(ball)
            
        if ub.Utils.get_uaibot_type(obj)=='uaibot.Ball':  
            a = 0  
    i+=1                    

#robot.update_col_object(0)
  
# dt=0.01
# htm_tg = ub.Utils.trn([0.6,0.2,-0.3])*robot.fkm()
# for i in range(1000):      
#     r, Jr = robot.task_function(htm_des=htm_tg)
    
#     qdot = ub.Utils.dp_inv(Jr)*(-0.5*r)
    
#     if i<100:
#         qdot[3]=0
    
#     robot.add_ani_frame(i*dt,robot.q+qdot*dt)
#     robot.update_col_object(i*dt)




sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/","part_2_5")