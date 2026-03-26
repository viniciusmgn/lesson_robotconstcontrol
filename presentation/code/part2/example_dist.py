import uaibot as ub
import numpy as np
import random


robot = ub.Robot.create_franka_emika_3(htm=ub.Utils.trn([0,0,0]),eef_frame_visible=False)




box1 = ub.Box(width=0.7, depth=1.5, height=0.1, htm=ub.Utils.trn([0.8,0,0.5]),color='magenta')
box2 = ub.Box(width=0.1, depth=0.1, height=0.45, htm=ub.Utils.trn([0.5,-0.6,0.45/2]),color='magenta')
box3 = ub.Box(width=0.1, depth=0.1, height=0.45, htm=ub.Utils.trn([0.5, 0.6,0.45/2]),color='magenta')
box4 = ub.Box(width=0.1, depth=0.1, height=0.45, htm=ub.Utils.trn([1.1, 0.6,0.45/2]),color='magenta')
box5 = ub.Box(width=0.1, depth=0.1, height=0.45, htm=ub.Utils.trn([1.1, -0.6,0.45/2]),color='magenta')

cyl1 = ub.Cylinder(radius=0.15, height=0.4, htm=ub.Utils.trn([0.8,-0.3,0.2+0.5]),color='yellow')
cyl2 = ub.Cylinder(radius=0.1, height=0.6, htm=ub.Utils.trn([0.9,0.4,0.3+0.5]),color='yellow')

wpA = ub.Ball(radius=0.025,color='cyan')
wpB = ub.Ball(radius=0.025,color='cyan')

obstacles=[box1, cyl1, cyl2]
sim = ub.Simulation([robot,box1, box2, box3, box4, box5, cyl1, cyl2, wpA, wpB])

sim.set_parameters(background_color='#191919', show_world_frame=False, show_grid=False)


index_found=[]
witness_A=[]
witness_B=[]
drange = [[0.35,0.4], [0.2,0.22], [0.04,0.09]]
max_dist = 0
for i in range(3):
    
    cont=True
    found_index = []
    
    while cont:
        q = np.matrix(np.zeros((7,1)))
        for j in range(7):
            q[j,0] = robot._joint_limit[j,0]+(robot._joint_limit[j,1]-robot._joint_limit[j,0])*random.random()
            
        dmin = 1000
        for k in range(len(obstacles)):
            dist = robot.compute_dist(q=q,obj=obstacles[k]).get_closest_item()
            if dist.distance<dmin:
                dmin = dist.distance
                index = k
                wa = dist.point_link
                wb = dist.point_object
        print("t = "+str(i)+","+str(round(dmin,2))+", fm "+str(round(max_dist,2)))  
        
        max_dist = max(max_dist, dmin)
        cont =  index in index_found or not ( dmin>drange[i][0] and dmin<drange[i][1] )
        
    index_found.append(index)
        
    print(index)
    print(dmin)
    robot.add_ani_frame(i,q)
    wpA.add_ani_frame(i,ub.Utils.trn(wa))
    wpB.add_ani_frame(i,ub.Utils.trn(wb))
    

sim.run()