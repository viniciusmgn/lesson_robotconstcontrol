
#%%

import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt
from helper_anim_part_3_ex3 import *

###################################   
#Parameters control

param_k = 1.0 
param_eta = 0.6 
param_eta_v = 0.3
param_v_max = 0.3 #0.8
param_a_max = 0.5
param_radius = 0.25
param_n_robots=9
param_dist_interm = 0.15
param_dist_final = 0.03

param_h_dist = 0.2
param_eps_dist = 0.01
param_delta_dist = 0.01


model_drone = ub.Model3D(
    'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/CrazyFlie/crazyflie.obj',
    scale=1.5+0*2.0, mesh_material=ub.MeshMaterial.create_rough_metal())

drones= []
drone_colors = ['red','green','blue','yellow','magenta','cyan','gray','brown','DarkSlateGray']


#Create the drones
    
for i in range(param_n_robots):
    drone_body = ub.RigidObject(list_model_3d=[model_drone],htm=ub.Utils.trn([0,0,0])*ub.Utils.rotx(np.pi/2))
    drone_ball = ub.Ball(color=drone_colors[i], radius=0.2, opacity=0.2)
    new_drone = ub.Group(list_of_objects=[drone_body, drone_ball])
    
    
    drones.append(new_drone)
    
#Create the obstacles

obs1 = ub.Box(htm=ub.Utils.trn([0,0,0.5]),width=1.0,depth=0.2,height=1.0)
obs2 = ub.Box(htm=ub.Utils.trn([0.6,0.1,0.5]),width=0.2,depth=1.0,height=1.0)
obs3 = ub.Box(htm=ub.Utils.trn([-1.0,-1.0,0.3]),width=0.4,depth=0.4,height=0.6)
obs4 = ub.Box(htm=ub.Utils.trn([-1.0,1.0,0.2]),width=0.4,depth=0.4,height=0.4)
obs5 = ub.Box(htm=ub.Utils.trn([0.6,-0.9,0.95]),width=0.2,depth=1.0,height=0.1)
obs6 = ub.Box(htm=ub.Utils.trn([0.6,-1.4,0.5]),width=0.2,depth=0.2,height=1.0)

wallxp = ub.Box(htm=ub.Utils.trn([-2.0,0.,0.5]),width=0.05,depth=4.0,height=1.0)
wallxn = ub.Box(htm=ub.Utils.trn([2.0,0.,0.5]),width=0.05,depth=4.0,height=1.0)
wallyp = ub.Box(htm=ub.Utils.trn([0,2.,0.5]),width=4.0,depth=0.05,height=1.0)
wallyn = ub.Box(htm=ub.Utils.trn([0,-2.,0.5]),width=4.0,depth=0.05,height=1.0)

all_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, wallxp, wallxn, wallyp, wallyn]

all_points = []

for obs in all_obstacles:
    all_points+=  [np.matrix(p).T for p in obs.to_point_cloud(disc=0.06).points.T]

pc = ub.PointCloud(points=all_points,size=0.03,color='white')



#Sample initial poses for the drones

start_points = [np.matrix([0,0,0]).T for i in range(param_n_robots)]

for i in range(param_n_robots):
    
    cont = True
       
    while cont:
    
        htm=ub.Utils.htm_rand([-1.5,-1.5,0.5],[1.5,1.5,1.0],0)
        drones[i].add_ani_frame(0,htm)
        start_points[i] = htm[0:3,-1]
        
        #Check if this is ok
        ball_drone_i = drones[i].list_of_objects[-1]
        
        
        collided = ball_drone_i.compute_dist(pc)[2]<param_radius
        
        j = 0
        while (not collided) and (j<i) :

            
            ball_drone_j = drones[j].list_of_objects[-1]
            collided = ball_drone_i.compute_dist(ball_drone_j)[2] < param_radius
            j+=1
         
  
        cont = collided
            
#Sample target points that are far apart from each other, not coliding
#and far for the respective drone

all_tg_points = []
all_tg_box = []

for i in range(param_n_robots):
    
    cont = True
       
    print("Placed "+str(i)) 
    while cont:
    
        tg_point_try=ub.Utils.htm_rand([-2.0,-2.0,0.5],[1.5,1.5,1.0],0)[0:3,-1]
        
        
        
        #Check if this is ok
        ball_drone_i = drones[i].list_of_objects[-1]
        
        
        unfit = pc.projection(tg_point_try)[1]<0.3 or ball_drone_i.projection(tg_point_try)[1]<1.5 
        
        #Check if far away from others points
        j = 0
        while (not unfit) and (j<i) :
            unfit = np.linalg.norm(all_tg_points[j]-tg_point_try)<1.0
            j+=1
         
  
        cont = unfit
        
    all_tg_points.append(tg_point_try)
    all_tg_box.append(ub.Box(htm=ub.Utils.trn(tg_point_try), color=drone_colors[i],width=0.03,depth=0.03,height=0.03))
                    
     
#Use motion planer

path_points = []

for i in range(param_n_robots):
    path = random_path_planner(start_points[i], all_tg_points[i], pc, param_radius, [-2., 2., -2. , 2., 0.5, 1.0])
    path_points.append(path)

           
        
sim = ub.Simulation.create_sim_hill(drones, light_intensity=1.5)
sim.add(pc)
sim.add(all_tg_box)
sim.set_parameters(pixel_ratio=0.9)



#################################################




def dfun(_q, _i, _j):
    
    qi = _q[3*_i:3*(_i+1),:]
    qj = _q[3*_j:3*(_j+1),:]
    
    return np.linalg.norm(qi-qj)-(2*param_radius)

def jac_dfun(_q, _i, _j):
    
    n = param_n_robots
    jac_dist = np.zeros((1,3*n))
    
    qi = _q[3*_i:3*(_i+1),:]
    qj = _q[3*_j:3*(_j+1),:]
    
    norm_ij = np.linalg.norm(qi-qj)
    
    jac_dist[0,3*_i:3*(_i+1)] = (qi-qj).T/norm_ij
    jac_dist[0,3*_j:3*(_j+1)] = (qj-qi).T/norm_ij
    
    return jac_dist

# def dofun(_q, _i, _pc):
#     qi = _q[3*_i:3*(_i+1),:]
#     return _pc.projection(qi)[1]-param_radius-0.02

# def jac_dofun(_q, _i, _pc):
#     qi = _q[3*_i:3*(_i+1),:]
#     qis = _pc.projection(qi)[0]
    
#     n = param_n_robots
#     jac_dist = np.zeros((1,3*n))
#     jac_dist[0,3*_i:3*(_i+1)] = (qi-qis).T/(1e-5+np.linalg.norm(qi-qis))

#     return jac_dist

def dofun(_q, _i, _pc):
    qi = _q[3*_i:3*(_i+1),:]
    
    return ub.Ball(htm=ub.Utils.trn(qi), radius=param_radius).compute_dist(obj = _pc, h=param_h_dist, eps=param_eps_dist)[2]-param_delta_dist

def jac_dofun(_q, _i, _pc):
    
    qi = _q[3*_i:3*(_i+1),:]
    pball, ppc, dist, _ = ub.Ball(htm=ub.Utils.trn(qi), radius=param_radius).compute_dist(obj = _pc, h=param_h_dist, eps=param_eps_dist)
    

    n = param_n_robots
    jac_dist = np.zeros((1,3*n))
    jac_dist[0,3*_i:3*(_i+1)] = (pball-ppc).T/(1e-5+dist+0.03)

    return jac_dist


############################
        
def control_fun(_q, _dotq, _all_tg, _pc):
    
    n = param_n_robots
    
    #Create the objective function
    H = 2*np.identity(3*n)
    f = np.zeros((3*n,1))
    
    error = 0
    
    for i in range(n):
        qi = _q[3*i:3*(i+1),:]
        dotqi = _dotq[3*i:3*(i+1),:]
        f[3*i:3*i+3,:] = ( 2*param_k*dotqi+ (param_k*param_k)*(qi-_all_tg[i]) )
        error+= np.linalg.norm(qi-_all_tg[i])
        
    f = 2*f
    
    
    #Create the constraints
    A = np.matrix(np.zeros((0,3*n)))
    b = np.matrix(np.zeros((0,1)))
    

    #Create the inter-agent collision constraints
    for i in range(n):
        for j in range(0,i):
            
            
            dist = dfun(_q, i, j)
            jac_dist = jac_dfun(_q, i, j)
            dist_dot = (dfun(_q+dt*_dotq, i, j)-dfun(_q-dt*_dotq, i, j))/(2*dt)
            dist_hess = ( (jac_dfun(_q+dt*_dotq, i, j)-jac_dfun(_q-dt*_dotq, i, j))/(2*dt))*_dotq
            

            A = np.vstack((A, jac_dist))
            b = np.vstack((b, -2*param_eta*dist_dot - (param_eta*param_eta)*dist-dist_hess ) )
            
    
    hh = []
      
    for i in range(n):
            dist = dofun(_q, i, _pc)
            jac_dist = jac_dofun(_q, i, _pc)
            dist_dot = (dofun(_q+dt*_dotq, i, _pc)-dofun(_q-dt*_dotq, i, _pc))/(2*dt)
            dist_hess = ( (jac_dofun(_q+dt*_dotq, i, _pc)-jac_dofun(_q-dt*_dotq, i, _pc))/(2*dt))*_dotq
            
            dist_hess = min(dist_hess, 0)
            
            hh.append(dist_hess)
            
            if dist_hess < 0 and abs(dist_hess)>5.0:
                print("Negative dhess = "+str(dist_hess))
            if dist<-0.05:
                print("Error!")
                a = a[0]

            
            A = np.vstack((A, jac_dist))
            b = np.vstack((b, -2*param_eta*dist_dot - (param_eta*param_eta)*dist-dist_hess ) )


    idm = np.identity(3*n)
    onev = np.ones((3*n,1))
                
    A =   np.vstack((A, idm, -idm)) 
    b =   np.vstack((b, -param_a_max*onev, -param_a_max*onev)) 
           
    A =   np.vstack((A, idm, -idm)) 
    b =   np.vstack((b, -2*param_eta_v*(_dotq+param_v_max*onev), -2*param_eta_v*(param_v_max*onev-_dotq) ))
    
    
    for i in range(n):
          jac_dist = np.zeros((1,3*n))
          jac_dist[0,3*i+2] = -1.0
          
          A =   np.vstack((A, jac_dist))
          b =   np.vstack((b, -2*param_eta*(-_dotq[3*i+2,-1])-(param_eta*param_eta)*( 1.0 - _q[3*i+2,-1] ) )) 
          
          jac_dist[0,3*i+2] = 1.0
          
          A =   np.vstack((A, jac_dist))
          b =   np.vstack((b, -2*param_eta*(dotq[3*i+2,-1])-(param_eta*param_eta)*(_q[3*i+2,-1]-param_radius) )) 
      
    try:                 
        return ub.Utils.solve_qp(H,f,A,b), hh
    except:
        return -0.5*_dotq, hh
            
    
 
######################   

q = np.matrix(np.zeros((3*param_n_robots,1)))
for i in range(param_n_robots):
    q[3*i:3*(i+1),:] = start_points[i]
    
dotq = np.matrix(0*q)


dt = 0.02

hist_dotq = []
hist_ddotq = []
hist_t = []
hist_hh = []

init_index = [0 for i in range(param_n_robots)]
current_tg = [path_points[i][0] for i in range(param_n_robots)]
finished = [False for i in range(param_n_robots)]

for i in range(param_n_robots):
    all_tg_box[i].add_ani_frame(0,htm=ub.Utils.trn(current_tg[i]))

cont = True 
while cont:
    
    ddotq, hh = control_fun(q, dotq, current_tg, pc)
    
    print("Step "+str(i))
    
    total_finished = True
    for j in range(param_n_robots):
        qj = q[3*j:3*(j+1),:]
        error = np.linalg.norm(qj-current_tg[j])
        
        
        no_targets = len(path_points[j])
        
        if finished[j]:
            print("Robot "+str(j+1)+", FINISHED!, error  = "+str(round(error,2)))
        else:
            print("Robot "+str(j+1)+", ind = "+str(init_index[j]+1)+"/"+str(no_targets)+", error = "+str(round(error,2)))
            
        
        
        if error <= (param_dist_interm if init_index[j] < no_targets-1 else param_dist_final):
            init_index[j]+=1
            if init_index[j] == no_targets:
                init_index[j] = no_targets-1
                finished[j] = True
                
            current_tg[j] = path_points[j][init_index[j]]
            all_tg_box[j].add_ani_frame(i*dt,htm=ub.Utils.trn(current_tg[j]))
            
        total_finished = total_finished and finished[j]
    
    
    
    cont = i < 2500 and not total_finished
    i = i + 1
    
    
    q += dotq*dt
    dotq += ddotq*dt
    
    hist_dotq.append(np.matrix(dotq))
    hist_ddotq.append(ddotq)
    hist_t.append(i*dt)
    hist_hh.append(hh)
    
    for j in range(param_n_robots):
        drones[j].add_ani_frame(time=i*dt,htm=ub.Utils.trn(q[3*j:3*(j+1),:]))

sim.save()
# %%

plt.figure()

n = param_n_robots

T = len(hist_dotq)

fig, axes = plt.subplots(int(np.ceil(n/2)), 2, figsize=(8, 2.5 * n), sharex=True)
axes = axes.flatten()

if n == 1:
    axes = [axes]

for i in range(n):
    axes[i].plot(hist_t, [u[3*i,0] for u in hist_dotq], color='red')
    axes[i].plot(hist_t, [u[3*i+1,0] for u in hist_dotq], color='green')
    axes[i].plot(hist_t, [u[3*i+2,0] for u in hist_dotq], color='blue')
    axes[i].plot(hist_t, [param_v_max for t in hist_t], color='magenta')
    axes[i].plot(hist_t, [-param_v_max for t in hist_t], color='magenta')
    
    
    axes[i].set_ylabel(f'Group {i}')
    axes[i].legend([f'{3*i}', f'{3*i+1}', f'{3*i+2}'])
    axes[i].grid(True)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.show()


plt.figure()

for i in range(param_n_robots):
    plt.plot(hist_t, [np.linalg.norm(u[3*i:3*(i+1),:]) for u in hist_dotq])
    
plt.show()

plt.figure()

for i in range(param_n_robots):
    plt.plot(hist_t, [np.linalg.norm(u[i]) for u in hist_hh])
    
plt.show()

plt.figure()

for i in range(param_n_robots):
    plt.plot(hist_t, [np.linalg.norm(u[3*i:3*(i+1),:]) for u in hist_ddotq])
    
plt.show()
