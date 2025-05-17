import numpy as np
import matplotlib.pyplot as plt
import uaibot as ub
import matplotlib.patches as patches


#We assume a path from a motion planner. Lets  assume that we got some waypoint
#paths q_path_wp. Then we interpolate using the function 'fun_path' to get 
#a function handle and sample n_sampled = 1000 points

n_sample = 1000
q_path_wp = [[-1.5, 0.2],[-1.5, 0.5],[-1.25,0.6],[-1.25,1.0], [-0.5,1.4],[0.5,0.6],[1.5,0]]
fun_path = ub.Utils.interpolate(q_path_wp, is_closed = False)
q_path = fun_path([i/n_sample for i in range(n_sample)])

#Parameters
kp = 0.5
eta = 0.4
dt = 0.01
tmax = 15
delta = 0.05
qF = np.matrix([1.5,0]).T
q0 = np.matrix([-1.5,0.2]).T
centers = [np.matrix([-0.5,0.7]).T, np.matrix([-0.5,-0.7]).T , np.matrix([0.0,0.0]).T]
radius = [0.5,0.5,0.5]

center_unpl = np.matrix([0.5,0.4]).T
radius_unpl = 0.3

#Start simulation
q = q0
hist_q = []

for i in range(round(tmax/dt)):


    hist_q.append(np.matrix(q))
    
    #Assembly the CBF controller
    H = 2*np.matrix([[1.0,0.0],[0.0,1.0]])
    psi, _, index = ub.Robot.vector_field(q,q_path, alpha=1.5, const_vel=0.5)
    
    #Since the curve is open, we need a function gamma(p) that vanishes
    #at the end point. At the end-point the closest index "index" is
    #index = n_sample
    
    gamma = max(min(10*(1-index/n_sample),1),0)
    f = -2*gamma*psi
    
    A = np.matrix(np.zeros((0,2)))
    b = np.matrix(np.zeros((0,1)))
    
    for k in range(len(centers)):
        B_k_fun_ = np.linalg.norm(q-centers[k])-radius[k]-delta
        grad_B_k_fun = (q-centers[k]).T/np.linalg.norm(q-centers[k])

        A = np.vstack( (A,  grad_B_k_fun) )
        b = np.vstack( (b,  -eta * B_k_fun_) )
      
    B_unp_fun = np.linalg.norm(q-center_unpl)-radius_unpl-delta
    grad_B_unp_fun = (q-center_unpl).T/np.linalg.norm(q-center_unpl) 
    A = np.vstack( (A,  grad_B_unp_fun) )
    b = np.vstack( (b,  -eta * B_unp_fun) )
        
    #Compute the control input
    u = ub.Utils.solve_qp(H,f,A,b)
    
    #Integrate the equation \dot{q} = u(q)
    q+= u*dt
 
#Plot image
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot([q[0, 0] for q in hist_q], [q[1, 0] for q in hist_q], color='#81d41a')
ax.plot([q[0, 0] for q in q_path], [q[1, 0] for q in q_path], color='#084594', \
        zorder = 20)
ax.scatter([qF[0,0]],[qF[1,0]], color='magenta',s=40)

for i in range(len(centers)):
    circle = patches.Circle((centers[i][0], centers[i][1]), radius=radius[i], \
    color='#5983b0')  
    ax.add_patch(circle)
    
circle = patches.Circle((center_unpl[0,0],center_unpl[1,0]), radius=radius_unpl, 
    color='#ec9ba4', zorder=15) 
ax.add_patch(circle)    
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 2.2)

plt.show()