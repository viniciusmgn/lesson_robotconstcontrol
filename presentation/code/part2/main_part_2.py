import uaibot as ub
from scipy.linalg import null_space
import numpy as np
import requests
from io import BytesIO

#Animação da junta revolta


#Animação da junta revolta


sim = ub.Simulation([], load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False)

url = "https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/PointCloud/data_wall_with_hole.npy"
wallpoints = np.load(BytesIO(requests.get(url).content))
pc = ub.PointCloud(points = wallpoints, size=0.03, color='cyan')

box = ub.Box(htm = ub.Utils.trn([-0.5,0.5,0.3]), color='yellow', width=0.3,depth=0.2,height=0.6)
cylinder = ub.Cylinder(htm = ub.Utils.trn([-0.5,0.5,0.7]), color='green', radius=0.05,height=0.2)

light1 =  ub.PointLight(name="light1", color="white", intensity=8, htm=ub.Utils.trn([-1, -1, 1.5]))
light2 =  ub.PointLight(name="light2", color="white", intensity=8, htm=ub.Utils.trn([-1, 1, 1.5]))
light3 =  ub.PointLight(name="light3", color="white", intensity=8, htm=ub.Utils.trn([1, -1, 1.5]))
light4 =  ub.PointLight(name="light4", color="white", intensity=8, htm=ub.Utils.trn([1, 1, 1.5]))

angle = (np.pi/180)*20
d = 0.3
z = np.sin(angle)

A = np.matrix([[1,0,z],[-1,0,z],[0,1,z],[0,-1,z],[0,0,-1]])
b = np.matrix([0.2,0.2,0.2,0.2,0]).transpose()

material = ub.MeshMaterial(color='red', opacity=1.0, metalness=1, roughness=1.0, side="DoubleSide")

pol = ub.ConvexPolytope(A=A,b=b,mesh_material = material)

sim.add([light1, light2, light3, light4, pc, box, cylinder,pol])

sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/","test")