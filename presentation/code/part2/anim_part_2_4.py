import uaibot as ub
from scipy.linalg import null_space
import numpy as np
import requests
from io import BytesIO
# import trimesh

mesh_material = ub.MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#606060",opacity=1)


bunny_model = ub.Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/Other/bunny.obj', mesh_material = mesh_material)

bunny = ub.RigidObject([bunny_model],htm=ub.Utils.rotx(np.pi/2))

light1 =  ub.PointLight(name="light1", color="white", intensity=8, htm=ub.Utils.trn([-1, -1, 1.5]))
light2 =  ub.PointLight(name="light2", color="white", intensity=8, htm=ub.Utils.trn([-1, 1, 1.5]))
light3 =  ub.PointLight(name="light3", color="white", intensity=8, htm=ub.Utils.trn([1, -1, 1.5]))
light4 =  ub.PointLight(name="light4", color="white", intensity=8, htm=ub.Utils.trn([1, 1, 1.5]))


csp = [1.0,-1.0,1.0,-0.1,0.1,0.0,6.0]
sim = ub.Simulation([light1,light2,light3,light4,bunny])
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_start_pose = csp)




sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/","part_2_4")