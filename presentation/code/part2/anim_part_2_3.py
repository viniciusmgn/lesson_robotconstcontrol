import uaibot as ub
from scipy.linalg import null_space
import numpy as np
import requests
from io import BytesIO
# import trimesh



cat_data = np.load("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/code/part2/cat.npy")

csp = [1.0,-1.0,1.0,-0.25,0.25,0.0,2.0]
pc = ub.PointCloud(points = cat_data, size=0.005, color='cyan')
sim = ub.Simulation([pc])
sim.set_parameters(load_screen_color="#191919", background_color="#191919", width=500, height=500, show_world_frame=False, show_grid=False, camera_start_pose = csp)




sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/","part_2_3")