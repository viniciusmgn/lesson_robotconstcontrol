import uaibot as ub
import numpy as np
import re

robot = ub.Robot.create_franka_emika_3(eef_frame_visible=False)

prob = ub.Utils.get_fishbotics_mp_problems()

prob_names = {re.match(r'^(.*)_\d+$', key).group(1) for key in prob.keys() if re.match(r'^(.*)_\d+$', key)}


print(sorted(prob_names))

sim = ub.Simulation([robot], background_color='#191919')

obs = []

name = 'dresser_task_oriented_1'

for obj in prob[name]['all_obs']:
    obj._mesh_material = ub.MeshMaterial.create_wood()
    obs.append(obj)
    

sim.add(obs[:-1])
sim.set_parameters(show_grid=False, show_world_frame=False)
robot.add_ani_frame(0,q=prob[name]['q0'],htm=prob[name]['htm_base'])

sim.run()

#sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/intro/","intro_1")