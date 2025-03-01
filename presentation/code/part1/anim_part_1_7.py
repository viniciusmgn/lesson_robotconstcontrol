from uaibot import *
import numpy as np


width=550
height=500

sim = Simulation([], load_screen_color="#191919", background_color="#191919", width=500, height=500,
                 camera_type="orthographic", show_grid = False, show_world_frame=False)

vector_w = Vector(color="orange",thickness=2)


texture_box = Texture(
    url='https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/Textures/gold_metal.png',
    wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[1, 1])

material_box = MeshMaterial(texture_map=texture_box, roughness=1, metalness=1, opacity=0.8)
box = Box(htm = np.identity(4), width=0.3, depth=0.2, height=0.1, mesh_material=material_box)

light1 = PointLight(name="light1", color="white", intensity=8, htm=Utils.trn([-1, -1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=8, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=8, htm=Utils.trn([1, -1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=8, htm=Utils.trn([1, 1, 1.5]))

sim.set_parameters(load_screen_color="#191919", width=500, height=500)

sim.add([vector_w, box,light1,light2,light3,light4])

dt = 0.01

for i in range(1000):
    t = i*dt
    omega = np.matrix([np.sin(0.03*t), np.sin(0.03*t), np.cos(0.03*t)])
    htm_new =  Utils.rot(omega,np.linalg.norm(omega)*dt)*box.htm
    box.add_ani_frame(t,htm_new)
    vector_w.add_ani_frame(t,origin=htm_new[0:3,-1],vector=0.5*omega)


sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/","part_1_7")