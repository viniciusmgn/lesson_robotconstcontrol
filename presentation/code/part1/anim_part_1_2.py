from uaibot import *
from scipy.linalg import null_space
import numpy as np
from uaibot.simobjects import *
from uaibot.graphics import *
from uaibot.simulation import *
from uaibot.utils import *

#Animação da junta revolta


#Animação da junta revolta


sim = Simulation([], load_screen_color="#191919", background_color="#191919", width=550, height=500,
                 camera_type="orthographic")



light1 = PointLight(name="light1", color="white", intensity=1, htm=Utils.trn([-1, -1, 1.5]))
light2 = PointLight(name="light2", color="white", intensity=1, htm=Utils.trn([-1, 1, 1.5]))
light3 = PointLight(name="light3", color="white", intensity=1, htm=Utils.trn([1, -1, 1.5]))
light4 = PointLight(name="light4", color="white", intensity=1, htm=Utils.trn([1, 1, 1.5]))

style = "top:75%; right: 0; font-size: 2.5vw;  width:100%;position:absolute;text-align:center;background-color:#191919;color:white;font-smooth:always;font-family:arial"
explanation = HTMLDiv(html_text="", style=style)

frame = Frame(htm = np.identity(4), axis_color=["#FF7777", "#77FF77", "#7777FF"])

sim.add([frame,explanation,light1,light2,light3,light4])

H1 = Utils.trn([1,0,0])
H2 = Utils.rotx(3.14/2)
H3 = Utils.trn([0,0.5,0])
H4 = Utils.roty(-3.14/4)

dt=0.01
k=0

explanation.add_ani_frame(k*dt, html_text= "<b style=\'color:#34eb5e\'>R<sub>x</sub>(&#960/4)</b>D<sub>y</sub>(-0.25)R<sub>z</sub>(&#960/2)D<sub>x</sub>(1)")
for i in range(400):
    htm = frame.htm  * Utils.rotx((3.14/4)*(dt/4))
    frame.add_ani_frame(k*dt, htm = htm)
    k+=1

k+=400

explanation.add_ani_frame(k*dt, html_text= "<b style=\'color:#34eb5e\'>R<sub>x</sub>(&#960/4)D<sub>y</sub>(-0.25)</b>R<sub>z</sub>(&#960/2)D<sub>x</sub>(1)")
for i in range(400):
    htm = frame.htm * Utils.trn([0,-0.25 * dt/4,0])
    frame.add_ani_frame(k*dt, htm = htm)
    k+=1

k+=400

explanation.add_ani_frame(k*dt, html_text= "<b style=\'color:#34eb5e\'>R<sub>x</sub>(&#960/4)D<sub>y</sub>(-0.25)R<sub>z</sub>(&#960/2)</b>D<sub>x</sub>(1)")
for i in range(400):
    htm = frame.htm * Utils.rotz((3.14 / 2) * (dt / 4))
    frame.add_ani_frame(k*dt, htm = htm)
    k+=1

k+=400
explanation.add_ani_frame(k*dt, html_text= "<b style=\'color:#34eb5e\'>R<sub>x</sub>(&#960/4)D<sub>y</sub>(-0.25)R<sub>z</sub>(&#960/2)D<sub>x</sub>(1)</b>")
for i in range(400):
    htm = frame.htm * Utils.trn([dt/4,0,0])
    frame.add_ani_frame(k*dt, htm = htm)
    k+=1
    
sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/","part_1_2")