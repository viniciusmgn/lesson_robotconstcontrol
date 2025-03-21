from uaibot import *
import numpy as np


width=550
height=500

robot = Robot.create_kuka_kr5(opacity=0.7, color="gray", eef_frame_visible=False)
light0 = PointLight(name="light0", color="white", intensity=40, htm=Utils.trn([ 0,0,5.]))
light1 = PointLight(name="light1", color="white", intensity=40, htm=Utils.trn([ 5.,0,0]))
light2 = PointLight(name="light2", color="white", intensity=40, htm=Utils.trn([ -5.,0,0]))
light3 = PointLight(name="light3", color="white", intensity=40, htm=Utils.trn([ 0,5.,0]))
light4 = PointLight(name="light4", color="white", intensity=40, htm=Utils.trn([ 0,-5.,0]))

sim = Simulation([robot, light0, light1, light2, light3, light4], load_screen_color="#191919", background_color="#191919", camera_type="orthographic",
                 show_world_frame=False, width=width, height=height)

# Create all objects that will be used during the simulation

far = Utils.trn([1000, 1000, 1000])


def clone_link_as_rigid_body(name, link):
    list_model_3d = []
    for model3d in link.list_model_3d:
        list_model_3d.append(
            Model3D(model3d.url, model3d.scale, model3d.htm, MeshMaterial(color="gold", opacity=0.8)))

    return RigidObject(list_model_3d, name=name, htm=far)


links_alternative = []
for i in range(len(robot.links)):
    links_alternative.append(clone_link_as_rigid_body("rigid_object_link_" + str(i), robot.links[i]))
    sim.add(links_alternative[-1])

htm_dh = robot.fkm(axis="dh")

htm = [robot.htm_base_0]
for i in range(len(robot.links)):
    htm.append(htm_dh[i])

x_axis = []
y_axis = []
z_axis = []

vector_length = 0.12

lines_purple = Cylinder(name="linePurple", htm=far, radius=0.002, height=2, color="magenta")
lines_orange = Cylinder(name="lineOrange", htm=far, radius=0.002, height=2, color="orange")
points_A = Ball(name="pointA", htm=far, radius=0.015, color="cyan")
points_B = Ball(name="pointB", htm=far, radius=0.015, color="green")


line_between_points = []
start_lines = []
end_lines = []
dist_between_points = 0.01

for i in range(len(robot.links)):
    p_A = (htm[i] * Utils.trn([0, 0, robot.links[i].d]))[0:3, 3]
    p_B = htm[i + 1][0:3, 3]

    n_points = floor(np.linalg.norm(p_A - p_B) / dist_between_points)
    start_lines.append(len(line_between_points))

    for j in range(n_points):
        line_between_points.append((1 - j / n_points) * p_A + (j / n_points) * p_B)

    end_lines.append(len(line_between_points))

n = len(robot.links)
line_between_points = np.array(line_between_points).transpose()
if np.shape(line_between_points)[0]==0:
    line_between_points = np.zeros((3,0))

line_between_points_pc = PointCloud(name="lineBetweenPoints", points=line_between_points[0], size=2, color="orange")
line_between_points_pc.add_ani_frame(0, 0, 0)

sim.add(lines_purple)
sim.add(lines_orange)
sim.add(points_A)
sim.add(points_B)
sim.add(line_between_points_pc)

for i in range(len(robot.links) + 1):
    x_axis.append(Cylinder(name="xAxis" + str(i), color="red", htm=far, radius=0.003, height=vector_length))
    y_axis.append(Cylinder(name="yAxis" + str(i), color="limegreen", htm=far, radius=0.003, height=vector_length))
    z_axis.append(Cylinder(name="zAxis" + str(i), color="blue", htm=far, radius=0.003, height=vector_length))

    sim.add(x_axis[-1])
    sim.add(y_axis[-1])
    sim.add(z_axis[-1])

style = "top:" + str(0.75 * height) + "px;right:" + str(0) + "px;width:" + str(
    width) + "px;position:absolute;text-align:center;color:white;background-color:#222224;font-smooth:always;font-family:arial"

explanation = HTMLDiv(html_text="", style=style)

sim.add(explanation)


def txt_link(i):
    return "<b>link<sub>" + str(i) + "</sub></b>"


def txt_frame(i):
    return "<b>F<sub>DH" + str(i) + "</sub></b>"


def txt_joint(i):
    return "<b>joint<sub>" + str(i) + "</sub></b>"


def txt_line_purple(i):
    return "<span style='color:#FF7AFF'><b>line<sub>" + str(i) + "</sub></b></span>"


def txt_line_orange(i):
    return "<span style='color:#FFA500'><b>line<sub>" + str(i) + "</sub></b></span>"


def txt_pointA(i):
    return "<span style='color:cyan'><b>point<sub>" + str(i) + "," + str(i + 1) + "</sub></b></span>"


def txt_pointB(i):
    return "<span style='color:#7bb572'><b>point<sub>" + str(i) + "," + str(i - 1) + "</sub></b></span>"


def txt_axis_x(i):
    return "<span style='color:red'><b>x<sub>DH" + str(i) + "</sub></b></span>"


def txt_axis_y(i):
    return "<span style='color:limegreen'><b>y<sub>DH" + str(i) + "</sub></b></span>"


def txt_axis_z(i):
    return "<span style='color:blue'><b>z<sub>DH" + str(i) + "</sub></b></span>"


# Make the animation
k = 0
dt = 0.01
deltak = 200

# Animate the creation of all DH frames
for i in range(len(robot.links) + 1):

    explanation.add_ani_frame(k * dt, html_text="Lets create frame " + txt_frame(i) + "...")

    if i > 0:
        k += 3 * deltak
        explanation.add_ani_frame(k * dt, html_text="Lets draw " + txt_line_orange(
            i) + ". This is the line aligned with joint " + txt_joint(i) + "</b>.")
        # Show the line_(i-1)
        lines_orange.add_ani_frame(k * dt, htm=htm[i - 1])

    # Show the line_i
    k += 3 * deltak
    if i < len(robot.links):
        explanation.add_ani_frame(k * dt, html_text="Lets draw " + txt_line_purple(
            i + 1) + ". This is the line aligned with joint  " + txt_joint(i + 1) + "</b>.")
    else:
        explanation.add_ani_frame(k * dt, html_text="There is no " + txt_line_purple(
            i + 1) + ", because there isnt " + txt_joint(i + 1) + ". So, lets choose " + txt_line_purple(
            i + 1) + " arbitrarily. Lets choose aligned with the previous line.</span>...")

    lines_purple.add_ani_frame(k * dt, htm=htm[i])

    # Show the pointB_i and pointA_i
    k += 3 * deltak
    text = ""
    if i > 0:

        text += "Lets draw " + txt_pointA(i) + ". This in the point in " + txt_line_orange(
            i) + " which is the closest to " + txt_line_purple(i + 1) + ".<br>"

        points_A.add_ani_frame(k * dt, htm=htm[i - 1] * Utils.trn([0, 0, robot.links[i - 1].d]))

        text += "Lets draw  " + txt_pointB(i + 1) + ".  This in the point in  " + txt_line_purple(
            i + 1) + " which is the closest   " \
                     "to " + txt_line_orange(i) + ".<br>"

        if i < len(robot.links) and abs(1 - abs(htm[i - 1][0:3, 2].transpose() * htm[i][0:3, 2])) < 0.001:
            text += "Since the lines " + txt_line_orange(
                i) + " and " + txt_line_purple(i + 1) + " are parallel, there isnt infinite pairs of " + txt_pointA(
                i) + " and " + txt_pointB(i + 1) + " to choose. Lets choose it arbitrarily.<br>"

    else:
        text += "For the first frame " + txt_pointB(
            i + 1) + " can be chosen arbitrarily as long it is in " + txt_line_purple(i + 1) + ".<br>"

    text += "The center of " + txt_frame(i) + " is " + txt_pointB(i + 1) + "."
    explanation.add_ani_frame(k * dt, html_text=text)

    points_B.add_ani_frame(k * dt, htm=htm[i])

    # Create the vector z_i
    k += 4 * deltak
    text = "The axis " + txt_axis_z(i) + " has to be aligned with " + txt_line_purple(i + 1) + ".<br>"
    if i < len(robot.links):
        text += "We can choose between the two directions arbitrarily (we are defining the 'positive' sense   " \
                "of " + (
                    "rotation" if robot.links[i].joint_type == 0 else "translation") + " of the axis)."
    else:
        text += "For the last axis, we can keep the same z as before."

    explanation.add_ani_frame(k * dt, html_text=text)
    z_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.trn([0, 0, vector_length / 2]))

    # Create the vector x_i
    k += 3 * deltak
    if i > 0:
        if start_lines[i - 1] == end_lines[i - 1]:
            explanation.add_ani_frame(k * dt, html_text="In this case, " + txt_pointA(i) + " and " + txt_pointB(
                i + 1) + " are equal. So, we can choose the axis " + txt_axis_x(
                i) + " arbitrarily as long as it is orthogonal to " + txt_line_orange(
                i) + " and " + txt_line_purple(i + 1) + ".")
        else:
            explanation.add_ani_frame(k * dt, html_text="The axis " + txt_axis_x(i) + " needs to go from " + txt_pointA(
                i) + " to " + txt_pointB(i + 1) + ".")

            for j in range(start_lines[i - 1], end_lines[i - 1]):
                k += round(0.05 * deltak)
                line_between_points_pc.add_ani_frame(k * dt, start_lines[i - 1], j)
    else:
        k += 3 * deltak
        explanation.add_ani_frame(k * dt,
                                  html_text="In the first frame, " + txt_frame(
                                      0) + ", we can choose the axis " + txt_axis_x(
                                      0) + " arbitrarily, as long as it is orthogonal to " \
                                           "axis " + txt_axis_z(0) + " .")

    k += deltak
    x_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.roty(3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
    line_between_points_pc.add_ani_frame(k * dt, 0, 0)

    # Create the vector y_i
    k += 3 * deltak
    explanation.add_ani_frame(k * dt,
                              html_text="The axis " + txt_axis_y(i) + " is uniquely determined by axis " + txt_axis_x(
                                  i) + " and " + txt_axis_z(i) + " from the righthand rule.")
    y_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.rotx(-3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))

    # Erase the lines and points
    k += 3 * deltak
    points_A.add_ani_frame(k * dt, htm=far)
    points_B.add_ani_frame(k * dt, htm=far)
    lines_orange.add_ani_frame(k * dt, htm=far)
    lines_purple.add_ani_frame(k * dt, htm=far)
    k += deltak

    # Flash the link object
    if i > 0:
        explanation.add_ani_frame(k * dt, html_text="The frame, " + txt_frame(
            i) + " is attached to the link and blinking in <span style='color:gold'><b>yellow</b></span>.<br>So" \
                 ", any joint that moves this link moves the frame.")
        for j in range(10):
            k += floor(0.1 * deltak)
            links_alternative[i - 1].add_ani_frame(k * dt, robot.fkm(axis="dh")[i - 1])
            k += floor(0.1 * deltak)
            links_alternative[i - 1].add_ani_frame(k * dt, far)
    else:
        explanation.add_ani_frame(k * dt,
                                  html_text="The first frame " + txt_frame(
                                      0) + ", is not attached to any link, it is fixed.")
        k += 2 * deltak

    # Move the robot from joint 0 to i (not included)
    for j in range(1, i + 1):
        q = robot.q0

        explanation.add_ani_frame(k * dt,
                                  html_text="When the joint " + txt_joint(j) + " moves, it moves " + txt_frame(i) + ".")

        for l in range(floor(0.5 * deltak)):
            q[j - 1] += (robot.joint_limit[j - 1,0] - robot.q0[j-1])/ (floor(0.5 * deltak) - 1)
            htm_i = robot.fkm(axis="dh", q=q)[i - 1]
            robot.add_ani_frame(k * dt, q=q)
            x_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.roty(3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            y_axis[i].add_ani_frame(k * dt,
                                    htm=htm_i * Utils.rotx(-3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            z_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.trn([0, 0, vector_length / 2]))
            k += 1

        for l in range(deltak):
            q[j - 1] += (robot.joint_limit[j - 1,1] - robot.joint_limit[j - 1,0]) / (deltak - 1)
            robot.add_ani_frame(k * dt, q=q)
            htm_i = robot.fkm(axis="dh", q=q)[i - 1]
            x_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.roty(3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            y_axis[i].add_ani_frame(k * dt,
                                    htm=htm_i * Utils.rotx(-3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            z_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.trn([0, 0, vector_length / 2]))
            k += 1

        for l in range(floor(0.5 * deltak)):
            q[j - 1] += -(robot.joint_limit[j - 1,1]- robot.q0[j-1]) / (floor(0.5 * deltak) - 1)
            robot.add_ani_frame(k * dt, q=q)
            htm_i = robot.fkm(axis="dh", q=q)[i - 1]
            x_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.roty(3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            y_axis[i].add_ani_frame(k * dt,
                                    htm=htm_i * Utils.rotx(-3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
            z_axis[i].add_ani_frame(k * dt, htm=htm_i * Utils.trn([0, 0, vector_length / 2]))
            k += 1

        k += deltak

    k += 2 * deltak
    if i < len(robot.links):
        if i > 0:
            explanation.add_ani_frame(k * dt,
                                      html_text="The joints that are ahead in the kinematic chain do not move the link, " \
                                                "and therefore not the frame.")
            k += 3 * deltak
    else:
        explanation.add_ani_frame(k * dt, html_text="Since this is the last frame, every joint moves it.")
        k += 3 * deltak

    # Move the axis i+1 (if available)
    if i < len(robot.links):

        explanation.add_ani_frame(k * dt, html_text="The " + txt_joint(
            i + 1) + ", for example, when it moves it does not move  " + txt_frame(
            i) + ". The same is true for <b>joint<sub>i</sub></b> with i > " + str(i) + ".")

        q = robot.q0
        for l in range(floor(0.5 * deltak)):
            q[i] += (robot.joint_limit[i,0]  - robot.q0[i])/ (floor(0.5 * deltak) - 1)
            robot.add_ani_frame(k * dt, q=q)
            k += 1

        for l in range(deltak):
            q[i] += (robot.joint_limit[i,1] - robot.joint_limit[i,0]) / (deltak - 1)
            robot.add_ani_frame(k * dt, q=q)
            k += 1

        for l in range(floor(0.5 * deltak)):
            q[i] += -(robot.joint_limit[i,1]- robot.q0[i]) / (floor(0.5 * deltak) - 1)
            robot.add_ani_frame(k * dt, q=q)
            k += 1

    # Erase the axis
    k += deltak
    x_axis[i].add_ani_frame(k * dt, htm=far)
    y_axis[i].add_ani_frame(k * dt, htm=far)
    z_axis[i].add_ani_frame(k * dt, htm=far)

    k += deltak

    k += 1

# Show all DH frames for a while
k += deltak
for i in range(len(robot.links) + 1):
    x_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.roty(3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
    y_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.rotx(-3.14 / 2) * Utils.trn([0, 0, vector_length / 2]))
    z_axis[i].add_ani_frame(k * dt, htm=htm[i] * Utils.trn([0, 0, vector_length / 2]))

explanation.add_ani_frame(k * dt, "Its over!")

sim.save("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/","part_1_5")