import uaibot as ub

# Create a robot 

robot = ub.Robot.create_franka_emika_3()

#Compute a auto-distance at configuration q=[0,0,0,0,0,0,0]
#You can also specify smoothing parameters h and eps (default are h=eps=0, i.e., traditional Euclidean)

ds = robot.compute_dist_auto(q=[0,0,0,0,0,0,0], h=0.1, eps = 0.05)

#ds is an object of the class 'DistStructRobotAuto' with many interesting information.
#The robot is modelled using collision objects. They are divided according to the index of the link, 
#and the index of the object at that link. For example, the following would give the distance (either
# Euclidean or Goncalves') between the first object  attached to the first link and the second 
#object attached to the fifth link.

dist = ds.get_item(link_number_1 = 0, link_col_obj_number_1=0,link_number_2 = 4, link_col_obj_number_2=1).distance

#Lets also get the transposed gradient (Jacobian) of this distance in the configuration.
#It is a 1xn np.matrix, in which n is the number of joints

grad_dist = ds.get_item(link_number_1 = 0, link_col_obj_number_1=0,link_number_2 = 4, link_col_obj_number_2=1).jac_distance

#Lets also get the witness points:

point_link_1 = ds.get_item(link_number_1 = 0, link_col_obj_number_1=0,link_number_2 = 4, link_col_obj_number_2=1).point_link_1
point_link_2 = ds.get_item(link_number_1 = 0, link_col_obj_number_1=0,link_number_2 = 4, link_col_obj_number_2=1).point_link_2
   