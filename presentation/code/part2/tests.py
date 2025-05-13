import uaibot as ub

# Create a robot and an object


robot = ub.Robot.create_franka_emika_3()
cvx_obj = ub.Box(htm=ub.Utils.trn([1.5, 0, 0]), width=0.1, depth=0.2, height=0.3)


#Compute a distance structure between them at configuration q=[0,0.1,0.2,0.3,0.4,0.5,0.6]
#You can also specify smoothing parameters h and eps (default are h=eps=0, i.e., traditional Euclidean)

ds = robot.compute_dist(obj = cvx_obj, q=[0,0.1,0.2,0.3,0.4,0.5,0.6], h=0.1, eps = 0.05)

#ds is an object of the class 'DistStructRobotObj' with many interesting information.
#The robot is modelled using collision objects. They are divided according to the index of the link, 
#and the index of the object at that link. For example, the following would give the distance (either
# Euclidean or Goncalves') between the first object  attached to the fourth link.

dist = ds.get_item(link_number = 3, link_col_obj_number=0).distance

#Lets also get the transposed gradient (Jacobian) of this distance in the configuration.
#It is a 1xn np.matrix, in which n is the number of joints

grad_dist = ds.get_item(link_number = 3, link_col_obj_number=0).jac_distance

#Lets also get the witness points:

point_link = ds.get_item(link_number = 3, link_col_obj_number=0).point_link
point_object = ds.get_item(link_number = 3, link_col_obj_number=0).point_object