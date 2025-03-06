import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def plot_contour():
    """
    Plots contour curves of the function f(x,y) = x^4 + x^2.
    """
    # Define the function
    def f(x, y):
        return 2*x+8*y   # Ensure variation in y
    
    # Create grid
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#191919')
    ax.set_facecolor('#191919')
    
    levels = [ 1.5*(i-15) for i in range(30) ]
    
    lvl_min = levels[0]
    lvl_max = levels[-1]
    
    # Plot contour lines
    color_level=[]
    for lvl in levels:
        x = (lvl-lvl_min)/(lvl_max-lvl_min)
        x=(1-x)**2
        value_red = int(190*x)
        value_green = int(50+205*x)
        value_blue = int(190*x)
        color_level.append( f"#{value_red:02X}{value_green:02X}{value_blue:02X}" )
    
    contour = ax.contour(X, Y, Z, levels=levels, colors=color_level)
    #ax.clabel(contour, inline=True, fontsize=8, colors='white')
    
    # Customize axis
    ax.set_xlabel('x', color='white')
    ax.set_ylabel('y', color='white')
    ax.tick_params(colors='white')
    
    return ax



def plot_polygon(ax, A, b):
    """
    Plots the boundary of a 2D polygon defined by the inequalities Ax <= b.
    
    Parameters:
    A (numpy.ndarray): Coefficient matrix of shape (n, 2)
    b (numpy.ndarray): Right-hand side vector of shape (n,)
    """
    # Convert inequalities into halfspaces representation for scipy
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])
    
    # Find an interior point by solving a feasibility problem
    c = np.zeros(A.shape[1])  # Zero objective (just find a feasible point)
    res = linprog(c, A_ub=A, b_ub=b, method='highs')
    
    if not res.success:
        raise ValueError("No feasible region found for the given inequalities.")
    interior_point = res.x
    
    # Compute polygon vertices using HalfspaceIntersection
    hs = HalfspaceIntersection(halfspaces, interior_point)
    vertices = hs.intersections
    
    # Sort vertices in counterclockwise order
    hull = ConvexHull(vertices)
    ordered_vertices = np.append(vertices[hull.vertices], [vertices[hull.vertices[0]]], axis=0)  # Ensure closed loop
    
    # Plot
    # fig, ax = plt.subplots()
    ax.set_facecolor('#191919')
    ax.plot(*ordered_vertices.T, 'r-', linewidth=2)
    ax.set_xlabel('$x$', color='white', fontsize=16)
    ax.set_ylabel('$y$', color='white', fontsize=16)
    ax.tick_params(colors='white')
    
    # plt.show()

# Run the plot function

A=np.matrix(np.zeros((0,2)))
b=np.matrix(np.zeros((0,1)))

for i in range(6):
    theta=i*2*np.pi/6
    a = np.matrix([np.cos(theta), np.sin(theta)])
    A =np.vstack((A,a))
    b = np.vstack((b,0.8+i/8))


ax = plot_contour()
plot_polygon(ax, A, b)

c = -np.array([2.0,8.0])  # Zero objective (just find a feasible point)
res = linprog(c, A_ub=A, b_ub=b, bounds=(-1000,1000), method='highs')

ax.scatter(res.x[0], res.x[1], color='cyan', s=100, zorder=100)

plt.show()



#plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/image8.png", facecolor=bg_color)

    