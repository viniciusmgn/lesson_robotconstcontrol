import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def closest_point_on_simplex(simplex):
    simplex = np.array(simplex)  

    if len(simplex) == 1:  
        return simplex[0], simplex

    elif len(simplex) == 2:  
        A, B = simplex
        AB = B - A
        AO = -A  

        t = np.dot(AO, AB) / (1e-5+np.dot(AB, AB))

        if t < 0:
            return A, [A]  
        elif t > 1:
            return B, [B]  
        else:
            closest_point = A + t * AB  
            return closest_point, [A, B]

    elif len(simplex) == 3:  
        A, B, C = simplex
        
        p1, s1 = closest_point_on_simplex([A, B])  
        p2, s2 = closest_point_on_simplex([A, C]) 
        p3, s3 = closest_point_on_simplex([C, B]) 
        
        d1 = p1[0]**2+p1[1]**2
        d2 = p2[0]**2+p2[1]**2
        d3 = p3[0]**2+p3[1]**2
        
        if d1<d2:
            if d1<d3:
                return p1, s1
            else:
                return p3, s3
        else:
            if d2<d3:
                return p2, s2
            else:
                return p3,s3
        




def draw_step_gjk(polygon, simplex, subsimplex, closest_point, origin, vector, draw_contour,k,type):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#191919")  # Dark background
    fig.patch.set_facecolor("#191919")

    # Set axis to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='both', colors='white')

    # Draw the polygon (closed loop)
    polygon = np.array(polygon)
    polygon = np.vstack((polygon,polygon[0,:]))
    ax.plot(*polygon.T, color='#81d41a', linewidth=2, linestyle='-', label="Polygon")

    # Draw the simplex
    if len(simplex)>0:
        simplex = np.array(simplex)
        simplex = np.vstack((simplex,simplex[0,:]))
        if len(simplex) > 1:
            ax.plot(*simplex.T, color='#ec2ed7', linewidth=2, linestyle='--', label="Simplex")
        ax.scatter(*simplex.T, color='#ec2ed7', s=50, label="Simplex Vertices")

    # Draw the subsimplex (1 or 2 points)
    if len(subsimplex)>0:
        subsimplex = np.array(subsimplex)
        subsimplex = np.vstack((subsimplex,subsimplex[0,:]))
        if len(subsimplex) > 1:
            ax.plot(*subsimplex.T, color='#5983b0', linewidth=2, linestyle='-', label="Subsimplex")
            

        ax.scatter(*subsimplex.T, color='#5983b0', s=80, edgecolors='#5983b0', label="Subsimplex Vertices", zorder=3)


    # Draw the closest point to the origin
    if len(closest_point)>0:
        if type:
            ax.scatter(*closest_point, color='#ffb66c', s=100, marker='o', edgecolors='#ffb66c', label="Closest Point", zorder=4)
        else:
            ax.scatter(*closest_point, color='#ff0000', s=100, marker='o', edgecolors='#ff0000', label="Closest Point", zorder=4)
        
    #Draw the direction
    if len(origin)>0:
        ax.quiver(*np.array(origin), *np.array(vector), angles='xy', scale_units='xy', scale=1, color='#ff0000', linewidth=2, zorder=8)


    ax.scatter(x=0,y=0, color='white', s=100, marker='o', edgecolors='white', label="Closest Point", zorder=4)
    
    # Compute automatic axis limits
    all_points = np.vstack([polygon, [0, 0]])  # Include origin
    min_x_t, min_y_t = all_points.min(axis=0)
    max_x_t, max_y_t = all_points.max(axis=0)
    
    delta = 0*0.25+max(max_x_t-min_x_t,max_y_t-min_y_t)
    
    max_x = (max_x_t+min_x_t)/2 + delta/2
    min_x = (max_x_t+min_x_t)/2 - delta/2
    max_y = (max_y_t+min_y_t)/2 + delta/2
    min_y = (max_y_t+min_y_t)/2 - delta/2    
    
    # Add margin
    margin_x = (max_x - min_x) * 0.05 if max_x - min_x > 0 else 1
    margin_y = (max_y - min_y) * 0.05 if max_y - min_y > 0 else 1

    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    
    
    if draw_contour:
        # Create grid
        x = np.linspace(min_x - margin_x, max_x + margin_x, 400)
        y = np.linspace(min_y - margin_y, max_y + margin_y)
        X, Y = np.meshgrid(x, y)
        Z = vector[0]*X+vector[1]*Y
        
        
        nrm = np.sqrt(vector[0]**2+vector[1]**2)
        levels = [ nrm*0.1*(i-25) for i in range(40) ]
        
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
        
        contour = ax.contour(X, Y, Z, levels=levels, colors=color_level,zorder=1)

    # Axis lines
    # ax.axhline(0, color='white', linewidth=1)
    # ax.axvline(0, color='white', linewidth=1)

    #ax.legend(loc="upper right", fontsize=8)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # **Remove the ticks**
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    # Optional: Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image27_"+str(k)+".svg")
    plt.show()
    


def support(polygon, direction):
    direction = -np.array(direction)  # Convert direction to a NumPy array
    polygon = np.array(polygon)  # Convert polygon to a NumPy array

    # Compute dot products of all points with the direction vector
    projections = np.dot(polygon, direction)

    # Find the index of the maximum projection
    max_index = np.argmax(projections)

    # Return the point with the max projection
    return tuple(polygon[max_index])

def compute_polygon_from_halfspaces(A, b):
    A = np.asarray(A)
    b = np.asarray(b).flatten()  # Ensure b is a 1D array

    n = A.shape[0]
    vertices = []

    # Generate all pairs of constraint lines
    for i, j in combinations(range(n), 2):
        A_sub = np.vstack([A[i], A[j]])  # Select two constraint rows
        b_sub = np.array([b[i], b[j]])   # Select corresponding b values

        # Solve A_sub * x = b_sub to find intersection point
        try:
            x_intersect = np.linalg.solve(A_sub, b_sub)
        except np.linalg.LinAlgError:
            continue  # Skip parallel or coincident lines
        
        # Check if the intersection satisfies all inequalities Ax <= b
        if np.all(A @ x_intersect <= b + 1e-9):  # Small tolerance to handle floating point errors
            vertices.append(tuple(x_intersect))

    # Remove duplicates and sort points counterclockwise
    vertices = list(set(vertices))  # Remove duplicates
    if len(vertices) < 3:
        raise ValueError("The given half-spaces do not form a valid polygon.")

    # Compute centroid to sort points counterclockwise
    centroid = np.mean(vertices, axis=0)
    vertices.sort(key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))

    return [list(v) for v in vertices]


x0=2
y0=4
N=30
polygon = []

A = np.matrix(np.zeros((0,2)))
b = np.matrix(np.zeros((0,1)))


  
hh=-0.8
xc = -1.2-1.27*hh
yc = 1.2+0.568*hh

for i in range(N):
    t = 2*np.pi*(N-i)/N
    r = 1.0+0.9*np.sin(1.8*t)
    x = np.cos(t)/2.5
    y = np.sin(t)
    a = np.matrix([x,y])
    
    
    
    A = np.vstack((A,a))
    b = np.vstack((b,r+a[0,0]*xc+a[0,1]*yc))
    
polygon = compute_polygon_from_halfspaces(A,b)

polygon[3][0]+=0.15
#polygon.append(polygon[0])

# polygon.append(polygon[0])    

# Example Usage
# polygon = [(1, 2), (3, 4), (2, 6), (-1, 5), (-2, 3),(1,2)]
c_dir = polygon[-2]
P = [c_dir]
    
k = 0
for i in range(4):

    c_new = support(polygon, c_dir)
    
    draw_step_gjk(polygon, P, [], c_dir, (0,0), c_dir, False,k,True)
    
    k+=1
    
    
    
    draw_step_gjk(polygon, P, [], [], c_dir, (-c_dir[0],-c_dir[1]) , False,k,True)
    
    k+=1
    
    draw_step_gjk(polygon, P, [], [], c_dir, (-c_dir[0],-c_dir[1]) , True,k,True)
    
    k+=1
    
    draw_step_gjk(polygon, P, [], c_new, [], [] , False,k,False)
    
    k+=1

    P = P+[c_new]
    
    draw_step_gjk(polygon, P, [], [],[],[], False,k,True)
    
    k+=1

    c_dir, P_sub = closest_point_on_simplex(P)


    draw_step_gjk(polygon, P, P_sub, c_dir,[],[], False,k,True)
    
    k+=1

    P = P_sub
    
    print(c_dir)

