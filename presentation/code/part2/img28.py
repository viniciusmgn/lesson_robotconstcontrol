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
        
 



def draw_pol_points(polygon, all_c):

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



    # Draw the closest point to the origin
    for c in all_c:
        ax.scatter(*c, color='#ffb66c', s=100, marker='o', edgecolors='#ffb66c', label="Closest Point", zorder=4)

        
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
    
    ax.scatter(x=0,y=0, color='white', s=100, marker='o', edgecolors='white', label="Closest Point", zorder=4)
    
    

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

    plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image28.svg")
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

all_c=[c_dir]
for i in range(4):

    c_new = support(polygon, c_dir)
    P = P+[c_new]
    c_dir, P_sub = closest_point_on_simplex(P)
    P = P_sub
    all_c.append(c_dir)

    
draw_pol_points(polygon, all_c)

