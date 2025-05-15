import numpy as np
import matplotlib.pyplot as plt

# Define the curve (e.g., sine wave)
t = np.linspace(-2 * np.pi, 2 * np.pi, 400)
x_curve = np.cos(t)+0.3*np.sin(2*t)
y_curve = np.sin(t)-0.1*np.cos(3*t)
curve_points = np.vstack((x_curve, y_curve)).T

# Example vector field: rotate 90 degrees around origin
def get_vector(p):
    
    
    dists = np.linalg.norm(curve_points - p, axis=1)
    closest_idx = np.argmin(dists)
    min_dist = np.linalg.norm(curve_points[closest_idx] - p)
    proj = curve_points[closest_idx]
    t_proj = t[closest_idx]

    # Compute vectors
    tangent_vec = np.array([-np.sin(t_proj)+0.6*np.cos(2*t_proj), np.cos(t_proj)+0.3*np.sin(3*t_proj)])
    tangent_vec = tangent_vec / np.linalg.norm(tangent_vec)
    normal_vec = proj - p
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    
    G = (2/np.pi)*np.arctan(min_dist)
    H = np.sqrt(1.0-0.999*G**2)
    
    vel_vect = 1.5*(G*normal_vec+H*tangent_vec)
    
    
    return [vel_vect[0],vel_vect[1]]

# Generate grid of points where vectors will be plotted
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
U = np.zeros_like(X)
V = np.zeros_like(Y)

# Evaluate the vector field at each point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec = get_vector([X[i, j], Y[i, j]])
        U[i, j], V[i, j] = vec

# Create figure and axis
fig = plt.figure(figsize=(8, 4.5), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])  # full canvas
ax.set_facecolor("#191919")
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor("#191919")
fig.patch.set_facecolor("#191919")
    
# Plot curve
ax.plot(x_curve, y_curve, color="#5983b0", linewidth=2,zorder=10)

# Plot vector field
ax.quiver(X, Y, U, V, color='white', angles='xy', scale_units='xy', scale=10, width=0.003)

# Save or display
plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part3/image1.svg", bbox_inches='tight', pad_inches=0)
plt.show()
