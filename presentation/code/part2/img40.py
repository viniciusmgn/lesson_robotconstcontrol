import numpy as np
import matplotlib.pyplot as plt

# Define Phi(s)
def phi(s):
    return np.where(s >= 0, s**3 / (2 * (s + 0.2)), 0)

# Define e(p), rho(p), E(p)
def e(x, y):
    return 0.25 * (phi(x - 0.5) + phi(-0.5 - x) + phi(y - 0.5) + phi(-0.5 - y))

def rho(x, y):
    return 0.5 * (x**2 + y**2 - 1)

def E(x, y, eps=0.1):
    rho_val = eps * rho(x, y)
    e_val = e(x, y)
    return rho_val + np.sqrt(rho_val**2 + (1 - 2 * eps) * e_val**2)

# Grid setup
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

# Evaluate functions
e_vals = e(X, Y)
E_vals = E(X, Y)

# Plot configuration
bg_color = '#191919'
contour_color = '#81d41a'
zero_set_color = 'yellow'
text_color = 'white'

fig, axes = plt.subplots(2, 1, figsize=(8, 12))

for ax, data, title in zip(axes, [e_vals, E_vals], [r'$e(p)$', r'$E(p)$']):
    ax.set_facecolor(bg_color)
    ax.contour(X, Y, data, levels=[0.025**2, 0.1**2,0.2**2,0.3**2,0.4**2,0.5**2], colors=contour_color)
    ax.contour(X, Y, data, levels=[0], colors=zero_set_color, linewidths=2)
    ax.set_title(f'Level sets of {title}', color=text_color, fontsize=16)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal')
    ax.tick_params(colors=text_color, labelsize=16)
    ax.set_xlabel('x', color=text_color, fontsize=16)
    ax.set_ylabel('y', color=text_color, fontsize=16)

fig.patch.set_facecolor(bg_color)
plt.tight_layout()

plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image40.svg")
