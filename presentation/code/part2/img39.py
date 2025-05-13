import numpy as np
import matplotlib.pyplot as plt

# Define the function Phi(s)
def phi(s):
    return np.where(s >= 0, s**3 / (2*(s + 0.5)), 0)

# Define numerical derivatives
def numerical_derivative(y, x):
    return np.gradient(y, x)

# Create input values
s_vals = np.linspace(-1, 3, 1000)
phi_vals = phi(s_vals)
phi_prime_vals = numerical_derivative(phi_vals, s_vals)
phi_double_prime_vals = numerical_derivative(phi_prime_vals, s_vals)

# Plot styling
line_color = '#81d41a'
bg_color = '#191919'
text_color = 'white'

plt.figure(figsize=(10, 9))

# Plot Phi(s)
plt.subplot(3, 1, 1)
plt.plot(s_vals, phi_vals, color=line_color, linewidth=2)
plt.title(r'$\Phi(s)$', color=text_color, fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_facecolor(bg_color)
plt.gcf().patch.set_facecolor(bg_color)
plt.tick_params(colors=text_color, labelsize=16)
plt.xlabel('s', color=text_color, fontsize=16)
plt.ylabel(r'$\Phi(s)$', color=text_color, fontsize=16)
plt.xlim(-1, 2.99)

# Plot Phi'(s)
plt.subplot(3, 1, 2)
plt.plot(s_vals, phi_prime_vals, color=line_color, linewidth=2)
plt.title(r"$\Phi'(s)$", color=text_color, fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_facecolor(bg_color)
plt.tick_params(colors=text_color, labelsize=16)
plt.xlabel('s', color=text_color, fontsize=16)
plt.ylabel(r"$\Phi'(s)$", color=text_color, fontsize=16)
plt.xlim(-1, 2.99)

# Plot Phi''(s)
plt.subplot(3, 1, 3)
plt.plot(s_vals, phi_double_prime_vals, color=line_color, linewidth=2)
plt.title(r"$\Phi''(s)$", color=text_color, fontsize=16)
plt.grid(True, linestyle='--', alpha=0.3)
plt.gca().set_facecolor(bg_color)
plt.tick_params(colors=text_color, labelsize=16)
plt.xlabel('s', color=text_color, fontsize=16)
plt.ylabel(r"$\Phi''(s)$", color=text_color, fontsize=16)
plt.xlim(-1, 2.99)

plt.tight_layout()


plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part2/image39.svg")