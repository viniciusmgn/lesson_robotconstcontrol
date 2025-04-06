import matplotlib.pyplot as plt
from mpmath import mp, mpf, cos, sqrt, log10

# Set arbitrary precision
mp.dps = 1000  # decimal places

# --- Parameters ---
max_iter = 17
epsilon = mpf('1e-1000')
bg_color = '#191919'
line_color = '#81d41a'
text_color = 'white'

# --- Iteration 1: x[k+1] = cos(x[k]) ---
def cos_iter(x0, max_iter):
    xs = [mpf(x0)]
    for _ in range(max_iter):
        xs.append(cos(xs[-1]))
    return xs

# --- Iteration 2: x[k+1] = x[k]/2 + 1/x[k] ---
def custom_iter(x0, max_iter):
    xs = [mpf(x0)]
    for _ in range(max_iter):
        if xs[-1] == 0:
            xs.append(mpf(0))
        else:
            xs.append(xs[-1] / 2 + 1 / xs[-1])
    return xs

# --- Compute sequences ---
cos_seq = cos_iter(1.0, max_iter)
cos_fixed = cos_seq[-1]
cos_errors = [abs(x - cos_fixed) for x in cos_seq]
cos_errors = [max(e, epsilon) for e in cos_errors]

custom_seq = custom_iter(1.5, max_iter)
custom_fixed = sqrt(2)
custom_errors = [abs(x - custom_fixed) for x in custom_seq]
custom_errors = [max(e, epsilon) for e in custom_errors]

# --- Plot ---
k_vals = list(range(max_iter + 1))
fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig.patch.set_facecolor(bg_color)

titles = [
    r'$x_{k+1} = \cos(x_k)$',
    r'$x_{k+1} = \frac{x_k}{2} + \frac{1}{x_k}$'
]

for ax, errors, title in zip(axs, [cos_errors, custom_errors], titles):
    ax.set_facecolor(bg_color)
    ax.plot(k_vals[0:10], [-float(log10(e)) for e in errors][0:10], color=line_color, marker='o')
    ax.set_ylabel(r'$-\log_{10}(|x_k - x^*|)$', color=text_color,fontsize=15)
    ax.set_title(title, color=text_color,fontsize=15)
    ax.grid(True, color='white', alpha=0.3)
    ax.tick_params(colors=text_color)
    ax.tick_params(axis='both', labelsize=15) 

axs[-1].set_xlabel('Iteration $k$', color=text_color,fontsize=15)

plt.tight_layout()
plt.show()
