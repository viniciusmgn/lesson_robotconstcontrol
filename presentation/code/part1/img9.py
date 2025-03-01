from uaibot import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

width=550
height=500

hist_s=[]
hist_phi=[]
for i in range(1000):
    s = -2+4*(i/1000)
    hist_s.append(s)
    hist_phi.append(-s/(np.sqrt(abs(s))+1e-6))
    


bg_color = "#191919"

# Create the figure and subplots
fig, axs = plt.subplots(1, 1, figsize=(10, 6), sharex=True)
fig.patch.set_facecolor(bg_color)

# Plot the first three entries of r vs t

axs.plot(hist_s, hist_phi)

axs.set_xlabel("s")
axs.set_ylabel("g(s)")
axs.grid(True, linestyle="--", alpha=0.5)


# Set colors for text, ticks, and labels

axs.set_facecolor(bg_color)
axs.tick_params(axis="both", colors="white")
axs.xaxis.label.set_color("white")
axs.yaxis.label.set_color("white")
for spine in axs.spines.values():
    spine.set_edgecolor("white")
axs.legend(facecolor=bg_color, edgecolor="white", labelcolor="white")
    

plt.show()

#plt.savefig("/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/part1/image8.png", facecolor=bg_color)

    