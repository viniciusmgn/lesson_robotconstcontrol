# vector_field_gif.py
# Robust GIF generator: no FuncAnimation, no ImageMagick needed.

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def animate_vector_field_gif(
    f,
    C,
    bounds,
    start=None,
    dt=0.02,
    steps=400,
    field_density=25,
    out_path="vector_field.gif",
    figsize=(6, 6),
    dpi=150,
    fps=30,
    arrow_scale=0.35,
):
    """
    Render and save a GIF showing:
      - a 2D vector field f(x,y)->[u,v]
      - a polyline curve C
      - a particle moving under the field (RK4)
    Dark background (#191919), no axes or grids.

    Parameters
    ----------
    f : callable
        f(x, y) -> [u, v] (list/tuple/array length 2).
    C : list[[x,y], ...] or np.ndarray (N,2)
        Curve to draw.
    bounds : [xmin, xmax, ymin, ymax]
        Plotting/simulation bounds.
    start : [x0, y0] or None
        Initial particle position; defaults to center of bounds.
    dt : float
        Integration time step (RK4).
    steps : int
        Number of frames in the animation (>=2 recommended).
    field_density : int
        Grid samples per axis for the quiver field.
    out_path : str
        Output GIF path.
    figsize : (w,h)
        Matplotlib figure size in inches.
    dpi : int
        Matplotlib figure DPI.
    fps : int
        Frames per second for the GIF (controls playback speed).
    arrow_scale : float
        Scales normalized arrow lengths for readability.
    """

    if steps <= 0:
        raise ValueError("steps must be > 0")
    xmin, xmax, ymin, ymax = bounds
    if start is None:
        start = [(xmin + xmax) * 0.5, (ymin + ymax) * 0.5]
    pos = np.array(start, dtype=float)

    # Prepare vector field grid
    xs = np.linspace(xmin, xmax, field_density)
    ys = np.linspace(ymin, ymax, field_density)
    X, Y = np.meshgrid(xs, ys)

    def _fx(x, y):
        u, v = f(float(x), float(y))
        return u

    def _fy(x, y):
        u, v = f(float(x), float(y))
        return v

    U = np.vectorize(_fx)(X, Y)
    V = np.vectorize(_fy)(X, Y)

    # Normalize for nice arrows
    mag = np.hypot(U, V)
    mag[mag == 0] = 1.0
    U_n = (U / mag) * arrow_scale
    V_n = (V / mag) * arrow_scale

    # Figure setup
    bg = "#191919"
    fg = "#eaeaea"
    curve_color = "#E69F00"
    particle_color = "#0072B2"
    trail_color = "#0072B2"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Static quiver and curve
    ax.quiver(X, Y, U_n, V_n, color=fg, alpha=0.8, pivot="mid", linewidth=0.5)

    C = np.asarray(C, dtype=float)
    if C.ndim == 2 and C.shape[1] == 2 and len(C) > 1:
        ax.plot(C[:, 0], C[:, 1], color=curve_color, lw=2, alpha=0.9, solid_capstyle="round")

    # Dynamic artists: trail and particle
    trail_line, = ax.plot([], [], color=trail_color, lw=2, alpha=0.9)
    particle_dot, = ax.plot([], [], "o", ms=8, color=particle_color, mec="none")

    # Integration helpers
    def rk4_step(p, h):
        k1 = np.array(f(p[0], p[1]), dtype=float)
        k2 = np.array(f(p[0] + 0.5*h*k1[0], p[1] + 0.5*h*k1[1]), dtype=float)
        k3 = np.array(f(p[0] + 0.5*h*k2[0], p[1] + 0.5*h*k2[1]), dtype=float)
        k4 = np.array(f(p[0] + h*k3[0], p[1] + h*k3[1]), dtype=float)
        return p + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def clamp(p):
        p[0] = np.clip(p[0], xmin, xmax)
        p[1] = np.clip(p[1], ymin, ymax)
        return p

    # Frame rendering loop → collect PIL Images
    images = []
    trail = [pos.copy()]

    for _ in range(steps):
        # advance
        pos = rk4_step(pos, dt)
        pos = clamp(pos)
        trail.append(pos.copy())

        # creation
        trail_line, = ax.plot([], [], color=trail_color, lw=2, alpha=0.9)
        particle_dot, = ax.plot([], [], "o", ms=8, color=particle_color, mec="none")

        def init():
            trail_line.set_data([], [])
            particle_dot.set_data([], [])
            return trail_line, particle_dot

        # inside the frame loop (or update function)
        T = np.asarray(trail)
        trail_line.set_data(T[:, 0], T[:, 1])

        # ⬇️ make them sequences
        #particle_dot.set_data([pos[0]], [pos[1]])


        # rasterize current figure into a PNG in-memory
        buf = io.BytesIO()
        fig.canvas.draw()
        fig.savefig(buf, format="png", facecolor=bg, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        frame = Image.open(buf).convert("P")  # palette mode helps reduce GIF size
        images.append(frame)
        buf.close()

    plt.close(fig)

    # Save GIF
    if len(images) == 1:
        images[0].save(out_path)
    else:
        duration_ms = int(1000 / max(1, fps))
        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            disposal=2,  # replace frame
        )
    print(f"Saved animation to: {out_path}")


# --------------------------
# Demo usage
# --------------------------
if __name__ == "__main__":
    # Example vector field (sink + rotation)
    def f(x, y):
        a = -0.6  # radial sink
        b = 0.8   # rotation
        D = x**2+y**2-1
        return [-1*x*D+y, -1*y*D-x]

    # Example curve: circle
    theta = np.linspace(0, 2*np.pi, 200)
    C_demo = np.column_stack([1.0*np.cos(theta), 1.0*np.sin(theta)])

    bounds = [-1.5, 1.5, -1.5, 1.5]
    start = [0.2, 0.2]

    animate_vector_field_gif(
        f=f,
        C=C_demo,
        bounds=bounds,
        start=start,
        dt=0.02,
        steps=300,
        field_density=25,
        out_path="/home/vinicius/Desktop/Aulas/Robot Constrained Control/presentation/images/extra/vector_field.gif",
        figsize=(6, 6),
        dpi=150,
        fps=30,
        arrow_scale=0.35,
    )
