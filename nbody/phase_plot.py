# ~\~ language=Python filename=nbody/phase_plot.py
# ~\~ begin <<lit/index.md|nbody/phase_plot.py>>[init]
from matplotlib import pyplot as plt
from matplotlib import rcParams

import numpy as np
from nbody.cft import Box

rcParams["font.family"] = "serif"

# ~\~ begin <<lit/index.md|create-triangulation>>[init]
def box_triangles(box):
    idx = np.arange(box.size, dtype=int).reshape(box.shape)

    x0 = idx[:-1,:-1]
    x1 = idx[:-1,1:]
    x2 = idx[1:,:-1]
    x3 = idx[1:,1:]
    upper_triangles = np.array([x0, x1, x2]).transpose([1,2,0]).reshape([-1,3])
    lower_triangles = np.array([x3, x2, x1]).transpose([1,2,0]).reshape([-1,3])
    return np.r_[upper_triangles, lower_triangles]
# ~\~ end
# ~\~ begin <<lit/index.md|triangle-area>>[init]
def triangle_area(x, y, t):
    return (x[t[:,0]] * y[t[:,1]] + x[t[:,1]] * y[t[:,2]] + x[t[:,2]] * y[t[:,0]] \
          - x[t[:,1]] * y[t[:,0]] - x[t[:,2]] * y[t[:,1]] - x[t[:,0]] * y[t[:,2]]) / 2
# ~\~ end
# ~\~ begin <<lit/index.md|phase-space-plot>>[init]
def plot_for_time(box, triangles, time, bbox=[(5,45), (5,45)], fig=None, ax=None):
    fn = 'data/x.{0:05d}.npy'.format(int(round(time*1000)))
    with open(fn, "rb") as f:
        x = np.load(f)
        p = np.load(f)

    area = abs(triangle_area(x[:,0], x[:,1], triangles)) / box.res**2
    sorting = np.argsort(area)[::-1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8,8))

    ax.tripcolor(x[:,0], x[:,1], triangles[sorting], np.log(1./area[sorting]),
                  alpha=0.3, vmin=-2, vmax=2, cmap='YlGnBu')
    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    return fig, ax
# ~\~ end

if __name__ == "__main__":
    box = Box(2, 256, 50.0)
    triangles = box_triangles(box)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for i, t in enumerate([0.5, 1.0, 2.0]):
        plot_for_time(box, triangles, t, fig=fig, ax=axs[0,i])
        axs[0,i].set_title(f"a = {t}")

    for i, t in enumerate([0.5, 1.0, 2.0]):
        plot_for_time(box, triangles, t, bbox=[(15,30), (5, 20)], fig=fig, ax=axs[1,i])

    fig.tight_layout()
    fig.savefig('docs/figures/x.collage.png', dpi=150)
# ~\~ end
