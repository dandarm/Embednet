import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from cycler import cycler
import numpy as np
from sklearn import datasets
from plt_parameters import get_colors_to_cycle_rainbow8, get_colors_to_cycle_rainbowN, get_colors_to_cycle_sequential


def parallel_coord1(ys, titolo_grafico):
    #iris = datasets.load_iris()
    #ynames = iris.feature_names
    #ys = iris.data



    ymins = ys.min(axis=0)
    ymin_abs = ymins.min()
    ymaxs = ys.max(axis=0)
    ymax_abs = ymaxs.max()
    dys = ymaxs - ymins
    dys_abs = ymax_abs - ymin_abs
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    ymin_abs -= dys_abs * 0.05
    ymax_abs += dys_abs * 0.05

    dys = ymaxs - ymins
    dys_abs = ymax_abs - ymin_abs

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    fig, host = plt.subplots(figsize=(10,4))

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]

    for i, ax in enumerate(axes):
        #ax.set_ylim(ymins[i], ymaxs[i])
        ax.set_ylim(ymin_abs, ymax_abs)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            #ax.set_axis_off()
            ax.spines['left'].set_visible(False)
            #ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))  #.set_visible(False)
            ax.yaxis.set_major_locator(ticker.NullLocator())

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    #host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    #host.xaxis.tick_top()
    host.set_title(titolo_grafico, fontsize=16, pad=12)

    #colors = plt.cm.Set2.colors
    #legend_handles = [None for _ in iris.target_names]

    print('start time')
    start = time.time()


    # for j in range(ys.shape[0]):
    #     # create bezier curves
    #     verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
    #                      np.repeat(zs[j, :], 3)[1:-1]))
    #     codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    #     path = Path(verts, codes)
    #     color = get_colors_to_cycle_sequential(len(ys))[j]
    #     patch = patches.PathPatch(path, facecolor='none', lw=2, alpha=0.3, edgecolor=color)
    #     #legend_handles[iris.target[j]] = patch
    #     host.add_patch(patch)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", get_colors_to_cycle_rainbow8())
    custom_cycler = (cycler(color=get_colors_to_cycle_sequential(len(ys))))
    host.set_prop_cycle(custom_cycler)
    host.plot(ys.T,  lw=2, alpha=0.1)  #, cmap=cmap) #, color=color)  marker='.',
    host.plot(ys.T, '.') #, alpha=0.8)


    end = time.time()
    print(end - start)
    #host.legend(legend_handles, iris.target_names,
    #            loc='lower center', bbox_to_anchor=(0.5, -0.18),
    #            ncol=len(iris.target_names), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.show()
    return