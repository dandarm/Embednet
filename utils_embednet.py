import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import cm
from copy import copy
import networkx as nx
import torch

from curved_edges import curved_edges


edge_color = '#6666bb'
node_color = '#df5c43'

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    bins = np.linspace(1, max(degrees), 25)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))
    plt.hist(degrees, bins=logbins, log=True, histtype='step')
    plt.xscale('log')
    plt.show()
    
# def plot_grafo(G):
#     degrees = dict(G.degree)
#     sizes = [v*20 + v*v/10  for v in degrees.values()]
#     edges_weights = [d['weight']/3 for (u, v, d) in G.edges(data=True)]
#     pos = nx.spring_layout(G, k=1.8, iterations=200)
#     nx.draw(G, nodelist=list(degrees.keys()), node_size=sizes, pos=pos)
    
def custom_draw_edges(ax, G, positions, edges_weights):    
    
    #disegno le edges curve
    curves = curved_edges(G, positions)
    lc = LineCollection(curves, color=edge_color, alpha=0.2, linewidths=edges_weights)
    new_lc=copy(lc) # faccio la copy perché se rieseguo la cella dà errore matplotlib
    ax.add_collection(new_lc)
    
    return ax
    
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)
    #plt.setp(ax.spines.values(), visible=False)
    
    
def plot_grafo2(G, iterations, sizes=None, edges_weights=None, communities=None,  limits=False, nome_file=None, dpi=96):
    """
    Calcola le posizioni dei nodi secondo lo spring layout  https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
    
    """
    plt.figure(figsize=(1000 / dpi, 1000 / dpi))
    ax = plt.gca()

    plot_grafo(ax, G, iterations, communities, edges_weights, limits, sizes)

    fig = plt.gcf()
    fig.tight_layout()
    if(nome_file is not None):
        print("Save graph png")
        plt.savefig(nome_file, dpi=dpi)
        
    print("Show")
    plt.show()
    return fig


def plot_grafo(ax, G, iterations, communities=None, edges_weights=None, limits=None, sizes=None):
    degrees = dict(G.degree)
    if not sizes:
        sizes = [v * 2 + v * v / 10 for v in degrees.values()]
    if not edges_weights:
        edges_weights = [d['weight'] / 30 if 'weight' in d.keys() else 1 for (u, v, d) in G.edges(data=True)]
    positions = nx.spring_layout(G, k=1.8, iterations=iterations)  # default weight='weight' per le edges
    print("Draw edges")
    custom_draw_edges(ax, G, positions, edges_weights)
    print("Draw nodes")
    node_list = np.array(G.nodes())
    node_size = np.array(sizes)
    if communities is not None:  # matrice con dimensione:  num_classi X num_nodi
        colors = cm.rainbow(np.linspace(0, 1, len(communities)))
        for i, c in enumerate(communities):
            community_nodes = node_list[c.astype(bool)]
            community_sizes = node_size[c.astype(bool)]
            nx.draw_networkx_nodes(G, positions, nodelist=community_nodes, node_size=community_sizes,
                                   node_color=colors[i].reshape(1, -1), alpha=0.7, ax = ax)
    else:
        nx.draw_networkx_nodes(G, positions, node_size=sizes, node_color=node_color, alpha=0.7, ax = ax)
    print("Plot")
    plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.setp(ax.spines.values(), visible=False)
    cut = 1.00
    xmax = cut * max(xx for xx, yy in positions.values())
    ymax = cut * max(yy for xx, yy in positions.values())
    if limits:
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)


# find minimum difference between any pair in an unsorted array
# This code is contributed by Pratik Chhajer
def findMinDiff(arr):
    n = len(arr)
    # Sort array in non-decreasing order
    arr = sorted(arr)

    # Initialize difference as infinite
    diff = 10 ** 20

    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(n - 1):
        if arr[i + 1] - arr[i] < diff:
            diff = arr[i + 1] - arr[i]

    # Return min diff
    return diff


def is_outlier(points, threshold=5.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > threshold

def array_wo_outliers(x_array, threshold=7.5):
    x_array = np.array(x_array)
    filtered = x_array[~is_outlier(x_array, threshold)]
    return filtered

def adjust_lightness(color, amount=0.5):
    """
    Per modificare la luminosità di un colore
    :param color: colore originale
    :param amount: se < 1 schiarisce, se > 1 scurisce
    :return:
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])