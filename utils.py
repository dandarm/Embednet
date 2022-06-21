import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import copy
import networkx as nx
import tensorflow as tf

from curved_edges import curved_edges

dpi = 96
edge_color = '#6666bb'
node_color = '#df5c43'

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    bins = np.linspace(1, max(degrees), 25)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(degrees, bins=logbins, log=True, histtype = 'step')
    plt.xscale('log')
    plt.show()
    
def plot_grafo(G):
    degrees = dict(G.degree)
    sizes = [v*20 + v*v/10  for v in degrees.values()]
    edges_weights = [d['weight']/3 for (u, v, d) in G.edges(data=True)]
    pos = nx.spring_layout(G, k=1.8, iterations=200) 
    nx.draw(G, nodelist=list(degrees.keys()), node_size=sizes, pos=pos)
    
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
    
    
def plot_grafo2(G, iterations, nome_file=None):
    plt.figure(figsize=(1000/dpi,1000/dpi))
    
    print("Draw nodes")
    positions = nx.spring_layout(G, k=1.8, iterations=iterations)
    degrees = dict(G.degree)
    sizes = [v*2 + v*v/10  for v in degrees.values()]
    nx.draw_networkx_nodes(G, positions, node_size=sizes, node_color=node_color, alpha=0.7)
    
    print("Draw edges")
    ax = plt.gca()
    edges_weights = [d['weight']/30 if 'weight' in d.keys() else 1 for (u, v, d) in G.edges(data=True)]
    custom_draw_edges(ax, G, positions, edges_weights)
    
    print("Plot")
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    plt.setp(ax.spines.values(), visible=False)
    fig = plt.gcf()
    fig.tight_layout()
    
    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    
    if(nome_file is not None):
        print("Save graph png")
        plt.savefig(nome_file, dpi=dpi * 10)
    
    print("Show")
    plt.show()


def add_histogram(writer, tag, values, step):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """
    # counts, bin_edges = np.histogram(values, bins=bins)
    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Therefore we drop the start of the first bin
    # bin_edges = bin_edges[1:]

    bins = np.arange(0, len(values)) + 1
    bins = np.linspace(1, len(values), len(values) + 1, endpoint=True)

    # Fill fields of histogram proto
    hist = tf.compat.v1.HistogramProto()
    hist.min = float(np.min(bins))
    hist.max = float(np.max(bins))
    # hist.num = int(np.prod(values.shape))
    # hist.sum = float(np.sum(values))
    # hist.sum_squares = float(np.sum(values ** 2))

    bins = bins[1:]

    for edge in bins:
        hist.bucket_limit.append(edge)
    for c in values:
        d = c * 30.0 / float(len(values))
        hist.bucket.append(d)

    summary = tf.compat.v1.summary.Summary(value=[tf.compat.v1.summary.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()


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