import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import copy
import networkx as nx

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
    

