import numpy as np
import itertools
import networkx as nx
import networkx.algorithms.isomorphism as iso

from multiprocessing import SimpleQueue
from multiprocessing.pool import Pool
from multiprocessing import set_start_method
#set_start_method("spawn")

# funzione di match per rendere diversi gli archi
em = iso.numerical_edge_match("rank", 1)

def build_permutation_complete_graph(n):
    graph = nx.complete_graph(n)
    print(f"Edges: {graph.edges()}")
    # Generazione delle permutazioni degli archi
    permutations = list(itertools.permutations(graph.edges()))

    return permutations


def init_worker(shared_queue):
    global valid_perms
    valid_perms = shared_queue



def get_valid_p(p, valid_perms):
    #global valid_perms
    # print(f"\n permutazione: {p}")
    G = nx.Graph(p)
    for i, e in enumerate(p):
        G.add_edge(*e, rank=i + 1)
        # G[e[0]][e[1]]['rank'] = i + 1
    # print(G.edges(data=True))
    #draw_graph(G)
    return all(not nx.is_isomorphic(G, H, edge_match=em) for H in valid_perms)  # era if
    #    print(f"aggiunto {G.edges(data=True)}")
    # valid_perms.append(G)
    #    valid_perms.put(G)




# with Pool(processes=2) as pool:
#    dataset_pyg = pool.map(self.convert_G, zip(graph_list_nx, range(total)))