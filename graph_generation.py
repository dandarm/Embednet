import networkx as nx
import numpy as np



def create_ER(Num_nodes, p, N_graphs):
    N = 300 # Graph nodes number
    p_er = 0.02   # Probability for edge creation
    Num_grafi_per_tipo = 100 # ogni tot abbiamo un tipo di grafi originato da una diversa distribuzione

    grafi = []
    for i in range(N_graphs):
        gr = nx.erdos_renyi_graph(Num_nodes, p)  # seed = 1
        #grafi.append(nx.to_numpy_matrix(gr))
        grafi.append(gr)
        
    print("Mean connectivity for each node:", end=' ')    
    print(round(np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() for t in grafi]).mean(), 3))
        
    return grafi