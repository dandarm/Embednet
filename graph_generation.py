import networkx as nx
import numpy as np




def create_ER(Num_nodes, p, N_graphs):
    """
    Num_nodes:  Graph nodes number
    p:  Probability for edge creation
    N_graphs: ogni tot abbiamo un tipo di grafi originato da una diversa distribuzione
    """
    grafi = []
    probs = []
    for i in range(N_graphs):
        gr = nx.erdos_renyi_graph(Num_nodes, p)  # seed = 1
        #grafi.append(nx.to_numpy_matrix(gr))
        grafi.append(gr)
        actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1)
        probs.append(actual_p)
        
    print("Mean connectivity for each node:", end=' ')    
    print(round(np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() for t in grafi]).mean(), 3), end= ' ')
    print(f'p={p}')
        
    return grafi, probs

# def dataset_2class_ER(config):
#     N = config['graph_dataset']['Num_nodes']
#     p1 = config['graph_dataset']['p1ER']
#     p2 = config['graph_dataset']['p2ER']
#     Num_grafi_per_tipo = config['graph_dataset']['Num_grafi_per_tipo']
#
#     grafi_0 = create_ER(N, p1, Num_grafi_per_tipo)
#     grafi_1 = create_ER(N, p2, Num_grafi_per_tipo)
#     dataset_grafi_nx = grafi_0 + grafi_1
#     #dataset_labels = np.array([0]*len(grafi_0) + [1]*len(grafi_1))
#
#     # nel caso dim_embedding = 1: le labels corrispondono proprio al valore della probabilità
#     dataset_labels = np.array([p1]*len(grafi_0) + [p2]*len(grafi_1))
#
#     return dataset_grafi_nx, dataset_labels, p1, p2

def dataset_nclass_ER(config):
    N = config['graph_dataset']['Num_nodes']
    list_p = config['graph_dataset']['list_p']
    Num_grafi_per_tipo = config['graph_dataset']['Num_grafi_per_tipo']
    
    dataset_grafi_nx = []
    dataset_labels = []
    if config['model']['neurons_per_layer'][-1] > 1:
        scalar_embedding = False
    else:
        scalar_embedding = True

    if not scalar_embedding:  # se l'embedding ha più di una sola dimensione la label è un intero o one-hot
        for i, p in enumerate(list_p):
            grafi_p, _ = create_ER(N, p, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_p
            dataset_labels = dataset_labels + [i] * len(grafi_p)
        print(f'Labels values: {set(dataset_labels)}')

    else:  # swe l'embedding è scalare devo calcolare la probabilità attuale del grafo,
           # non quella che ho impostato alla generazione
        for i, p in enumerate(list_p):
            grafi_p, probs = create_ER(N, p, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_p
            for p in probs:
                dataset_labels = dataset_labels + [p]
        #print(f'Labels values: {set(dataset_labels)}')
        
    return dataset_grafi_nx, np.array(dataset_labels), list_p