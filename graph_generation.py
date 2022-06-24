import networkx as nx
import numpy as np
from config_valid import Config, Labels



def create_ER(Num_nodes, p, N_graphs):
    """
    Num_nodes:  Graph nodes number
    p:  Probability for edge creation
    N_graphs: ogni tot abbiamo un tipo di grafi originato da una diversa distribuzione
    """
    grafi = []
    actual_probs = []
    for i in range(N_graphs):
        gr = nx.erdos_renyi_graph(Num_nodes, p)  # seed = 1
        #grafi.append(nx.to_numpy_matrix(gr))
        grafi.append(gr)
        actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1)
        actual_probs.append(actual_p)
        
    print("Mean connectivity for each node:", end=' ')    
    print(round(np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() for t in grafi]).mean(), 3), end=' ')
    print(f'p={p}')
        
    return grafi, actual_probs

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

def dataset_nclass_ER(config_class):
    conf = config_class.conf
    N = conf['graph_dataset']['Num_nodes']
    list_p = conf['graph_dataset']['list_p']
    Num_grafi_per_tipo = conf['graph_dataset']['Num_grafi_per_tipo']
    
    dataset_grafi_nx = []
    dataset_labels = []
    label_kind = config_class.get_mode()['labels']
    #
    # if conf['model']['neurons_per_layer'][-1] > 1:
    #     scalar_embedding = False
    # else:
    #     scalar_embedding = True

    if label_kind == Labels.onehot:
        n_classi = len(list_p)
        onehot_matrix = np.eye(n_classi)
        for i, p in enumerate(list_p):
            grafi_p, _ = create_ER(N, p, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_p
            dataset_labels = dataset_labels + [onehot_matrix[i]] * len(grafi_p)
            #print(dataset_labels)

    elif label_kind == Labels.zero_one:
        # ho quindi solo due classi
        for i, p in enumerate(list_p):
            grafi_p, _ = create_ER(N, p, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_p
            dataset_labels = dataset_labels + [i] * len(grafi_p)

    elif label_kind == Labels.prob:
        for i, p in enumerate(list_p):
            grafi_p, actual_probs = create_ER(N, p, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_p
            dataset_labels.extend(actual_probs)
        
    return dataset_grafi_nx, np.array(dataset_labels), list_p

def dataset_regression_ER(config_class):
    conf = config_class.conf
    Num_nodes = conf['graph_dataset']['Num_nodes']
    range_p = conf['graph_dataset']['range_p']
    Num_grafi_tot = conf['graph_dataset']['Num_grafi_totali']

    dataset_grafi_nx = []
    dataset_labels = []
    assert conf['model']['neurons_per_layer'][-1] == 1
    probs = np.random.uniform(low=range_p[0], high=range_p[1], size=Num_grafi_tot)
    for p in probs:
        gr = nx.erdos_renyi_graph(Num_nodes, p)
        dataset_grafi_nx.append(gr)
        actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1)
        dataset_labels.append(actual_p)

    return dataset_grafi_nx, np.array(dataset_labels)
