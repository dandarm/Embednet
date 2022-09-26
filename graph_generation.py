from enum import Enum
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool

from config_valid import Labels, TrainingMode
from train import GeneralDataset

class GraphType(Enum):
    ER = 1
    Regular = 2
    CM = 3
    SBM = 4

class GenerateGraph():
    def __init__(self, config_class):
        self.config_class = config_class
        self.conf = self.config_class.conf
        self.N = self.conf['graph_dataset']['Num_nodes']
        if isinstance(self.N, list):
            self.numnodes_islist = True
        self.list_p = self.conf['graph_dataset']['list_p']
        self.Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        self.list_d = self.conf['graph_dataset']['list_degree']

        self.dataset = None

        if self.conf['graph_dataset']['ERmodel']:
            self.type = GraphType.ER

        elif self.conf['graph_dataset']['regular']:
            self.type = GraphType.Regular

        elif self.conf['graph_dataset']['confmodel']:
            self.type = GraphType.CM

    # def make_dataset(self):
    #     switcher = {
    #         GraphType.ER: " This is Case Zero ",
    #         GraphType.Regular: " This is Case One ",
    #         GraphType.CM: " This is Case Two ",
    #     }
    #     func_name = switcher.get(self.type, "nothing")
    #     create_dataset_function = getattr(self, func_name)
    #     create_dataset_function()

    def hot_encoding(self, array_of_distinct):
        """
        :param array_of_distinct:
        :return: una matrice dove ogni riga è l'encoded corrispondente alla riga di array_of_distinct
        """
        lista = []
        for i in array_of_distinct:
            lista.append([i])
        encoded = OneHotEncoder().fit_transform(lista).toarray()
        return encoded

    def initialize_dataset(self):
        print('Generating dataset...')
        modo = self.config_class.modo
        dataset = None
        if self.type == GraphType.ER:
            if modo == TrainingMode.mode1 or modo == TrainingMode.mode2:
                dataset = self.dataset_nclass_ER()
            if modo == TrainingMode.mode3:
                dataset = self.dataset_regression_ER()

        elif self.type == GraphType.Regular:
            dataset = self.dataset_regular()

        elif self.type == GraphType.CM:
            dataset = self.dataset_classification_CM()

        print("Dataset generated")
        return dataset

    def create_ER(self, Num_nodes, p, N_graphs):
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

        self.info_connectivity(grafi, p)

        return grafi, actual_probs

    def info_connectivity(self, grafi, p):
        print("Mean connectivity for each node:", end=' ')
        print(round(np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() for t in grafi]).mean(), 3), end=' ')
        print(f'p={p}')

    def create_regular(self, Num_nodes, d, N_graphs):
        """
            Num_nodes:  Graph nodes number
            d: Degree of each node
            N_graphs: quanti grafi restituire
            """
        grafi = []
        with Pool(processes=32) as pool:
            input_list = zip([d] * N_graphs, [Num_nodes] * N_graphs)
            grafi = pool.map(self.dummy_nx_random_reg_graph, input_list)
        # for i in range(N_graphs):
        #     gr = nx.random_regular_graph(d=d, n=Num_nodes)
        #     grafi.append(gr)

        self.info_connectivity(grafi, d)
        return grafi

    def dummy_nx_random_reg_graph(self, d_n):
        d, n = d_n
        return nx.random_regular_graph(d=d, n=n)

    def create_confmodel(self, Num_nodes, N_graphs, exponent=-1):
        grafi_actual_degrees = []
        with Pool(processes=32) as pool:
            input_list = zip([Num_nodes]*N_graphs, [exponent]*N_graphs)
            grafi_actual_degrees = pool.map(self.build_cm_graph, input_list)
        grafi_actual_degrees = np.array(grafi_actual_degrees)
        grafi = grafi_actual_degrees[:,0]
        actual_degrees = grafi_actual_degrees[:,1]
        #print(grafi, len(grafi), type(grafi))
        #print(actual_degrees, len(actual_degrees), type(actual_degrees))
        print(f"Nodi rimanenti in media: {np.array([len(gr.nodes()) for gr in grafi]).mean()}")
        return grafi, actual_degrees

    def build_cm_graph(self, Num_nodes_exponent):
        Num_nodes, exponent = Num_nodes_exponent
        s = rndm(1, 100, exponent, Num_nodes)
        s = np.array(s, int)
        if s.sum() % 2 != 0:
            s[0] += 1
        gr = nx.configuration_model(s)
        # check for graphical sequence:
        gr = nx.Graph(gr)  # remove multiple edges
        gr.remove_edges_from(nx.selfloop_edges(gr))  # remove self loops
        # print(f"Nodi iniziali: {Num_nodes}")
        # tengo solo la giant component
        Gcc = sorted(nx.connected_components(gr), key=len, reverse=True)
        gr0 = gr.subgraph(Gcc[0]).copy()
        gr.clear()
        # print(f"Nodi rimanenti: {len(gr0.nodes())}")
        degree = [d for v, d in gr0.degree()]
        return gr0, degree

    def dataset_nclass_ER(self):
        N = self.conf['graph_dataset']['Num_nodes']
        list_p = self.conf['graph_dataset']['list_p']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']

        dataset_grafi_nx = []
        dataset_labels = []
        label_kind = self.config_class.get_mode()['labels']

        if label_kind == Labels.onehot:
            n_classi = len(list_p)
            onehot_matrix = np.eye(n_classi)
            for i, p in enumerate(list_p):
                grafi_p, _ = self.create_ER(N, p, Num_grafi_per_tipo)
                dataset_grafi_nx = dataset_grafi_nx + grafi_p
                dataset_labels = dataset_labels + [onehot_matrix[i]] * len(grafi_p)
                #print(dataset_labels)

        elif label_kind == Labels.zero_one:
            # ho quindi solo due classi
            for i, p in enumerate(list_p):
                grafi_p, _ = self.create_ER(N, p, Num_grafi_per_tipo)
                dataset_grafi_nx = dataset_grafi_nx + grafi_p
                dataset_labels = dataset_labels + [i] * len(grafi_p)

        elif label_kind == Labels.prob:
            for i, p in enumerate(list_p):
                grafi_p, actual_probs = self.create_ER(N, p, Num_grafi_per_tipo)
                dataset_grafi_nx = dataset_grafi_nx + grafi_p
                dataset_labels.extend(actual_probs)

        self.dataset = GeneralDataset(dataset_grafi_nx, np.array(dataset_labels))
        return self.dataset

    def dataset_regression_ER(self):
        Num_nodes = self.conf['graph_dataset']['Num_nodes']
        range_p = self.conf['graph_dataset']['range_p']
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_p = self.conf['graph_dataset']['list_p']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        is_continuous_distribution = self.conf['graph_dataset']['continuous_p']

        dataset_grafi_nx = []
        dataset_labels = []
        original_class = []  # used for recovering original class other than label for regression in discrete distrib.
        assert self.conf['model']['neurons_per_layer'][-1] == 1

        if is_continuous_distribution:
            probs = np.random.uniform(low=range_p[0], high=range_p[1], size=Num_grafi_tot)
            for p in probs:
                gr = nx.erdos_renyi_graph(Num_nodes, p)
                dataset_grafi_nx.append(gr)
                actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1)
                dataset_labels.append(actual_p)
        else:
            for i, p in enumerate(list_p):
                grafi_p, actual_probs = self.create_ER(Num_nodes, p, Num_grafi_per_tipo)
                dataset_grafi_nx = dataset_grafi_nx + grafi_p
                dataset_labels.extend(actual_probs)
                original_class = original_class + [p]*len(actual_probs)

        self.dataset = GeneralDataset(dataset_grafi_nx, np.array(dataset_labels), original_class=original_class)

        return self.dataset

    def dataset_regular(self):
        N = self.conf['graph_dataset']['Num_nodes']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        list_degree = self.conf['graph_dataset']['list_degree']
        list_degree = [int(i) for i in list_degree]

        dataset_grafi_nx = []
        dataset_labels = []
        encoded = self.hot_encoding(list_degree)
        n_classi = len(list_degree)
        onehot_matrix = np.eye(n_classi)
        #assert (onehot_matrix == encoded).all(), "Errore nella costruzione del 1-hot encoding"
        for i, d in enumerate(list_degree):
            grafi_d = self.create_regular(N, d, Num_grafi_per_tipo)
            dataset_grafi_nx = dataset_grafi_nx + grafi_d
            dataset_labels = dataset_labels + [encoded[i]] * len(grafi_d)

        self.dataset = GeneralDataset(dataset_grafi_nx, np.array(dataset_labels))
        return self.dataset

    def dataset_classification_CM(self):
        N = self.conf['graph_dataset']['Num_nodes']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_exp = self.conf['graph_dataset']['list_exponents']
        list_exp = [int(i) for i in list_exp]
        encoded = self.hot_encoding(list_exp)

        dataset_grafi_nx = []
        dataset_labels = []
        dataset_degree_seq = []
        for i, exp in enumerate(list_exp):
            if self.numnodes_islist:
                num_nodes = N[i]
            else:
                num_nodes = N
            grafi, actual_degrees = self.create_confmodel(num_nodes, Num_grafi_per_tipo, exponent=exp)
            grafi = grafi.tolist()
            actual_degrees = actual_degrees.tolist()
            dataset_grafi_nx = dataset_grafi_nx + grafi
            dataset_labels = dataset_labels + [encoded[i]] * len(grafi)
            dataset_degree_seq.extend(actual_degrees)

        self.dataset = GeneralDataset(dataset_grafi_nx, np.array(dataset_labels), original_class=dataset_degree_seq)
        return self.dataset



    def perturb_dataset(self, amount_sigma, verbose=False):
        dataset_list_perturbed = []
        with Pool(processes=32) as pool:
            l = len(self.dataset.dataset_list)
            input_list = zip(self.dataset.dataset_list, [amount_sigma]*l, [verbose]*l)
            #dataset_list_perturbed = [perturb_nx_graph(g, amount_sigma, verbose) for g in self.dataset.dataset_list]
            dataset_list_perturbed = pool.map(parallel_perturb_nx_graph, input_list)
        # con l'algoritmo che ho creato la connettività mi aumenta di 1 circa
        if verbose:
            print([nx.to_numpy_matrix(g).sum(axis=1).mean() for g in dataset_list_perturbed])
        self.dataset.dataset_list = dataset_list_perturbed


def perturb_np_array(np_array, p=10):
    """
    Perturba una matrice numpy con valori random da una bernoulliana
    le modifiche effettuate da 0 a 1 sono tante quante quelle da 1 a 0, in media
    :param np_array:
    :param p: la probabilità non normalizzata di ottenere 1 da una bernoulliana
    :return:
    """
    #print(f"Original mean: {np_array.sum(axis=1).mean()}")
    noise = np.random.normal(loc=0.5, scale=sigma, size=np_array.shape)
    flattened = noise.flatten()

    m1 = flattened > 1
    percent_great_than_1 = flattened[m1].shape[0] / flattened.shape[0]
    effective_noise1 = np.where(noise > 1, 1, 0)
    m2 = flattened < 0
    percent_less_than_1 = flattened[m2].shape[0] / flattened.shape[0]
    effective_noise2 = np.where(noise < 0, 0, 0)
    effective_noise = effective_noise1 + effective_noise2
    #print(round(effective_noise.sum() / effective_noise.size))

    #perturbed = np_array + effective_noise
    perturbed = np.logical_xor(np_array, effective_noise).astype(int)
    #perturbed_pos = np.where(perturbed > 1, 1, perturbed)
    #perturbed = np.where(perturbed_pos < 0, 0, perturbed_pos)
    #print(f"Perturbed mean: {perturbed.sum(axis=1).mean()}")

    # adesso dobbiamo renderla simmetrica
    perturb_tri = np.triu(perturbed)
    perturbed_final = perturb_tri + perturb_tri.T - np.diag(perturb_tri.diagonal())
    assert np.allclose(perturbed_final, perturbed_final.T, rtol=1e-05, atol=1e-08), "Errore: la matrice di adiacenza non è simmetrica"

    return perturbed_final, percent_great_than_1, percent_less_than_1, effective_noise


def perturb_nx_graph(nx_graph, sigma=0.5, verbose=False):
    np_arr = nx.to_numpy_array(nx_graph)
    perturbed_array, percent_great_than_1, percent_less_than_1, effective_noise = perturb_np_array(np_arr, sigma)
    graph_perturbed = nx.from_numpy_matrix(perturbed_array)
    if verbose:
        print(f'% valori > 1: {percent_great_than_1}, valori < -1 : {percent_less_than_1}')
    return graph_perturbed
def parallel_perturb_nx_graph(args):
    nx_graph, sigma , verbose = args
    return perturb_nx_graph(nx_graph, sigma, verbose)

def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b
    Secondo questa formula il vero esponente è g-1 quindi se voglio una power law con -3 devo passare g=-2
    ---> aggiungo quì 1. ma per 0 non è definita quindi aggiungo 0.99"""
    g = g + 0.99
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)


import matplotlib.pyplot as plt
def plott(flattened, sigma):
    count, bins, ignored = plt.hist(flattened, 300, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                   np.exp( - (bins - 0)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.show()
    return count, bins