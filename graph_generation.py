from enum import Enum
from functools import partial
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool

from config_valid import Labels, TrainingMode, GraphType
from Dataset import Dataset, GeneralDataset



class GenerateGraph():
    def __init__(self, config_class):
        self.config_class = config_class
        self.conf = self.config_class.conf
        self.N = self.conf['graph_dataset']['Num_nodes']
        #print(f"Num nodi conf: {self.N}")
        if isinstance(self.N, list):
            self.numnodes_islist = True
        self.list_p = self.conf['graph_dataset']['list_p']
        self.Num_grafi_per_tipo = int(self.conf['graph_dataset']['Num_grafi_per_tipo'])
        self.list_d = self.conf['graph_dataset']['list_degree']
        self.community_probs = self.conf['graph_dataset'].get('community_probs')

        self.dataset = None
        self.dataset_grafi_nx = []
        self.target_labels = []
        self.dataset_degree_seq = []
        self.scalar_label = []
        self.node_label = []

        self.graphtype = config_class.graphtype


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

    def initialize_dataset(self, parallel=True):
        print('Generating dataset...')
        modo = self.config_class.modo
        if self.graphtype == GraphType.ER:
            if modo == TrainingMode.mode1 or modo == TrainingMode.mode2:
                self.dataset_nclass_ER()
            if modo == TrainingMode.mode3:
                self.dataset_regression_ER()

        elif self.graphtype == GraphType.Regular:
            self.dataset_regular(parallel=parallel)

        elif self.graphtype == GraphType.CM:
            if modo == TrainingMode.mode1 or modo == TrainingMode.mode2:
                self.dataset_classification_CM(parallel=parallel)
            if modo == TrainingMode.mode3:
                self.dataset_regression_CM(parallel=parallel)

        elif self.graphtype == GraphType.SBM:
            if modo == TrainingMode.mode1 or modo == TrainingMode.mode2:
                self.dataset_nclass_SBM(parallel=parallel)

        print("Dataset generated")

    def create_ER(self, Num_nodes, p, N_graphs):
        """
        Num_nodes:  Graph nodes number
        p:  Probability for edge creation
        N_graphs: ogni tot abbiamo un tipo di grafi originato da una diversa distribuzione
        """
        grafi = []
        actual_probs = []
        actual_degrees = []
        for i in range(N_graphs):
            gr = nx.erdos_renyi_graph(Num_nodes, p)  # seed = 1
            #grafi.append(nx.to_numpy_matrix(gr))
            grafi.append(gr)
            #actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1) # VERSION
            actual_p = nx.to_numpy_array(gr).sum(axis=1).mean() / (Num_nodes - 1)
            actual_probs.append(actual_p)
            actual_degrees.append(gr.degree())

        self.info_connectivity(grafi, p)

        return grafi, actual_probs, actual_degrees

    def info_connectivity(self, grafi, p):
        print("Mean connectivity for each node:", end=' ')
        #print(round(np.array([nx.to_numpy_matrix(t).sum(axis=1).mean() for t in grafi]).mean(), 3), end=' ') # VERSION
        print(round(np.array([nx.to_numpy_array(t).sum(axis=1).mean() for t in grafi]).mean(), 3), end=' ')
        print(f'p={p}')

    def create_regular(self, Num_nodes, d, N_graphs, parallel=True):
        """
            Num_nodes:  Graph nodes number
            d: Degree of each node
            N_graphs: quanti grafi restituire
            """
        grafi = []
        if parallel:
            with Pool(processes=32) as pool:
                input_list = zip([d] * N_graphs, [Num_nodes] * N_graphs)
                grafi = pool.map(self.dummy_nx_random_reg_graph, input_list)
        else:
            for i in range(N_graphs):
                gr = nx.random_regular_graph(d=d, n=Num_nodes)
                grafi.append(gr)

        self.info_connectivity(grafi, d)
        return grafi

    def dummy_nx_random_reg_graph(self, d_n):
        d, n = d_n
        return nx.random_regular_graph(d=d, n=n)

    def create_confmodel(self, Num_nodes, N_graphs, exponent=-1, parallel=True):
        if parallel:
            with Pool(processes=32) as pool:
                input_list = zip([Num_nodes]*N_graphs, [exponent]*N_graphs)
                grafi_actual_degrees = pool.map(self.build_cm_graph, input_list)
                grafi = [gr[0] for gr in grafi_actual_degrees]
                actual_degrees = [gr[1] for gr in grafi_actual_degrees]
        else:
            grafi = []
            actual_degrees = []
            for n in range(N_graphs):
                gr0, degree = self.build_cm_graph((Num_nodes, exponent))
                grafi.append(gr0)
                actual_degrees.append(degree)

        #actual_degrees = grafi_actual_degrees[:,1]
        #print(grafi, len(grafi), type(grafi))
        #print(actual_degrees, len(actual_degrees), type(actual_degrees))
        #print(type(grafi))
        #print(f"Nodi rimanenti in media: {np.array([len(gr.nodes()) for gr in grafi]).mean()}")
        return grafi, actual_degrees

    def build_cm_graph(self, Num_nodes_exponent):
        Num_nodes, exponent = Num_nodes_exponent
        s = rndm(3, Num_nodes, exponent, Num_nodes)
        s = np.array(s, int)
        if s.sum() % 2 != 0:
            s[-1] += 1
        gr = nx.configuration_model(s)
        # check for graphical sequence:
        gr = nx.Graph(gr)  # remove multiple edges
        gr.remove_edges_from(nx.selfloop_edges(gr))  # remove self loops
        #print(f"Nodi iniziali: {Num_nodes}")
        # tengo solo la giant component
        Gcc = sorted(nx.connected_components(gr), key=len, reverse=True)
        gr0 = gr.subgraph(Gcc[0]).copy()
        gr.clear()
        #print(f"Nodi rimanenti: {len(gr0.nodes())}")
        #degree = [d for v, d in gr0.degree()]
        return gr0, gr0.degree()  # !!! quì ora voglio portarmi appresso anche il node ID

    def build_sbm(self, Num_nodes, community_probs, N_graphs, parallel=False):
        parallel = False
        actual_degrees = []
        if parallel:
            with Pool(processes=32) as pool:
                input_list = []
                for i in range(N_graphs):
                    input_list.append({"sizes": Num_nodes, "p": community_probs})
                #print(input_list[0])
                #nx.stochastic_block_model(**input_list[0])
                mapfunc = partial(nx.stochastic_block_model, input_list)
                grafi = pool.imap(nx.stochastic_block_model, input_list)
                #grafi = pool.starmap(nx.stochastic_block_model, list(input_list.items()))
            actual_degrees = [sbmg.degree() for sbmg in grafi]
        else:
            grafi = []
            for i in range(N_graphs):
                #print(community_probs)
                sbmg = nx.stochastic_block_model(Num_nodes, community_probs)
                grafi.append(sbmg)
                actual_degrees.append(sbmg.degree())
        return grafi, actual_degrees

    def dataset_nclass_ER(self):
        N = self.conf['graph_dataset']['Num_nodes']
        list_p = self.conf['graph_dataset']['list_p']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']


        label_kind = self.config_class.get_mode()['labels']

        if label_kind == Labels.onehot:
            n_classi = len(list_p)
            onehot_matrix = np.eye(n_classi)
            for i, p in enumerate(list_p):
                grafi_p, actual_p, actual_degrees = self.create_ER(N, p, Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels = self.target_labels + [onehot_matrix[i]] * len(grafi_p)
                self.scalar_label = self.scalar_label + [p] * Num_grafi_per_tipo
                self.node_label.extend([[p] * N] * Num_grafi_per_tipo)
                only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
                self.dataset_degree_seq.extend(only_degrees)
                #print(f"{set(self.node_label)} <-> {set(self.scalar_label)}")

        elif label_kind == Labels.zero_one:
            # ho quindi solo due classi
            for i, p in enumerate(list_p):
                grafi_p, actual_probs, actual_degrees = self.create_ER(N, p, Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels = self.target_labels + [i] * len(grafi_p)
                self.scalar_label = self.scalar_label + [p] * Num_grafi_per_tipo
                self.node_label.extend([[p] * N] * Num_grafi_per_tipo)
                only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
                self.dataset_degree_seq.extend(only_degrees)

        elif label_kind == Labels.prob:
            for i, p in enumerate(list_p):
                grafi_p, actual_probs, _ = self.create_ER(N, p, Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels.extend(actual_probs)
                self.scalar_label = self.scalar_label + [p] * Num_grafi_per_tipo
                self.node_label.extend([[p] * N] * Num_grafi_per_tipo)

        # TODO: original class lo facciamo diventare un target vettoriale
        original_class = [[i] for i in self.target_labels]
        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label)

    def dataset_regression_ER(self):
        Num_nodes = self.conf['graph_dataset']['Num_nodes']
        range_p = self.conf['graph_dataset']['range_p']
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_p = self.conf['graph_dataset']['list_p']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        is_continuous_distribution = self.conf['graph_dataset']['continuous_p']

        if is_continuous_distribution:
            probs = np.random.uniform(low=range_p[0], high=range_p[1], size=Num_grafi_tot)
            for p in probs:
                gr = nx.erdos_renyi_graph(Num_nodes, p)
                self.dataset_grafi_nx.append(gr)
                actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1)
                self.target_labels.append(actual_p)
                self.node_label.append([p] * Num_nodes)
                self.scalar_label.append(p)  # voglio tracciare la probabilità usata per generare il grafo
        else:
            for i, p in enumerate(list_p):
                grafi_p, actual_probs = self.create_ER(Num_nodes, p, Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels.extend(actual_probs)
                self.scalar_label = self.scalar_label + [p] * Num_grafi_per_tipo
                self.node_label.append([p]*Num_grafi_per_tipo*Num_nodes)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      scalar_label=self.scalar_label,
                                      node_label=self.node_label)


    def dataset_regular(self, parallel=True):
        N = self.conf['graph_dataset']['Num_nodes']
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        list_degree = self.conf['graph_dataset']['list_degree']
        list_degree = [int(i) for i in list_degree]

        if self.config_class.modo == TrainingMode.mode1:
            encoded = self.hot_encoding(list_degree)
        elif self.config_class.modo == TrainingMode.mode2:
            encoded = [0, 1]

        n_classi = len(list_degree)
        onehot_matrix = np.eye(n_classi)
        #assert (onehot_matrix == encoded).all(), "Errore nella costruzione del 1-hot encoding"
        for i, d in enumerate(list_degree):
            grafi_d = self.create_regular(N, d, Num_grafi_per_tipo, parallel=parallel)
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_d
            self.target_labels = self.target_labels + [encoded[i]] * len(grafi_d)

        original_class = [[i] for i in self.target_labels]
        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels), original_class=original_class)

    def dataset_classification_CM(self, parallel=True):
        list_exp = self.conf['graph_dataset']['list_exponents']
        list_exp = [float(i) for i in list_exp]
        if self.config_class.modo == TrainingMode.mode1:
            encoded = self.hot_encoding(list_exp)
        elif self.config_class.modo == TrainingMode.mode2:
            encoded = [0, 1]

        for i, exp in enumerate(list_exp):
            if self.numnodes_islist:
                num_nodes = int(self.N[i])
            else:
                num_nodes = int(self.N)
            grafi, actual_degrees = self.create_confmodel(num_nodes, self.Num_grafi_per_tipo, exponent=exp, parallel=parallel)
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi
            self.target_labels = self.target_labels + [encoded[i]] * len(grafi)
            self.node_label.extend([[exp] * num_nodes] * self.Num_grafi_per_tipo)
            # prima di aggiungere tolgo l'id dei nodi da questo array, per ora non mi serve
            # type nx.classes.reportviews.DegreeView
            only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
            self.dataset_degree_seq.extend(only_degrees)
            self.scalar_label = self.scalar_label + [exp] * len(grafi)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label,
                                      exponent=list(zip(self.scalar_label, self.target_labels)))

    def dataset_regression_CM(self, parallel=True):
        N = self.conf['graph_dataset']['Num_nodes']
        if self.numnodes_islist:
            Num_nodes = N[0]
        else:
            Num_nodes = N
        Num_grafi_per_tipo = self.conf['graph_dataset']['Num_grafi_per_tipo']
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_exp = self.conf['graph_dataset']['list_exponents']
        list_exp = [float(i) for i in list_exp]
        is_continuous_distribution = self.conf['graph_dataset']['continuous_p']

        if is_continuous_distribution:
            esponenti = np.random.uniform(low=list_exp[0], high=list_exp[1], size=Num_grafi_tot)
            if parallel:
                with Pool(processes=32) as pool:
                    input_list = zip([Num_nodes] * Num_grafi_tot, esponenti)
                    grafi_actual_degrees = pool.map(self.build_cm_graph, input_list)
                    self.dataset_grafi_nx = [gr[0] for gr in grafi_actual_degrees]
                    self.dataset_degree_seq = [dict(gr[1]).values() for gr in grafi_actual_degrees]
                self.target_labels = esponenti
            else:
                for e in esponenti:
                    gr, actual_degrees = self.build_cm_graph((Num_nodes, e))
                    self.dataset_grafi_nx.append(gr)
                    only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
                    self.dataset_degree_seq.append(only_degrees)
                    self.target_labels.append(e)
        else:
            for i, p in enumerate(list_exp):  # TODO: perché createER? sostituire
                grafi_p, actual_probs = self.create_ER(Num_nodes, p, Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels.extend(actual_probs)
                original_class = original_class + [p]*len(actual_probs)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels), original_class=self.dataset_degree_seq)


    def dataset_nclass_SBM(self, parallel=True):
        # quì N deve essere una lista
        # quì list_p deve essere una lista di liste

        label_kind = self.config_class.modo['labels']
        if label_kind == Labels.onehot:
            labels = np.eye(self.config_class.num_classes)
        elif label_kind == Labels.zero_one:
            labels = [0,1]
        elif label_kind == Labels.prob:
            labels = self.list_p
        else:
            labels = None

        for i, matrix_p in enumerate(self.community_probs):
            # al momento Num_nodes è una lista di due, una per comunnità, per ogni classe sono previste due e solo due comunità
            #community_probs = self.make_planted_matrix(communities)
            grafi, actual_degrees = self.build_sbm(self.N, matrix_p, self.Num_grafi_per_tipo, parallel)
            #print(f"actual_degrees shape : {np.array(actual_degrees).shape}")
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi
            self.target_labels = self.target_labels + [labels[i]] * len(grafi)
            self.scalar_label = self.scalar_label + [matrix_p] * self.Num_grafi_per_tipo
            self.node_label.extend([[matrix_p] * self.N[0]] * self.Num_grafi_per_tipo)  # aggiungo per la prima comunità
            self.node_label.extend([[matrix_p] * self.N[1]] * self.Num_grafi_per_tipo)  # per la seconda
            # actual degrees: tolgo l'id...l'ho fatto in mille modi diversi :(
            only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
            self.dataset_degree_seq.extend(only_degrees)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label)
    def make_planted_matrix(self, a):
        mat = [a, a[::-1]]
        return mat
    def perturb_dataset(self, amount_p, verbose=False):
        dataset_list_perturbed = []
        with Pool(processes=32) as pool:
            l = len(self.dataset.dataset_list)
            input_list = zip(self.dataset.dataset_list, [amount_p] * l, [verbose] * l)
            #dataset_list_perturbed = [perturb_nx_graph(g, amount_sigma, verbose) for g in self.dataset.dataset_list]
            dataset_list_perturbed = pool.map(parallel_perturb_nx_graph, input_list)
        # con l'algoritmo che ho creato la connettività mi aumenta di 1 circa
        if verbose:
            print([nx.to_numpy_matrix(g).sum(axis=1).mean() for g in dataset_list_perturbed])
        self.dataset.dataset_list = dataset_list_perturbed


def perturb_np_array(np_array, p=10, verbose=False):
    """
    Perturba una matrice numpy con valori random da una bernoulliana
    le modifiche effettuate da 0 a 1 sono tante quante quelle da 1 a 0, in media
    :param np_array:
    :param p: la probabilità non normalizzata di ottenere 1 da una bernoulliana
    :return:
    """
    print(f"Original mean degree: {np_array.sum(axis=1).mean()}")
    uni = np_array == 1
    zeri = np_array == 0
    if verbose: print(f"1: {np_array[uni].sum()}, 0:{np.logical_not(np_array[zeri]).sum()}")
    s_create = np.random.binomial(1, p / zeri.sum(), zeri.sum())
    s_delete = np.random.binomial(1, p / uni.sum(), uni.sum())
    if verbose: print(f"elementi che saranno aggiunti: {s_create.sum()}, elementi che saranno tolti: {s_delete.sum()}")

    # prendo gli 1 e li cambio in base ai True(1) in s_delete con XOR
    np_array[uni] = np.logical_xor(np_array[uni], s_delete).astype(int)
    # prendo gli 0 e li cambio in base ai True(1) in s_create con XOR
    np_array[zeri] = np.logical_xor(np_array[zeri], s_create).astype(int)
    if verbose: print(f"Nuovi 1: {np_array[uni].sum()}, 0:{np.logical_not(np_array[zeri]).sum()}")

    if verbose: print(f"Perturbed mean: {np_array.sum(axis=1).mean()}")

    # adesso dobbiamo renderla simmetrica
    perturb_tri = np.triu(np_array)
    perturbed_final = perturb_tri + perturb_tri.T - np.diag(perturb_tri.diagonal())
    assert np.allclose(perturbed_final, perturbed_final.T, rtol=1e-05, atol=1e-08), "Errore: la matrice di adiacenza non è simmetrica"
    print(f"Perturbed mean: {np_array.sum(axis=1).mean()}")
    return perturbed_final


def perturb_nx_graph(nx_graph, p=100, verbose=False):
    np_arr = nx.to_numpy_array(nx_graph)
    perturbed_array = perturb_np_array(np_arr, p, verbose)
    graph_perturbed = nx.from_numpy_matrix(perturbed_array)
    return graph_perturbed
def parallel_perturb_nx_graph(args):
    nx_graph, p , verbose = args
    return perturb_nx_graph(nx_graph, p, verbose)

def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b
    Secondo questa formula il vero esponente è g-1 quindi se voglio una power law con -3 devo passare g=-2
    ---> aggiungo quì 1. ma per 0 non è definita quindi aggiungo 0.99"""
    g = g + 1.0
    #print(f"a: {a}, b: {b}, g: {g}")
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)

def powerlaw_dist(x0, x1, n, size):
    """
    da: https://stackoverflow.com/questions/918736/random-number-generator-that-produces-a-power-law-distribution
    y is a uniform variate,
    n is the distribution power,
    x0 and x1 define the range of the distribution,
    x is power-law distributed variate
    x = [(x1 ^ (n + 1) - x0 ^ (n + 1)) * y + x0 ^ (n + 1)] ^ (1 / (n + 1))
    :return:
    """
    n = n + 1
    y = np.random.random(size=size)
    ((x1**n - x0**n) * y + x0**n) ** (1. / n)  # risulta la stessa della funzione rndm

import matplotlib.pyplot as plt
def plott(flattened, sigma):
    count, bins, ignored = plt.hist(flattened, 300, density=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                   np.exp( - (bins - 0)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
    plt.show()
    return count, bins




        