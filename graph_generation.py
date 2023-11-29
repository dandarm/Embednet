from enum import Enum
from functools import partial
import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from multiprocessing import Pool

from config_valid import Labels, TrainingMode, GraphType
from Dataset import Dataset, GeneralDataset

from NEMtropy import UndirectedGraph
from NEMtropy.network_functions import build_adjacency_from_edgelist as list2adjs



class GenerateGraph():
    def __init__(self, config_class, verbose=False):
        self.config_class = config_class
        self.conf = self.config_class.conf
        self.verbose = verbose
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
        self.actual_ERprobs = []
        self.dataset_degree_seq = []
        self.dataset_cluster_coeff = []
        self.scalar_label = []
        self.node_label = []

        self.graphtype = config_class.graphtype

        self._num_nodes_per_graph = None


    @property
    def num_nodes_per_graph(self):
        if self._num_nodes_per_graph is None:
            self._num_nodes_per_graph = [self.N] * len(self.dataset_grafi_nx)
        return self._num_nodes_per_graph
    @num_nodes_per_graph.setter
    def num_nodes_per_graph(self, list_value):
        #if not self.config_class.conf['graph_dataset'].get('real_dataset'):
        self._num_nodes_per_graph = list_value

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

        elif self.graphtype == GraphType.CONST_DEG_DIST:
            self.dataset_CONST_DEG_DIST(parallel=parallel)

        print("Dataset generated")

# region build single graph
    def create_ER(self, Num_nodes, p, N_graphs):
        """
        Num_nodes:  Graph nodes number
        p:  Probability for edge creation
        N_graphs: ogni tot abbiamo un tipo di grafi originato da una diversa distribuzione
        """
        grafi = []
        actual_probs = []
        actual_degrees = []
        actual_clust_coeff = []
        N_graphs = int(N_graphs)
        for i in range(N_graphs):
            gr = nx.erdos_renyi_graph(Num_nodes, p)  # seed = 1
            #grafi.append(nx.to_numpy_matrix(gr))
            grafi.append(gr)
            #actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (Num_nodes - 1) # VERSION
            actual_p = nx.to_numpy_array(gr).sum(axis=1).mean() / (Num_nodes - 1)
            actual_probs.append(actual_p)
            actual_degrees.append(gr.degree())
            actual_clust_coeff.append(nx.clustering(gr))

        if self.verbose: self.info_connectivity(grafi, p)

        return grafi, actual_probs, actual_degrees, actual_clust_coeff

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
            il prodotto Num_nodes * d deve essere pari, altrimenti il grafo non esiste
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

        if self.verbose: self.info_connectivity(grafi, d)
        return grafi

    def dummy_nx_random_reg_graph(self, d_n):
        d, n = d_n
        return nx.random_regular_graph(d=d, n=n)

    def create_confmodel(self, Num_nodes, N_graphs, exponent=-1, parallel=True):
        if parallel:
            with Pool(processes=32) as pool:
                input_list = zip([Num_nodes]*N_graphs, [exponent]*N_graphs)
                grafi_actual_degrees_clustcoeff = pool.map(self.build_cm_graph, input_list)
                grafi = [gr[0] for gr in grafi_actual_degrees_clustcoeff]
                actual_degrees = [gr[1] for gr in grafi_actual_degrees_clustcoeff]
                actual_clust_coeff = [gr[2] for gr in grafi_actual_degrees_clustcoeff]
        else:
            grafi = []
            actual_degrees = []
            actual_clust_coeff = []
            for n in range(N_graphs):
                gr0, degree, cc = self.build_cm_graph((Num_nodes, exponent))
                grafi.append(gr0)
                actual_degrees.append(degree)
                actual_clust_coeff.append(cc)

        #actual_degrees = grafi_actual_degrees[:,1]
        #print(grafi, len(grafi), type(grafi))
        #print(actual_degrees, len(actual_degrees), type(actual_degrees))
        #print(type(grafi))
        #print(f"Nodi rimanenti in media: {np.array([len(gr.nodes()) for gr in grafi]).mean()}")
        return grafi, actual_degrees, actual_clust_coeff

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
        cc = nx.clustering(gr0)
        #print(f"Nodi rimanenti: {len(gr0.nodes())}")
        #degree = [d for v, d in gr0.degree()]
        return gr0, gr0.degree(), cc

    def get_starting_matrix(self, Num_nodes, exponent):
        # calcola p_ij che è la matrice di partenza che usa poi NEMtropy per generare il dataset
        s = rndm(3, Num_nodes, exponent, Num_nodes)
        x = np.array(s)  #TODO: quì non devo rendere i valori interi?
        ps = x * x[:, np.newaxis]
        p_ij = ps / (1 + ps)
        return p_ij

    def create_confmodel_nemtropy(self, Num_nodes, N_graphs, exponent=-2, parallel=True):
        p_ij = self.get_starting_matrix(Num_nodes, exponent)

        graph = UndirectedGraph(p_ij)
        graph.solve_tool(model="cm_exp", method="newton", initial_guess="random")
        res = graph.ensemble_sampler(N_graphs, cpu_n=32, output_dir="None")
        adjs = [list2adjs(np.array(r), is_directed=False) for r in res]
        grafi = [nx.from_numpy_array(a) for a in adjs]

        return grafi, p_ij

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

# endregion

# build dataset_list and GeneralDataset class
    def dataset_nclass_ER(self):
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']
        list_p = self.conf['graph_dataset']['list_p']

        for i, p in enumerate(list_p):
            N = nodes_per_class[i]
            grafi_p, actual_probs, actual_degrees, actual_clust_coeff = self.create_ER(N, p, self.Num_grafi_per_tipo)
            self.dataset_grafi_nx.extend(grafi_p)
            only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
            only_ccs = [list(dict(dw).values()) for dw in actual_clust_coeff]
            self.dataset_degree_seq.extend(only_degrees)
            self.dataset_cluster_coeff.extend(only_ccs)
            self.actual_ERprobs.extend(actual_probs)
            self.node_label.extend([[p] * N] * self.Num_grafi_per_tipo)
            # shape: Num_per_tipo*len(list_p) X N
            self.scalar_label = self.scalar_label + [p] * self.Num_grafi_per_tipo

        # definisco le label in 3 modi diversi a seconda del caso
        label_kind = self.config_class.get_mode()['labels']
        if label_kind == Labels.onehot:
            n_classi = len(list_p)
            onehot_matrix = np.eye(n_classi)
            for i, p in enumerate(list_p):
                self.target_labels = self.target_labels + [onehot_matrix[i]] * self.Num_grafi_per_tipo

        elif label_kind == Labels.zero_one:  # ho quindi solo due classi
            for i, p in enumerate(list_p):
                self.target_labels = self.target_labels + [i] * self.Num_grafi_per_tipo

        elif label_kind == Labels.prob:
            self.target_labels = self.actual_ERprobs

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label,
                                      actual_cluster_coeff=self.dataset_cluster_coeff)

    def dataset_regression_ER(self):
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']
        range_p = self.conf['graph_dataset']['range_p']
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_p = self.conf['graph_dataset']['list_p']
        is_continuous_distribution = self.conf['graph_dataset']['continuous_p']

        if is_continuous_distribution:
            probs = np.random.uniform(low=range_p[0], high=range_p[1], size=Num_grafi_tot)
            for i, p in enumerate(probs):
                N = nodes_per_class[i]
                gr = nx.erdos_renyi_graph(N, p)
                self.dataset_grafi_nx.append(gr)
                actual_p = nx.to_numpy_matrix(gr).sum(axis=1).mean() / (N - 1)
                self.target_labels.append(actual_p)
                self.node_label.append([p] * N)
                self.scalar_label.append(p)  # voglio tracciare la probabilità usata per generare il grafo
        else:
            for i, p in enumerate(list_p):
                N = nodes_per_class[i]
                grafi_p, actual_probs, _, _ = self.create_ER(N, p, self.Num_grafi_per_tipo)
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels.extend(actual_probs)
                self.scalar_label = self.scalar_label + [p] * self.Num_grafi_per_tipo
                self.node_label.append([p]*self.Num_grafi_per_tipo*N)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      scalar_label=self.scalar_label,
                                      node_label=self.node_label)


    def dataset_regular(self, parallel=True):
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']
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
            N = nodes_per_class[i]
            grafi_d = self.create_regular(N, d, self.Num_grafi_per_tipo, parallel=parallel)
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_d
            self.target_labels = self.target_labels + [encoded[i]] * len(grafi_d)
            self.node_label.extend([[d] * N] * self.Num_grafi_per_tipo)
            self.scalar_label = self.scalar_label + [d] * self.Num_grafi_per_tipo
            only_degrees = [list(dict(gr.degree()).values()) for gr in grafi_d]
            self.dataset_degree_seq.extend(only_degrees)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label)
        #print(f"ci siamo?")

    def dataset_classification_CM(self, parallel=True):
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']
        list_exp = self.conf['graph_dataset']['list_exponents']
        list_exp = [float(i) for i in list_exp]
        if self.config_class.modo == TrainingMode.mode1:
            encoded = self.hot_encoding(list_exp)
        elif self.config_class.modo == TrainingMode.mode2:
            encoded = [0, 1]

        for i, exp in enumerate(list_exp): #
            #if self.numnodes_islist:
            #    num_nodes = int(self.N[i])
            #else:
            #    num_nodes = int(self.N)
            N = nodes_per_class[i]
            #grafi, actual_degrees, actual_clust_coeff = self.create_confmodel(N, self.Num_grafi_per_tipo, exponent=exp, parallel=parallel)
            grafi, p_ij = self.create_confmodel_nemtropy(N, self.Num_grafi_per_tipo, exponent=exp)
            #p_ij=0
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi
            self.target_labels = self.target_labels + [encoded[i]] * len(grafi)
            self.node_label.extend([[exp] * N] * self.Num_grafi_per_tipo)
            # prima di aggiungere tolgo l'id dei nodi da questo array, per ora non mi serve
            # type nx.classes.reportviews.DegreeView
            actual_degrees = [g.degrees() for g in grafi]
            only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
            actual_clust_coeff = [nx.clustering(g) for g in grafi]
            only_ccs = [list(dict(dw).values()) for dw in actual_clust_coeff]
            self.dataset_degree_seq.extend(only_degrees)
            self.dataset_cluster_coeff.extend(only_ccs)
            self.scalar_label = self.scalar_label + [exp] * self.Num_grafi_per_tipo

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label,
                                      exponent=list(zip(self.scalar_label, self.target_labels)),
                                      p_ij=p_ij,
                                      actual_cluster_coeff=self.dataset_cluster_coeff)

    def dataset_regression_CM(self, parallel=True):
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']
        #if self.numnodes_islist:
        #    Num_nodes = N[0]
        #else:
        #    Num_nodes = N
        Num_grafi_tot = self.conf['graph_dataset']['Num_grafi_totali']
        list_exp = self.conf['graph_dataset']['list_exponents']
        list_exp = [float(i) for i in list_exp]
        is_continuous_distribution = self.conf['graph_dataset']['continuous_p']

        if is_continuous_distribution:
            esponenti = np.random.uniform(low=list_exp[0], high=list_exp[1], size=Num_grafi_tot)
            N = nodes_per_class[0]
            if parallel:
                with Pool(processes=32) as pool:
                    input_list = zip([N] * Num_grafi_tot, esponenti)
                    grafi_actual_degrees = pool.map(self.build_cm_graph, input_list)
                    self.dataset_grafi_nx = [gr[0] for gr in grafi_actual_degrees]
                    self.dataset_degree_seq = [dict(gr[1]).values() for gr in grafi_actual_degrees]
                self.target_labels = esponenti
            else:
                for e in esponenti:
                    gr, actual_degrees = self.build_cm_graph((N, e))
                    self.dataset_grafi_nx.append(gr)
                    only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
                    self.dataset_degree_seq.append(only_degrees)
                    self.target_labels.append(e)
        else:
            for i, e in enumerate(list_exp):  # TODO: perché createER? sostituire
                N = nodes_per_class[i]
                grafi_p, actual_probs = self.build_cm_graph((N, e))
                self.dataset_grafi_nx = self.dataset_grafi_nx + grafi_p
                self.target_labels.extend(actual_probs)
                original_class = original_class + [e]*len(actual_probs)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels), original_class=self.dataset_degree_seq)


    def dataset_nclass_SBM(self, parallel=True):
        # quì N deve essere una lista
        # quì list_p deve essere una lista di liste
        nodes_per_class = self.conf['graph_dataset']['Num_nodes']

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
            N = nodes_per_class[i]
            # al momento Num_nodes è una lista di due, una per comunnità, per ogni classe sono previste due e solo due comunità
            #community_probs = self.make_planted_matrix(communities)
            grafi, actual_degrees = self.build_sbm(N, matrix_p, self.Num_grafi_per_tipo, parallel)
            #print(f"actual_degrees shape : {np.array(actual_degrees).shape}")
            self.dataset_grafi_nx = self.dataset_grafi_nx + grafi
            self.target_labels = self.target_labels + [labels[i]] * len(grafi)
            self.scalar_label = self.scalar_label + [matrix_p] * self.Num_grafi_per_tipo
            self.node_label.extend([[matrix_p] * N[0]] * self.Num_grafi_per_tipo)  # aggiungo per la prima comunità
            self.node_label.extend([[matrix_p] * N[1]] * self.Num_grafi_per_tipo)  # per la seconda
            # actual degrees: tolgo l'id...l'ho fatto in mille modi diversi :(
            only_degrees = [list(dict(dw).values()) for dw in actual_degrees]
            self.dataset_degree_seq.extend(only_degrees)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array(self.target_labels),
                                      original_node_class=self.node_label,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=self.scalar_label)

    def dataset_CONST_DEG_DIST(self, parallel):
        grafi = []
        degrees = []
        i = 0
        #for _ in range(self.Num_grafi_per_tipo*2):
        while len(grafi) < self.Num_grafi_per_tipo:
            # ho un solo tipo quì! faccio per 2 perché in media la metà viene scartata
            r = np.random.randint(1, 50, self.N[0])
            if nx.is_graphical(r):
                gcm = nx.configuration_model(r)
                grcm = nx.Graph(gcm)  # remove multiple edges
                grcm.remove_edges_from(nx.selfloop_edges(grcm))  # remove self loops
                gccmm = sorted(nx.connected_components(grcm), key=len, reverse=True)
                grcm0 = grcm.subgraph(gccmm[0]).copy()
                if len(grcm0.nodes()) == self.N[0]:
                    grafi.append(grcm0)
                    degrees.append(np.array(list(dict(grcm0.degree()).values())))
                    i += 1
            self.dataset_grafi_nx = grafi

        print(f"Generati {i} grafi")

        self.dataset = GeneralDataset(self.dataset_grafi_nx, np.array([0]*len(grafi)),
                                      original_node_class=degrees,
                                      actual_node_class=degrees,
                                      scalar_label=np.array([0]*len(grafi)))



# endregion

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




class GenerateGraph_from_numpyarray():
    """
    Genera grafi oggetto di networkx a partire da matrici di adiacenza come numpy arrays
    """
    def __init__(self, config_class, np_arrays):
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.np_arrays = np_arrays

        self.dataset = None
        self.dataset_grafi_nx = []
        self.target_labels = []
        self.dataset_degree_seq = []
        self.scalar_label = []
        self.node_label = []

        self._num_nodes_per_graph = None

    @property
    def num_nodes_per_graph(self):
        if self._num_nodes_per_graph is None:
             self._num_nodes_per_graph = [a.shape[0] for a in self.np_arrays]

        #if isinstance(self.num_nodes_per_graph, list):
        #    return self.num_nodes_per_graph[0]
        #else:
        return self._num_nodes_per_graph

    @num_nodes_per_graph.setter
    def num_nodes_per_graph(self, value):
    #    self.num_nodes_per_graph = [a.shape[0] for a in self.np_arrays]
         self._num_nodes_per_graph = value

    def create_nx_graphs(self):
        """
        np_arrays è la lista di matrici di adiacenza
        :param np_arrays:
        :return:
        """
        for a in self.np_arrays:
            gr = nx.from_numpy_array(a)
            self.dataset_grafi_nx.append(gr)
            actual_degrees = gr.degree()
            only_degrees = list(dict(actual_degrees).values())
            self.dataset_degree_seq.append(only_degrees)

        self.dataset = GeneralDataset(self.dataset_grafi_nx, labels=[1]*len(self.dataset_grafi_nx),
                                      original_node_class=None,
                                      actual_node_class=self.dataset_degree_seq,
                                      scalar_label=None
                                      )

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

#import matplotlib.pyplot as plt
# def plott(flattened, sigma):
#     count, bins, ignored = plt.hist(flattened, 300, density=True)
#     plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                    np.exp( - (bins - 0)**2 / (2 * sigma**2) ),
#              linewidth=2, color='r')
#     plt.show()
#     return count, bins




        