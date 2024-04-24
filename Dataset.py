import random
from collections import Counter
from time import time
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from multiprocessing import Pool
import itertools
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use('Qt5Agg')

class SingleGraph():
    def __init__(self, nx_graph, graph_label, node_labels):
        self.nx_graph = nx_graph
        self.graph_label = graph_label
        self.node_labels = node_labels
# TODO: sarebbe il caso di mettere la classe embedding dentro al dataset?

class GeneralDataset:
    def __init__(self, dataset_list, labels, **kwarg):
        self.dataset_list = dataset_list
        self.labels = labels
        self.original_node_class = kwarg.get('original_node_class')
        self.actual_node_class = kwarg.get('actual_node_class')
        self.exponent = kwarg.get('exponent')
        self.scalar_label = kwarg.get('scalar_label')

        self.p_ij = kwarg.get('p_ij')

        self.actual_cluster_coeff = kwarg.get('actual_cluster_coeff')
        self.actual_knn = kwarg.get('actual_knn')

        # devo ricalcolare il num_nodes_per_graph
        self._num_nodes_per_graph = None

    @property
    def num_nodes_per_graph(self):
        if self._num_nodes_per_graph is None:
            self._num_nodes_per_graph = [len(data.x) for data in self.dataset_pyg]
        return self._num_nodes_per_graph

        #calcola il max degree
        #[dataset_list]

class Dataset(GeneralDataset):

    # def __init__(self, config_class, dataset_list, labels, original_node_class, exponent, actual_node_class, scalar_label,_num_nodes_per_graph, verbose):
    #     super().__init__(dataset_list, labels,
    #                      original_node_class=original_node_class,
    #                      exponent=exponent,
    #                      actual_node_class=actual_node_class,
    #                      scalar_label=scalar_label,
    #                      _num_nodes_per_graph=_num_nodes_per_graph)
    def __init__(self, config_class, **kwarg):
        super().__init__(**kwarg)

        self.config_class = config_class
        self.conf = config_class.conf
        self.percentage_train = self.conf['training']['percentage_train']
        self.bs = self.conf['training']['batch_size']
        self.device = config_class.device
        self.verbose = kwarg.get('verbose')

        self.dataset_pyg = None
        self.len_data = len(self.dataset_list)
        self.tt_split = int(self.len_data * self.percentage_train)
        self.train_dataset = None
        self.train_len = None
        self.test_dataset = None
        self.test_len = None

        self.train_loader = None
        self.test_loader = None

        self.last_neurons = self.config_class.lastneuron
        self.all_data_loader = None

        self.num_grafi = int(self.conf['graph_dataset']['Num_grafi_per_tipo'])
        self.num_nodi = int(self.conf['graph_dataset']['Num_nodes'][0])

        if self.config_class.conf['model'].get('my_normalization_adj'):
            # calcolo le statistiche per la normalizzazione della matrice di adiacenza
            # che sarà contenuta dentro ogni elemento Data
            self.calc_degree_probabilities()


    def calc_degree_probabilities(self):
        # considero quì le statistiche per la normalizzazione della matrice di adiacenza
        # che ora viene calcolata prima e non on-the-fly durante il training
        flat_list = [x for l in self.actual_node_class for x in l]
        degree_count = Counter(flat_list)
        # Normalizzazione dei conteggi -> probabilità!
        total_count = sum(degree_count.values())
        self.degree_prob = {degree: count / total_count for degree, count in degree_count.items()}
        self.tot_links_per_graph = sum(flat_list) / 2 / self.num_grafi
        self.average_links_per_graph = sum(flat_list) / self.num_nodi / self.num_grafi

        #for key, value in self.degree_prob.items():
        #    self.degree_prob[key] = self.degree_prob[key].to(self.device)

    @classmethod
    def from_super_instance(cls, config_class, super_instance, verbose):
        return cls(config_class, **super_instance.__dict__, verbose=verbose)

    def convert_G(self, g_i):
        g, i = g_i
        # aggiungo i metadati x e y per l'oggetto Data di PYG
        nodi = list(g.nodes)
        for n in nodi:
            g.nodes[n]["x"] = [1.0]
            g.nodes[n]["id"] = [n]

        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        #if len(type_graph) == 1:
        #    type_graph = [type_graph]
        #print(f'type_graph {type_graph}')
        # if self.config.modo == TrainingMode.mode1 or self.config.modo == TrainingMode.mode2:
        #     tipo = torch.long
        # if self.last_neurons == 1:  # TODO: cambiare anche qui
        #     tipo = torch.float
        # else:
        #     tipo = torch.long
        #
        # if self.config['graph_dataset']['regular']:

        tipo = torch.float
        pyg_graph.y = torch.tensor(np.array([type_graph]), dtype=tipo)
        #print(pyg_graph.y)

        return pyg_graph

    def convert_G_random_feature(self, g_i):
        g, i = g_i
        nodi = list(g.nodes)
        for n in nodi:
            r = np.random.randn() + 1.0
            g.nodes[n]["x"] = [r]
        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        pyg_graph.y = torch.tensor([type_graph], dtype=torch.float)
        return pyg_graph

    def convert_G_autoencoder(self, g_i):
        pyg_graph = self.convert_G(g_i)
        pyg_graph, _, _ = self.transform4ae(pyg_graph)
        return pyg_graph

    # def process_each(self, each_list):
    #     each_pyg = []
    #     for i, g in enumerate(each_list):
    #         if self.config['model']['autoencoder']:
    #             pyg_graph = self.convert_G_autoencoder((g, i))
    #         elif self.config['graph_dataset']['random_node_feat']:
    #             print("randommmm featureeeeee!!!!!!!!!!!")
    #             pyg_graph = self.convert_G_random_feature((g, i))
    #         else:
    #             pyg_graph = self.convert_G((g, i))
    #         each_pyg.append(pyg_graph)
    #     return each_pyg

    def nx2pyg(self, graph_list_nx, parallel=False):
        dataset_pyg = []
        total = len(graph_list_nx)
        """
        with Pool(processes=12) as p:
            total = len(graph_list_nx)
            with tqdm(total=total) as pbar:
                for pyg_graph in p.imap_unordered(self.convert_G, zip(graph_list_nx, range(total)) ):
                    pbar.update()
                    dataset_pyg.append(pyg_graph)
        """

        if parallel:
            #process the test list elements in parallel
            with Pool(processes=2) as pool:
                dataset_pyg = pool.map(self.convert_G, zip(graph_list_nx, range(total)))

        else:
            i = 0
            for g in tqdm(graph_list_nx, total=len(graph_list_nx)):
                #if self.config['model']['autoencoder']:
                #    pyg_graph = self.convert_G_autoencoder((g, i))
                if self.conf['graph_dataset']['random_node_feat']:
                    pyg_graph = self.convert_G_random_feature((g, i))
                else:
                    pyg_graph = self.convert_G((g, i))
                if self.config_class.conf['model'].get('my_normalization_adj'):
                    # associo all'elemento Data un edge_weights per la normalizzazione custom
                    self.calc_custom_normalization_weights(pyg_graph)
                else:
                    pyg_graph.edge_weight_normalized = None

                dataset_pyg.append(pyg_graph)
                i += 1

        """
        from joblib import Parallel, delayed
        def process(i):
            return i * i

        results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
        print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        """
        #starttime = time()
        for pyg_graph in dataset_pyg:
            pyg_graph = pyg_graph.to(self.device)
        #durata = time() - starttime
        #print(f"Tempo impiegato per spostare su GPU: {durata}")

        return dataset_pyg

    def get_distance_from_mean(self, node_degrees):
        distance_from_mean = torch.abs(node_degrees - self.average_links_per_graph) / self.tot_links_per_graph
        return distance_from_mean

    def get_norm_probability(self, node_degrees):
        probabilities = torch.tensor([self.degree_prob.get(int(d), 1.0) for d in node_degrees])
        max_prob = max(probabilities)
        max_inv_prob = max(1 / probabilities)
        return max_prob / probabilities / max_inv_prob
    def calc_custom_normalization_weights(self, pyg_graph):
        num_nodes = pyg_graph.num_nodes
        num_edges = pyg_graph.edge_index.size(1)

        # Aggiungi self-loop a 'edge_index'
        self_loop_edge_index = torch.arange(0, num_nodes).unsqueeze(0).repeat(2, 1)
        updated_edge_index = torch.cat([pyg_graph.edge_index, self_loop_edge_index], dim=1)

        existing_edge_weight = torch.ones(num_edges) if pyg_graph.edge_weight is None else pyg_graph.edge_weight
        assert existing_edge_weight.size(0) == num_edges
        updated_edge_weight = torch.cat([existing_edge_weight, torch.ones(num_nodes)])

        row, col = updated_edge_index
        node_degrees = row.bincount(minlength=num_nodes)

        norm_probs = self.get_norm_probability(node_degrees)
        distance_from_mean = self.get_distance_from_mean(node_degrees)
        w = distance_from_mean  #   norm_probs
        weights = torch.sqrt(w/(max(w)))
        assert weights.size(0) == num_nodes
        norm_edge_weight = weights[row] * updated_edge_weight * weights[col]
        #print(np.histogram(norm_edge_weight))
        #print(min(norm_edge_weight))

        pyg_graph.edge_weight_normalized = norm_edge_weight
        pyg_graph.edge_index = updated_edge_index

    def prepare(self, shuffle=True, parallel=False):
        starttime = time()
        self.dataset_pyg = self.nx2pyg(self.dataset_list, parallel)
        durata = time() - starttime
        if self.verbose: print(f"Tempo impiegato: {round(durata, 3)}")

        # shuffle before train test split
        if shuffle:
            x = list(enumerate(self.dataset_pyg))
            random.shuffle(x)
            indices, self.dataset_pyg = zip(*x)
            lista_indici = list(indices)
            self.labels = self.labels[lista_indici]
            # cambio l'ordine anche al dataset di grafi nx (non uso la numpy mask)
            self.dataset_list = [self.dataset_list[i] for i in lista_indici]
            # e cambio l'ordine anche alle orginal class nel caso regression con discrete distrib
            #if self.config['training']['mode'] == 'mode3' and not self.config['graph_dataset']['continuous_p']:
            if self.original_node_class is not None:
                self.original_node_class = [self.original_node_class[i] for i in lista_indici]
            # ho aggiunto le exponent e quindi devo cambiare l'ordine anche a loro...
            if self.exponent is not None:
                self.exponent = [self.exponent[i] for i in lista_indici]
            if self.scalar_label is not None:
                self.scalar_label = [self.scalar_label[i] for i in lista_indici]

            # NON STAVO CAMBIANDO ANCHE LE NODE LABEL.... :'(
            self.actual_node_class = [self.actual_node_class[i] for i in lista_indici]


        self.train_dataset = self.dataset_pyg[:self.tt_split]
        self.test_dataset = self.dataset_pyg[self.tt_split:]
        self.train_len = len(self.train_dataset)
        self.test_len = len(self.test_dataset)

        #aggiungo anche le actual_node_clas ( le degree sequences) divise per traine  test
        self.actual_node_class_train = self.actual_node_class[:self.tt_split]
        self.actual_node_class_test = self.actual_node_class[self.tt_split:]

        #print(self.train_dataset[0].y, self.train_len)
        #print(self.test_dataset[0].y, self.test_len)

        # per impostare il controllo dei seed per ogni wrker del dataloader
        g = torch.Generator()
        g.manual_seed(0)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=shuffle, worker_init_fn=self.seed_worker, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False, worker_init_fn=self.seed_worker, num_workers=0)
        
        self.all_data_loader = self.get_all_data_loader()

        """
        for step, data in enumerate(self.train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()
        """

    def sample_dummy_data(self):
        #whole_data = self.dataset_pyg
        #all_data_loader = DataLoader(whole_data, batch_size=self.bs, shuffle=False)
        batch = next(iter(self.test_loader))
        return batch

    def get_all_data_loader(self):
        whole_data = self.dataset_pyg
        all_data_loader = DataLoader(whole_data, batch_size=self.bs, shuffle=False)
        return all_data_loader

    def seed_worker(self, worker_id):
        #worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(0)  #worker_seed)
        random.seed(0)  #worker_seed)

    def get_coppie_from_dataset(self, loader):
        NN = len(loader.dataset)
        coppie_numeric = list(itertools.combinations(range(NN), 2))

        Adjs = self.get_concatenated_input_adjs(loader)
        coppie = Adjs[coppie_numeric]

        # print(f"Ci sono {len(coppie)} coppie")
        return coppie

    def get_concatenated_input_adjs(self, loader):
        Adjs = []
        for data in loader:
            input_adj = to_dense_adj(data.edge_index, data.batch)
            input_adj = input_adj.detach().cpu().numpy()
            Adjs.append(input_adj)
        Adjs = np.array(Adjs)
        # alla fine devo appiattire lungo la dimensione dei batch, che no nserve
        Adjs = Adjs.reshape(-1, Adjs.shape[-1], Adjs.shape[-1])
        return torch.Tensor(Adjs)

    def get_concatenated_constant_matrix(self, loader):
        for data in loader:
            input_adj = to_dense_adj(data.edge_index, data.batch)[0]  # loader mi ritorna un batch intero
            break# TODO: cambiare nel aso in cui la batch size sia più piccola del dataset!
        out = []
        for p in self.scalar_label:  # nel caso di ER questa l è la p_ER
            m = self.matrix_constant_ER(input_adj, p)
            out.append(m)
        out = torch.Tensor(np.array(out))
        return out
    def matrix_constant_ER(self, input_matrix, p_ER):
        # creo una matrice costante = p_ER
        shape = input_matrix.shape
        cost_adj = np.full(shape, p_ER)
        # cost_adj e' la predicted (x) da confrontare con input (y)
        return cost_adj

    def get_concatenated_starting_matrix(self, loader):
        for data in loader:
            input_adj = to_dense_adj(data.edge_index, data.batch)[0]  # loader mi ritorna un batch intero
            break   # TODO: cambiare nel aso in cui la batch size sia più piccola del dataset!
        out = []
        for p in self.scalar_label:  # nel caso di powerlaw questa è l'esponente
            m = np.tile(self.p_ij, (input_adj.shape[0],1,1))
            out.append(m)
        out = torch.Tensor(np.array(out))
        return out

