import random
from time import time
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from multiprocessing import Pool

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

        #calcola il max degree
        #[dataset_list]

class Dataset(GeneralDataset):

    def __init__(self, percentage_train, batch_size, device, config_class, dataset_list, labels, original_node_class, exponent, actual_node_class, scalar_label):
        super().__init__(dataset_list, labels, original_node_class=original_node_class, exponent=exponent, actual_node_class=actual_node_class, scalar_label=scalar_label)
        #self.dataset_list = dataset_list  # rename in dataset_list
        self.dataset_pyg = None
        #self.labels = labels
        self.len_data = len(self.dataset_list)
        self.tt_split = int(self.len_data * percentage_train)
        self.train_dataset = None
        self.train_len = None
        self.test_dataset = None
        self.test_len = None
        self.bs = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.config_class = config_class
        self.config = config_class.conf
        self.last_neurons = self.config_class.lastneuron
        self.all_data_loader = None


    @classmethod
    def from_super_instance(cls, percentage_train, batch_size, device, config_class, super_instance):
        return cls(percentage_train, batch_size, device, config_class, **super_instance.__dict__)

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
        # else:
        #     tipo = torch.float
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
                if self.config['graph_dataset']['random_node_feat']:
                    pyg_graph = self.convert_G_random_feature((g, i))
                else:
                    pyg_graph = self.convert_G((g, i))
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

    def prepare(self, shuffle=True, parallel=False):
        starttime = time()
        self.dataset_pyg = self.nx2pyg(self.dataset_list, parallel)
        durata = time() - starttime
        print(f"Tempo impiegato: {durata}")

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

