import random
from time import time
from tqdm import tqdm
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from Dataset import Dataset
import torch_geometric.transforms as T
from multiprocessing import Pool


class DatasetAutoencoder(Dataset):
    """
    Questo Dataset splitta i link in positive e negative test per calcolare una AUC
    """
    def __init__(self, percentage_train, batch_size, device, config_class, dataset_list, labels, original_node_class, exponent, actual_node_class, scalar_label):
        super().__init__(percentage_train, batch_size, device, config_class, dataset_list, labels, original_node_class, exponent, actual_node_class, scalar_label)
        train_percent = config_class.conf['training']['percentage_train']
        test_percent = 1 - train_percent
        self.transform4ae = T.RandomLinkSplit(num_val=0.0, num_test=test_percent, is_undirected=True,
                                          split_labels=True, add_negative_train_samples=False)
        self.all_data_loader = None

    def convert_G(self, g_i):
        g, i = g_i
        # aggiungo i metadati x e y per l'oggetto Data di PYG
        nodi = list(g.nodes)
        for n in nodi:
            g.nodes[n]["x"] = [1.0]
            g.nodes[n]["id"] = [n]

        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        tipo = torch.float
        pyg_graph.y = torch.tensor(np.array([type_graph]), dtype=tipo)

        # applico lo split sul singolo grafo, ottengo le edges per il training e per il test
        # imposto 0 per la validation
        pyg_graph, val_data, test_data = self.transform4ae(pyg_graph)

        return pyg_graph, test_data

    def nx2pyg(self, graph_list_nx, parallel=False):
        train_pyg = []
        test_pyg = []
        train_pos_edge_index = []
        total = len(graph_list_nx)
        i = 0
        for g in tqdm(graph_list_nx, total=total):
            train_edges, test_edges = self.convert_G((g, i))
            train_edges = train_edges.to(self.device)
            test_edges = test_edges.to(self.device)
            train_pyg.append(train_edges)
            test_pyg.append(test_edges)
            #train_pos_edge_index.append(train_graph.train_pos_edge_index)
            i += 1

        return train_pyg, test_pyg


    def prepare(self, shuffle=True, parallel=False):
        starttime = time()
        # abbiamo una lista di grafi che contengono solo un subset di edges per training
        # e una lista di grafi che contengono solo un subset di edges per il test
        self.dataset_pyg, test_pyg = self.nx2pyg(self.dataset_list, parallel)
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

            # cambio anche il test
            self.test_dataset = [test_pyg[i] for i in lista_indici]

        self.train_dataset = self.dataset_pyg

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

    def get_all_data_loader(self):
        # non ha senso applicare,l'autoencoder a tutti i link...
        # ma TODO: potrei comunque ottnere un auc su tutto il dataset
        return self.test_loader

class DatasetReady(Dataset):
    def __init__(self, percentage_train, batch_size, device, config_class, dataset_list):
        super().__init__(percentage_train, batch_size, device, config_class, dataset_list, None, None, None, None, None) #, labels, original_node_class, exponent, actual_node_class, scalar_label)
        train_percent = config_class.conf['training']['percentage_train']
        test_percent = 1 - train_percent
        #self.transform4ae = T.RandomLinkSplit(num_val=0.0, num_test=test_percent, is_undirected=True, split_labels=True, add_negative_train_samples=False)
        self.all_data_loader = None
        self.dataset_pyg = dataset_list
        
        for pyg_graph in self.dataset_pyg:
            pyg_graph = pyg_graph.to(self.device)
        
        self.train_dataset = self.dataset_pyg[:self.tt_split]
        self.test_dataset = self.dataset_pyg[self.tt_split:]
        self.train_len = len(self.train_dataset)
        self.test_len = len(self.test_dataset)
        
        shuffle = self.config['training']['shuffle_dataset']
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=shuffle, worker_init_fn=self.seed_worker, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False, worker_init_fn=self.seed_worker, num_workers=0)

        self.all_data_loader = self.get_all_data_loader()

    def get_all_data_loader(self):
        # non ha senso applicare,l'autoencoder a tutti i link...
        # ma TODO: potrei comunque ottnere un auc su tutto il dataset
        return self.test_loader
        