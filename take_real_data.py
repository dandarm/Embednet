import sys
sys.path.append("../")
from abc import abstractmethod
import pickle
import torch
from graph_generation import GenerateGraph, GenerateGraph_from_numpyarray
from Graph_AE.utils.CustomDataSet import SelectGraph



class TakeRealDataset():
    def __init__(self, config_class, verbose):
        self.config_class = config_class
        self.dataset_name = config_class.conf['graph_dataset']['real_data_name']
        #self.dataset = None
        self.verbose = verbose
        self.gg = None  # il riferimento al graph generation originale
        
    @abstractmethod
    def get_dataset(self):
        if self.dataset_name == 'REDDIT-BINARY':
            dataset = TakeTUDataset(self.config_class).get_dataset()
            self.gg = GenerateGraph(self.config_class)
            self.gg.dataset = dataset  # il dataset di gg in genere è una lista, in questo caso di oggetti Data
            self.gg.num_nodes_per_graph = self.get_num_nodes_per_graph(dataset)
        if self.dataset_name == 'BACI':
            gg = TakeWTWdataset(self.config_class).get_dataset()
            self.gg = gg
            # TakeWTWdataset già istanzia un gg e il gg.dataset

    def get_num_nodes_per_graph(self, dataset):
        num = []
        for d in dataset:
            num.append(d.num_nodes)
        return num

    
    
    
class TakeTUDataset(TakeRealDataset):
    def __init__(self, config_class, verbose=False):
        super().__init__(config_class, verbose)
        print(f"Verrà usato il dataset {self.dataset_name}")
        
    def get_dataset(self):
        SelectGraph.data_name = self.dataset_name
        data_set = SelectGraph('../graphdata/' + SelectGraph.data_name)
        
        #check sulle feature
        dataset_mod = []
        for d in data_set:
            if d.x is None:
                d.x = torch.ones([d.num_nodes], dtype=torch.float).unsqueeze(1)
            if d.y is None:  # aggiungo la graph label y
                d.y = torch.ones(1, dtype=torch.int8)
            dataset_mod.append(d)
        return dataset_mod
        
class TakeWTWdataset(TakeRealDataset):
    def __init__(self, config_class, verbose=False):
        super().__init__(config_class, verbose)
        print(f"Verrà usato il dataset {self.dataset_name}")

    def get_dataset(self):
        with open('BACI_binary.pickle', 'rb') as handle:
            diz_adj = pickle.load(handle)
        list_adj = [v.to_numpy() for k,v in diz_adj.items()]

        self.gg = GenerateGraph_from_numpyarray(self.config_class, list_adj)
        self.gg.create_nx_graphs()
        return self.gg
