import sys
sys.path.append("../")
from abc import abstractmethod
from Graph_AE.utils.CustomDataSet import SelectGraph 


class TakeRealDataset():
    def __init__(self, config_class, verbose):
        self.config_class = config_class
        self.dataset_name = config_class.conf['graph_dataset']['real_data_name']
        self.dataset = None
        self.verbose = verbose
        
    @abstractmethod
    def get_dataset(self):
        if self.dataset_name == 'REDDIT-BINARY':
            self.dataset = TakeTUDataset(self.config_class).get_dataset()
    
    
    
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
            dataset_mod.append(d)
        return dataset_mod
        
        