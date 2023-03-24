import torch
from torch_geometric.data import DataLoader
from classification.Graph_AE import Net as MIAGAE
from utils.train_utils import train_cp
from utils.CustomDataSet import SelectGraph


class Trainer_AutoencoderMIAGAE_DEBUG():
    def __init__(self, config_class, verbose=False):
        self.config_class = config_class
        
        self.test_loss_list = []
        self.metric_list = []
        
    def reinit_conf(self, config_class):
        self.config_class = config_class
        self.config_class.valid_conf()
        self.conf = self.config_class.conf

        self.percentage_train = self.conf['training']['percentage_train']
        self.lr = self.conf['training']['learning_rate']
        self.epochs = self.conf['training']['epochs']
        self.batch_size = self.conf['training']['batch_size']
        self.last_layer_neurons = self.config_class.get_mode()['last_neuron']
        #self.mode = self.conf['training']['mode']  # 'classification'  or 'regression'  or 'unsupervised'
        self.unique_train_name = self.config_class.unique_train_name

        self.epochs_checkpoint = self.conf['training'].get('epochs_model_checkpoint')
        self.shuffle_dataset = self.conf['training']['shuffle_dataset']
        self.save_best_model = self.conf['training']['save_best_model']

        self.criterion = self.config_class.get_mode()['criterion']

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"
            
    def init_all(self, parallel=True, verbose=False): 
        graph_ae_configs = self.conf.get('graph_ae_model')
        num_kernels = graph_ae_configs['num_kernels']
        depth = graph_ae_configs['depth']
        comp_rate = graph_ae_configs['comp_rate']
        GCNneurons_per_layer = graph_ae_configs['GCNneurons_per_layer']
        self.model = MIAGAE(GCNneurons_per_layer[0], num_kernels, depth, [comp_rate] * depth, GCNneurons_per_layer[1:], self.device).to(self.device)
        self.embedding_dimension =  GCNneurons_per_layer[-1]
        
        SelectGraph.data_name = self.config_class.conf['graph_dataset']['real_data_name']
        data_set = SelectGraph('../graphdata/' + SelectGraph.data_name)
        
        #check sulle feature
        dataset_mod = []
        for d in data_set:
            if d.x is None:
                d.x = torch.ones([d.num_nodes], dtype=torch.float).unsqueeze(1) 
            dataset_mod.append(d)

        len_data = len(dataset_mod)
        tt_split = int(len_data * self.percentage_train)
        self.train_set = DataLoader(dataset_mod[:tt_split], batch_size=self.batch_size, shuffle=True)
        self.test_set = DataLoader(dataset_mod[tt_split:], batch_size=self.batch_size, shuffle=False)
        

    
    def launch_training(self, verbose=0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_cp(self.model, optimizer, self.device, self.train_set, self.test_set, self.epochs, None, None)
    
    def take_embedding_all_data(self):
        return None, None, None, None