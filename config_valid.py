import yaml
from enum import Enum
import torch
from pandas import json_normalize
import numpy as np

basic_config_file_path = "configs.yml"

class Labels(Enum):
    onehot = '1-hot'
    zero_one = '{0,1}'
    prob = 'p'

class TrainingMode():
    mode1 = {'type': 'classification', 'criterion': torch.nn.CrossEntropyLoss(), 'labels': Labels.onehot, 'last_neuron': 'n_class'}
    mode2 = {'type': 'classification', 'criterion': torch.nn.BCEWithLogitsLoss(), 'labels': Labels.zero_one, 'last_neuron': 1}
    mode3 = {'type': 'regression', 'criterion': torch.nn.MSELoss(), 'labels': Labels.prob, 'last_neuron': 1}

class GraphType(Enum):
    ER = 1
    Regular = 2
    CM = 3
    SBM = 4

class Inits():  # TODO: capire perché se estendo da Enum succede un CASINO! non vanno più bene le uguaglianze
    normal = 'normal'
    constant = 'constant'
    uniform = 'uniform'
    eye = 'eye'
    dirac = 'dirac'
    xavier_uniform = 'xavier_uniform'
    xavier_normal = 'xavier_normal'
    kaiming_uniform = 'kaiming_uniform'
    kaiming_normal = 'kaiming_normal'
    trunc_normal = 'trunc_normal'
    orthogonal = 'orthogonal'
    sparse = 'sparse'

class Config():
    def __init__(self, config_file=None, data_dict=None):
        if data_dict:
            self.conf = data_dict
        else:
            self.conf = None
        if config_file:
            self.config_file = config_file
            self.load_conf()
        elif not config_file and not data_dict:
            self.config_file = basic_config_file_path
            self.load_conf()

        self.modo = None
        self.graphtype = None
        self.num_classes = 0
        self.unique_train_name = "Noname"
        self.longstring_graphtype = "Nographtype"
        self.long_string_experiment = "Nostring"
        self.valid_conf()
        #self.reload_conf()

    @classmethod
    def fromdict(cls, datadict):
        return cls(None,datadict)

    def reload_conf(self):
        self.conf = yaml.safe_load(open(self.config_file))
        
    def load_conf(self, config_file_path=None):
        if config_file_path:
            self.conf = yaml.safe_load(open(config_file_path))
        else:
            self.conf = yaml.safe_load(open(self.config_file))

    def get_mode(self):
        return eval(f"TrainingMode.{self.conf['training']['mode']}")

    def get_init_weights_mode(self):
        init_weights_mode = self.conf['model']['init_weights']
        if init_weights_mode == 'None':
            return None
        return eval(f"Inits.{init_weights_mode}")

    def valid_conf(self):
        self.modo = self.get_mode()
        # assegno il tipo di grafo
        self.set_graphtype()

        # verifico che in graph_dataset ci sia un solo True
        assert self.only1_graphtype(), "Errore nel config file: scegliere un solo tipo di grafi"

        neurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        if self.conf['model']['last_layer_dense']:
            self.lastneuron = self.conf['model']['neurons_last_linear'][-1]
        else:
            self.lastneuron = neurons_per_layer[-1]

        self.init_weights_mode = self.get_init_weights_mode()

        # verifico che l'ultimo neurone sia consistente col training mode
        self.get_num_classes()
        if self.lastneuron == 1:
            assert self.lastneuron == self.modo['last_neuron'], 'Ultimo neurone = 1 ma training mode diverso'
        else:
            assert self.modo['last_neuron'] == 'n_class', 'Ultimi neuroni > 1 ma training mode diverso'
            assert self.lastneuron == self.num_classes, f'Ultimi neuroni ({self.lastneuron}) diversi dal numero di classi ({self.num_classes}) '

        #last_layer_dense = self.config['model']['last_layer_dense']

        # verifico il caso in cui ho solo due classi
        n_class = len(self.conf['graph_dataset']['list_p'])
        if self.modo['labels'] == Labels.zero_one:
            assert n_class == 2, 'Num classi in list_p non consistente con training mode'

        # verifico che Num nodes sia consistente col training mode:
        # nel caso di power law CM E SBM(!) -> voglio poter settare num nodes diverso per ogni classe  (per ogni comunità)
        if isinstance(self.conf['graph_dataset']['Num_nodes'], list):
            assert self.graphtype == GraphType.CM or self.graphtype == GraphType.SBM
        if isinstance(self.conf['graph_dataset']['Num_nodes'], int):
            assert self.graphtype != GraphType.CM

        # verifico che nel caso dello SBM la Num_Nodes sia una matrice
        if self.graphtype == GraphType.SBM:
            assert isinstance(self.conf['graph_dataset']['Num_nodes'], list), "Lo Stochastic Block Model richiede una lista per Num nodes"
        #else:
        #    assert np.array(self.conf['graph_dataset']['list_p']).ndim == 1, "probabilità inserite come matrice ma non stiamo nel SBM"




        # verifico nel cm multiclass che il numero di numnodes sia uguale al numero di esponenti
        if self.graphtype == GraphType.CM:
            assert len(self.conf['graph_dataset']['Num_nodes']) == len(self.conf['graph_dataset']['list_exponents'])
        # verifico che le BMS il numero di num nodes sia uguale al numero di comunità
        if self.graphtype == GraphType.SBM:
            assert len(self.conf['graph_dataset']['Num_nodes']) == np.array(self.conf['graph_dataset']['community_probs']).shape[1]

        self.unique_train_name, self.long_string_experiment = self.create_unique_train_name()
        #print(f"Training with {self.num_classes} classes")
    def set_graphtype(self):
        if self.conf['graph_dataset']['ERmodel']:
            self.graphtype = GraphType.ER
            self.longstring_graphtype = "Erdos-Renyi"
        elif self.conf['graph_dataset']['regular']:
            self.graphtype = GraphType.Regular
            self.longstring_graphtype = "Grafo Regolare"
        elif self.conf['graph_dataset']['confmodel']:
            self.graphtype = GraphType.CM
            self.longstring_graphtype = "Configuration Model"
        elif self.conf['graph_dataset']['sbm']:
            self.graphtype = GraphType.SBM
            self.longstring_graphtype = "Stochastic Block Model"

    def only1_graphtype(self):
        bool_array = [self.conf['graph_dataset']['ERmodel'],
                    self.conf['graph_dataset']['regular'],
                    self.conf['graph_dataset']['confmodel'],
                    self.conf['graph_dataset'].get('sbm', False)]
        # check if bool_array contains one and only one True
        true_found = False
        for v in bool_array:
            if v and not true_found:
                true_found = True
            elif v and true_found:
                return False  # "Too Many Trues"
        return true_found

    def create_layer_neuron_string(self):
        gcnneurons = self.conf['model']['GCNneurons_per_layer']
        linears = self.conf['model']['neurons_last_linear']
        s1 = str(gcnneurons).replace(', ', '-').strip('[').strip(']')
        s2 = str(linears).replace(', ', '-').strip('[').strip(']')
        if self.conf['model']['last_layer_dense']:
            res = '§' + s1 + '+' + s2 + '§'
        else:
            res = '§' + s1 + '§'
        return res

    def create_unique_train_name(self):
        numnodi = self.conf['graph_dataset']['Num_nodes']
        if isinstance(numnodi, list):
            numnodi = numnodi[0]
        numgrafi = self.conf['graph_dataset']['Num_grafi_per_tipo']
        percentuale_train = self.conf['training']['percentage_train']
        modo = self.conf['training']['mode']
        freezed = self.conf['model']['freezeGCNlayers']
        if self.graphtype == GraphType.CM:
            data_label = self.conf['graph_dataset']['list_exponents']
            if isinstance(data_label, list):
                data_label = f"{len(data_label)}exps"
        elif self.graphtype == GraphType.ER:
            data_label = self.conf['graph_dataset']['list_p']
        elif self.graphtype == GraphType.SBM:
            data_label = self.conf['graph_dataset']['community_probs']

        layer_neuron_string = self.create_layer_neuron_string()
        lr = self.conf['training']['learning_rate']
        init_weights = self.conf['model']['init_weights']
        #nome = f"{self.graphtype}_{data_label}_nodi{numnodi}_grafiXtipo{numgrafi}_{modo}_layers{layer_neuron_string}_initw-{init_weights}_lr{lr}_freezed{freezed}"
        nome = f"{self.graphtype}_Classi{self.num_classes}_nodi{numnodi}_grafiXtipo{numgrafi}_{modo}_layers{layer_neuron_string}_initw-{init_weights}_lr{lr}_GCNfreezed{freezed}"
        nome = nome.replace(', ', '_')

        # creo stringa lunga
        long_string = f"{self.longstring_graphtype} - {numnodi} nodi - {numgrafi} grafi per classe \n {layer_neuron_string} - {init_weights} - lr:{lr} - GCNfreezed:{freezed}"
        return nome, long_string


    def get_num_classes(self):
        if self.conf['graph_dataset']['ERmodel'] \
                or self.conf['graph_dataset']['regular']:
            self.num_classes = len(self.conf['graph_dataset']['list_p'])
        elif self.conf['graph_dataset']['confmodel']:
            self.num_classes = len(self.conf['graph_dataset']['list_exponents'])
        elif self.conf['graph_dataset']['sbm']:
            self.num_classes = len(self.conf['graph_dataset']['community_probs'])
        #print(f"Abbiamo {self.num_classes} classi.")
        #print(self.conf['graph_dataset'])
        return self.num_classes


    def get_config_dataframe(self):
        if self.conf:
            conf_df = json_normalize(self.conf)
            return conf_df
        else:
            raise Exception("Configuration dictionary not initialized")
        