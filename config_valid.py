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
        self.unique_train_name = "Noname"
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

        neurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        if self.conf['model']['last_layer_dense']:
            self.lastneuron = self.conf['model']['neurons_last_linear'][-1]
        else:
            self.lastneuron = neurons_per_layer[-1]

        self.init_weights_mode = self.get_init_weights_mode()

        # verifico che l'ultimo neurone sia consistente col training mode
        if self.lastneuron == 1:
            assert self.lastneuron == self.modo['last_neuron'], 'Ultimo neurone = 1 ma training mode diverso'
        else:
            assert self.modo['last_neuron'] == 'n_class', 'Ultimi neuroni > 1 ma training mode diverso'
            assert self.lastneuron == self.num_classes(), f'Ultimi neuroni ({self.lastneuron}) diversi dal numero di classi ({self.num_classes()}) '

        #last_layer_dense = self.config['model']['last_layer_dense']

        # verifico il caso in cui ho solo due classi
        n_class = len(self.conf['graph_dataset']['list_p'])
        if self.modo['labels'] == Labels.zero_one:
            assert n_class == 2, 'Num classi in list_p non consistente con training mode'

        # verifico che Num nodes sia consistente col training mode:
        # nel caso di power law CM voglio poter settare num nodes diverso per ogni classe
        if isinstance(self.conf['graph_dataset']['Num_nodes'], list):
            assert self.graphtype == GraphType.CM
        if isinstance(self.conf['graph_dataset']['Num_nodes'], int):
            assert self.graphtype != GraphType.CM

        # verifico che nel caso dello SBM la list_p sia una matrice
        if self.graphtype == GraphType.SBM:
            assert np.array(self.conf['graph_dataset']['list_p']).ndim == 2, "Lo Stochastic Block Model richiede una matrice di probabilità"
        else:
            assert np.array(self.conf['graph_dataset']['list_p']).ndim == 1, "probabilità inserite come matrice ma non stiamo nel SBM"

        # verifico che in graph_dataset ci sia un solo True
        assert self.only1_graphtype(), "Errore nel config file: scegliere un solo tipo di grafi"

        # assegno il tipo di grafo
        self.set_graphtype()

        # verifico nel cm multiclass che il numero di numnodes sia uguale al numero di esponenti
        if self.conf['graph_dataset']['confmodel']:
            assert len(self.conf['graph_dataset']['Num_nodes']) == len(self.conf['graph_dataset']['list_exponents'])

        self.unique_train_name = self.create_unique_train_name()
    def set_graphtype(self):
        if self.conf['graph_dataset']['ERmodel']:
            self.graphtype = GraphType.ER
        elif self.conf['graph_dataset']['regular']:
            self.graphtype = GraphType.Regular
        elif self.conf['graph_dataset']['confmodel']:
            self.graphtype = GraphType.CM
        elif self.conf['graph_dataset']['sbm']:
            self.graphtype = GraphType.SBM

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
        return '§' + s1 + '+' + s2 + '§'

    def create_unique_train_name(self):
        numnodi = self.conf['graph_dataset']['Num_nodes']
        if isinstance(numnodi, list):
            numnodi = numnodi[0]
        numgrafi = self.conf['graph_dataset']['Num_grafi_per_tipo'] * 2
        percentuale_train = self.conf['training']['percentage_train']
        modo = self.conf['training']['mode']
        freezed = self.conf['model']['freezeGCNlayers']
        if self.graphtype == GraphType.CM:
            data_label = self.conf['graph_dataset']['list_exponents']
            if isinstance(data_label, list):
                data_label = f"{len(data_label)}exps"
        else:
            data_label = self.conf['graph_dataset']['list_p']

        layer_neuron_string = self.create_layer_neuron_string()
        lr = self.conf['training']['learning_rate']
        init_weights = self.conf['model']['init_weights']
        nome = f"{self.graphtype}_{data_label}_nodi{numnodi}_grafi{numgrafi}_{modo}_layers{layer_neuron_string}_initw{init_weights}_lr{lr}_freezed{freezed}"
        nome = nome.replace(', ', '_')
        return nome

    def num_classes(self):
        if self.conf['graph_dataset']['ERmodel'] \
                or self.conf['graph_dataset']['sbm'] \
                or self.conf['graph_dataset']['regular']:
            return len(self.conf['graph_dataset']['list_p'])

        elif self.conf['graph_dataset']['confmodel']:
            return len(self.conf['graph_dataset']['list_exponents'])


    def get_config_dataframe(self):
        if self.conf:
            conf_df = json_normalize(self.conf)
            return conf_df
        else:
            raise Exception("Configuration dictionary not initialized")
        