import yaml
from enum import Enum
import torch

basic_config_file_path = "configs.yml"

class Labels(Enum):
    onehot = '1-hot'
    zero_one = '{0,1}'
    prob = 'p'

class TrainingMode():
    mode1 = {'type': 'classification', 'criterion': torch.nn.CrossEntropyLoss(), 'labels': Labels.onehot, 'last_neuron': 'n_class'}
    mode2 = {'type': 'classification', 'criterion': torch.nn.BCEWithLogitsLoss(), 'labels': Labels.zero_one, 'last_neuron': 1}
    mode3 = {'type': 'regression', 'criterion': torch.nn.MSELoss(), 'labels': Labels.prob, 'last_neuron': 1}

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

        neurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        if self.conf['model']['last_layer_dense']:
            self.lastneuron = self.conf['model']['neurons_last_linear'][-1]
        else:
            self.lastneuron = neurons_per_layer[-1]
        self.valid_conf()
        self.modo = self.get_mode()


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

    def valid_conf(self):
        modo = self.get_mode()
        modo_str = self.conf['training']['mode']

        # verifico che l'ultimo neurone sia consistente col training mode

        if self.lastneuron == 1:
            assert self.lastneuron == modo['last_neuron'], 'Ultimo neurone = 1 ma training mode diverso'
        else:
            assert modo['last_neuron'] == 'n_class', 'Ultimi neuroni > 1 ma training mode diverso'
            assert self.lastneuron == self.num_classes_ER(), 'Ultimi neuroni diversi dal numero di classi '

        #last_layer_dense = self.config['model']['last_layer_dense']

        # verifico il caso in cui ho solo due classi
        n_class = len(self.conf['graph_dataset']['list_p'])
        if modo['labels'] == Labels.zero_one:
            assert n_class == 2, 'Num classi in list_p non consistente con training mode'

        # verifico che in graph_dataset ci sia un solo True
        bool_arr = [self.conf['graph_dataset']['ERmodel'],
                    self.conf['graph_dataset']['regular'],
                    self.conf['graph_dataset']['confmodel']]
        assert self.only1(bool_arr), "Errore nel config file: scegliere un solo tipo di grafi"

    def only1(self, bool_array):
        # check if bool_array contains one and only one True
        true_found = False
        for v in bool_array:
            if v and not true_found:
                true_found = True
            elif v and true_found:
                return False  # "Too Many Trues"
        return true_found

    def layer_neuron_string(self):
        neurons = self.conf['model']['GCNneurons_per_layer']
        return str(neurons).replace(', ', '-').strip('[').strip(']')

    def num_classes_ER(self):
        return len(self.conf['graph_dataset']['list_p'])

        