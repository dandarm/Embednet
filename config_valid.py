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
        else:
            self.config_file = basic_config_file_path

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
        neurons_per_layer = self.conf['model']['neurons_per_layer']
        lastneuron = neurons_per_layer[-1]
        if lastneuron == 1:
            assert lastneuron == modo['last_neuron'], 'Ultimo neurone = 1 ma training mode diverso'
        else:
            assert modo['last_neuron'] == 'n_class', 'Ultimi neuroni > 1 ma training mode diverso'
            assert lastneuron == self.num_classes_ER(), 'Ultimi neuroni diversi dal numero di classi '

        #last_layer_dense = self.config['model']['last_layer_dense']

        # verifico il caso in cui ho solo due classi
        n_class = len(self.conf['graph_dataset']['list_p'])
        if modo['labels'] == Labels.zero_one:
            assert n_class == 2, 'Num classi in list_p non consistente con training mode'


    def layer_neuron_string(self):
        neurons = self.conf['model']['neurons_per_layer']
        return str(neurons).replace(', ', '-').strip('[').strip(']')

    def num_classes_ER(self):
        return len(self.conf['graph_dataset']['list_p'])

        