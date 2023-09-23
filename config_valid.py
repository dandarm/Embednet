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

class CriterionType():
    criterions = {
        "MSELoss": torch.nn.MSELoss(),
        "BCELoss": torch.nn.BCELoss(),
        "MSELoss_ww": torch.nn.MSELoss(reduction='none'),
        "BCELoss_ww": torch.nn.BCELoss(reduction='none'),
    }


class GraphType(Enum):
    ER = 1
    Regular = 2
    CM = 3
    SBM = 4
    REAL = 5
    CONST_DEG_DIST = 6

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
    def __init__(self, config_file=None, data_dict=None, verbose=False):
        self.graphtype = None
        self.verbose = verbose
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
            
        self.NumNodes = self.conf['graph_dataset']['Num_nodes']
        self.length_list_NumNodes = len(self.NumNodes)

        self.num_classes = -1
        self.unique_train_name = "Noname"
        self.longstring_graphtype = "Nographtype"
        self.long_string_experiment = "Nostring"

        self.autoencoding = False
        if (self.conf['model'].get('autoencoder') or
                self.conf['model'].get('autoencoder_confmodel') or
                self.conf['model'].get('autoencoder_mlpdecoder') or
                self.conf['model'].get('autoencoder_fullMLP') or
                self.conf['model'].get('autoencoder_graph_ae')):
            self.autoencoding = True

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"


        self.modo = self.get_mode()
        self.set_graphtype()
        self.unique_train_name, self.long_string_experiment = self.create_unique_train_name()

        self.valid_conf()



        #self.reload_conf()

    @classmethod
    def fromdict(cls, datadict, verbose=False):
        return cls(None,datadict, verbose=verbose)

    def reload_conf(self):
        self.conf = yaml.safe_load(open(self.config_file))
        
    def load_conf(self, config_file_path=None):
        if config_file_path:
            self.conf = yaml.safe_load(open(config_file_path))
        else:
            self.conf = yaml.safe_load(open(self.config_file))

    def get_mode(self):
        return eval(f"TrainingMode.{self.conf['training']['mode']}")

    def get_loss(self):
        loss_string = self.conf['training'].get('loss')
        if loss_string is not None:
            #return eval(f"CriterionType.criterions[{loss_string}]")
            return CriterionType.criterions[f"{loss_string}"]
        else:
            return None

    def get_init_weights_mode(self):
        init_weights_mode = self.conf['model']['init_weights']
        if init_weights_mode == 'None':
            return None
        return eval(f"Inits.{init_weights_mode}")

    def valid_conf(self):
        if self.verbose:
            print(f"Validando la config: {self.unique_train_name}")


        # verifico che in graph_dataset ci sia un solo True
        self.only1_graphtype()
        # lo stesso per i tipi di modello
        self.only1_model()

        neurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        if self.conf['model']['last_layer_dense']:
            self.lastneuron = self.conf['model']['neurons_last_linear'][-1]
        else:
            self.lastneuron = neurons_per_layer[-1]

        self.init_weights_mode = self.get_init_weights_mode()

        # verifico che l'ultimo neurone sia consistente col training mode
        self.get_num_classes()
        if not self.autoencoding:
            if self.lastneuron == 1:
                assert self.lastneuron == self.modo['last_neuron'], 'Ultimo neurone = 1 ma training mode diverso'
            else:
                assert self.modo['last_neuron'] == 'n_class', 'Ultimi neuroni > 1 ma training mode diverso'
                assert self.lastneuron == self.num_classes, f'Ultimi neuroni ({self.lastneuron}) diversi dal numero di classi ({self.num_classes}) '


        if self.modo['labels'] == Labels.zero_one:
            assert self.num_classes == 2, 'Num classi non consistente con training mode'

        # verifico che Num nodes sia sempre una lista. nel caso di una classe o tutte uguali
        # voglio dover speciricare lo stesso num nodes per ogni classe
        if not isinstance(self.NumNodes, list):
            raise ValueError('Num_nodes deve essere una lista')

        # verifico che il numero di numnodes sia uguale al numero di classi
        self.check_lungh_numnodes_with_classes()


        # verifico che nel caso dello SBM la Num_Nodes sia una matrice
        #if self.graphtype == GraphType.SBM:
        #    assert isinstance( self.NumNodes, list), "Lo Stochastic Block Model richiede una lista per Num nodes"
        #else:
        #    assert np.array(self.conf['graph_dataset']['list_p']).ndim == 1, "probabilità inserite come matrice ma non stiamo nel SBM"

        # verifico che nel SBM il numero di num nodes sia uguale al numero di comunità
        if self.graphtype == GraphType.SBM:
            shape_nodi = np.array(self.NumNodes).shape
            shape_comunity = np.array(self.conf['graph_dataset']['community_probs']).shape[:-1]
            assert shape_nodi == shape_comunity, \
                f"Num_nodes non ha la stessa shape delle comunità nello SBM: {shape_nodi} != {shape_comunity}"

        self.check_last_layer_activation_autoencoder()

        
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
        elif self.conf['graph_dataset'].get('sbm'):
            self.graphtype = GraphType.SBM
            self.longstring_graphtype = "Stochastic Block Model"
        elif self.conf['graph_dataset'].get('const_degree_dist'):
            self.graphtype = GraphType.CONST_DEG_DIST
            self.longstring_graphtype = "Sequenza di grado costante"
        elif self.conf['graph_dataset'].get('real_dataset'):
            self.graphtype = GraphType.REAL
            self.longstring_graphtype = self.conf['graph_dataset']['real_data_name']

    def only1_graphtype(self):
        #if self.conf['graph_dataset'].get('real_dataset', False):
        #    print("Dataset reale: ignoro tutti gli altri graphtype sintetici")
        #    return 1
        bool_array = [self.conf['graph_dataset']['ERmodel'],
                    self.conf['graph_dataset']['regular'],
                    self.conf['graph_dataset']['confmodel'],
                    self.conf['graph_dataset'].get('sbm', False),
                    self.conf['graph_dataset'].get('const_degree_dist', False),
                      self.conf['graph_dataset'].get('real_dataset', False)]

        true_found = self.only1Bool(bool_array, tipi='graph_dataset')
        return true_found

    def only1_model(self):
        bool_array = [self.conf['model']['only_encoder'],
                      self.conf['model']['autoencoder'],
                      self.conf['model']['autoencoder_confmodel'],
                      self.conf['model']['autoencoder_graph_ae'],
                      self.conf['model']['autoencoder_mlpdecoder'],
                      self.conf['model']['autoencoder_fullMLP']]
        true_found = self.only1Bool(bool_array, tipi='modello')
        return true_found

    def only1Bool(self, bool_array, tipi):
        # check if bool_array contains one and only one True
        true_found = sum(bool_array)
        # print(true_found)
        # for v in bool_array:
        #     if v and not true_found:
        #         true_found = True
        #     elif v and true_found:
        #         return False  # "Too Many Trues"
        assert true_found == 1, f"Errore nel config: scegliere un solo tipo di {tipi}. Trovati {true_found}"
        return true_found

    def create_layer_neuron_string(self):
        neurons = self.conf['model']['GCNneurons_per_layer']
        if self.conf['model']['autoencoder_graph_ae']:
            neurons = self.conf['graph_ae_model']['neurons_per_layer']
        elif self.conf['model']['autoencoder_fullMLP']:
            neurons = self.conf['mlp_ae_model']['neurons_per_layer']

        s1 = str(neurons).replace(', ', '-').strip('[').strip(']')

        if not self.autoencoding and self.conf['model']['last_layer_dense']:
            linears = self.conf['model']['neurons_last_linear']
            s2 = str(linears).replace(', ', '-').strip('[').strip(']')
            s1 = s1 + '+' + s2

        res = '§' + s1 + '§'
        return res

    # UNIQUE TRAIN NAME
    def create_unique_train_name(self):

        #region string dataset
        tipo_grafo = ""
        if self.graphtype != GraphType.REAL:
            numnodi = self.conf['graph_dataset']['Num_nodes']
            if isinstance(numnodi, list):
                numnodi = numnodi[0]
            numgrafi = int(self.conf['graph_dataset']['Num_grafi_per_tipo'])
            if self.graphtype == GraphType.CM:
                tipo_grafo += "CM"
                data_label = self.conf['graph_dataset']['list_exponents']
                #if isinstance(data_label, list):
                #    data_label = f"{len(data_label)}exps"
            elif self.graphtype == GraphType.ER:
                tipo_grafo += "ER"
                data_label = self.conf['graph_dataset']['list_p']
            elif self.graphtype == GraphType.SBM:
                tipo_grafo += "SBM"
                data_label = self.conf['graph_dataset']['community_probs']
            elif self.graphtype == GraphType.Regular:
                tipo_grafo += "Reg"
                data_label = self.conf['graph_dataset']['list_degree']
            elif self.graphtype == GraphType.CONST_DEG_DIST:
                tipo_grafo += "ConstDist"
                data_label = self.conf['graph_dataset']['max_degree_const_dist']
        
        elif self.graphtype == GraphType.REAL:
            numnodi = ""
            numgrafi = ""
            data_label = ""
            tipo_grafo = self.conf['graph_dataset']['real_data_name']

        string_dataset = f"{tipo_grafo.ljust(3, '_')}_{data_label}_Classi{self.num_classes}_nodi{str(numnodi).ljust(3, '_')}_grafiX{str(numgrafi).ljust(4, '_')}"
        long_string_dataset = f"{self.longstring_graphtype} - {numnodi} nodi - {numgrafi} grafi per classe - parametri: {data_label}\n"

        # endregion

        # region string Modello

        modo = self.conf['training']['mode']
        if self.conf['model']['autoencoder']:
            modo = "AE"
        elif self.conf['model']['autoencoder_confmodel']:
            modo = "AE_CM"
        elif self.conf['model']['autoencoder_mlpdecoder']:
            modo = "AE_decMLP"
        elif self.conf['model']['autoencoder_fullMLP']:
            modo = "AE_fullMLP"
        elif self.conf['model']['autoencoder_graph_ae']:
            modo = "MIAGAE"


        if self.conf['model']['autoencoder_fullMLP']:
            freezed = ""
        else:
            freezed = "- GCN freezed" if self.conf['model']['freezeGCNlayers'] else ""

        layer_neuron_string = self.create_layer_neuron_string()
            
        if self.conf['model']['autoencoder_graph_ae'] or self.conf['model']['autoencoder_fullMLP']:
            init_weights = ""
        else:
            init_weights = self.conf['model']['init_weights']

        activation_type = self.conf['model'].get('activation')
        last_activation_type = self.conf['model'].get('last_layer_activation')

        insert_batchnorm = self.conf['model']['put_batchnorm']
        if insert_batchnorm:
            btchnrm = "btchnrmSI"
        else:
            btchnrm = "btchnrmNO"
        if self.conf['model'].get('put_graphnorm'):
            btchnrm = "grphnorm"

        string_model = f"_{modo.ljust(10, '_')}_{layer_neuron_string.ljust(14,'_')}__{activation_type}+{last_activation_type}__{btchnrm}__-{init_weights.ljust(10, '_')}"
        long_string_model = f"Modello {modo} {layer_neuron_string} - activ func: {activation_type}+{last_activation_type} - {btchnrm} - {init_weights} \n"
        # endregion

        # region string trainer
        # percentuale_train = self.conf['training']['percentage_train']
        loss_string = self.conf['training']['loss']
        lr = self.conf['training']['learning_rate']
        optim = self.conf['training'].get('optimizer')
        shuffled = self.conf['training']['shuffle_dataset']

        string_trainer = f"_lr{str(lr).replace('.','')}_{optim}_{loss_string}"  #_shfl{shuffled}"
        long_string_trainer = f"learn rate {str(lr).replace('.','')} - {optim} - {loss_string} {freezed}"
        # endregion


        short_string = string_dataset + "_#_" + string_model + "_#_" + string_trainer
        short_string = short_string.replace(', ', '_').replace('.', '')

        # creo stringa lunga
        long_string = long_string_dataset + long_string_model + long_string_trainer
            
    
        return short_string, long_string

    def check_lungh_numnodes_with_classes(self):
        if not self.conf['graph_dataset']['real_dataset']:
            assert self.length_list_NumNodes == self.num_classes, f"Lunghezza di Num_nodes {self.length_list_NumNodes} != num classi {self.num_classes}, numnodes: {self.NumNodes }"
        else:
            # nel caso di dataset reale non devo controllare che numnodes sia qualcosa,
            # perché non decido io i nodi
            pass

    def check_last_layer_activation_autoencoder(self):
        actv_f = self.conf['model'].get('last_layer_activation')
        if self.conf['model'].get('autoencoder_confmodel'):
            assert actv_f == 'RELU', f"last_layer_activation {actv_f}  -  {actv_f == 'RELU'}"
        if self.conf['model'].get('autoencoder'):
            assert actv_f != 'RELU', "Nel caso di autoencoder con sigmoide(z.zT) la RELU taglia i valori negativi, e dopo la sigmoide non ho mai p=ij minori di 0.5"

    def get_num_classes(self):
        if self.conf['graph_dataset']['ERmodel']:
            self.num_classes = len(self.conf['graph_dataset']['list_p'])
        elif self.conf['graph_dataset']['confmodel']:
            self.num_classes = len(self.conf['graph_dataset']['list_exponents'])
        elif self.conf['graph_dataset']['sbm']:
            self.num_classes = len(self.conf['graph_dataset']['community_probs'])
        elif self.conf['graph_dataset']['regular']:
            self.num_classes = len(self.conf['graph_dataset']['list_degree'])
        elif self.conf['graph_dataset']['const_degree_dist']:
            self.num_classes = 1  # perché è massimamente random

        #print(f"Abbiamo {self.num_classes} classi.")
        #print(self.conf['graph_dataset'])
        elif self.conf['graph_dataset'].get('real_dataset'):
            if self.conf['graph_dataset']['real_data_name'] == 'REDDIT-BINARY':
                self.num_classes = 2
            if self.conf['graph_dataset']['real_data_name'] == 'BACI':
                self.num_classes = 0
        return self.num_classes


    def get_config_dataframe(self):
        if self.conf:
            conf_df = json_normalize(self.conf)
            return conf_df
        else:
            raise Exception("Configuration dictionary not initialized")
        