import numpy as np
from enum import Enum
import torch
from torch.nn import Linear, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GCNConv, GAE, VGAE, TopKPooling
#from torch_geometric.nn import global_mean_pool
from torch_geometric import nn
from torch_geometric.nn.aggr.basic import MeanAggregation
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from config_valid import Inits

from models import GCN
import sys
sys.path.insert(0,'../Graph_AE/')
sys.path.insert(0,'../Graph_AE/classification')
from classification.Graph_AE import Net as MIAGAE




# region MODELLI AUTOENCODER

class AutoencoderMIAGAE(GCN):
    def __init__(self, config_class, **kwargs):
        super().__init__(config_class, **kwargs)
        #self.encoder = encoder
        #self.decoder = None
        #self.convs = encoder.convs
        #self.linears = encoder.linears
        # self.__dict__.update(dic_attr)
        
        graph_ae_configs = self.conf.get('graph_ae_model')
        if graph_ae_configs is None:
            assert False, "Parametri di configurazione mancanti per il modello GRAPH_AE"
        num_kernels = graph_ae_configs['num_kernels']
        depth = graph_ae_configs['depth']
        comp_rate = graph_ae_configs['comp_rate']
        GCNneurons_per_layer = graph_ae_configs['GCNneurons_per_layer']
        device = self.conf['device']
        self.model = MIAGAE(GCNneurons_per_layer[0], num_kernels, depth, [comp_rate] * depth, GCNneurons_per_layer[1:], device).to(device)
    
    def forward(self, data):
        return self.model(data)
        

    #def set_decoder(self, encoder):
    #    self.decoder = GAE(encoder)

    #def forward(self, x, edge_index, batch=None, graph_embedding=False, node_embedding=False):
    #    return self.decoder(x, edge_index, batch, graph_embedding, node_embedding)
    #def encode(self, x, edge_index, batch, node_embedding=False, graph_embedding=False):
    #    return self.decoder.encode(x, edge_index, batch, graph_embedding, node_embedding)
    #def test(self, z, pos_edge_label_index, neg_edge_label_index):
    #    return self.decoder.test(z, pos_edge_label_index, neg_edge_label_index)
    #def recon_loss(self, z, pos_edge_label_index, neg_edge_index=None):
    #    return self.decoder.recon_loss(z, pos_edge_label_index, neg_edge_index)
    #def forward_all(self, z, sigmoid: bool = True):
    #    return self.decoder.decoder.forward_all(z)

    @classmethod
    def from_parent_instance(cls, dic_attr, parent_instance):
        #return cls(dic_attr, **parent_instance.__dict__)
        return cls(encoder=parent_instance, config_class=parent_instance.config_class)

# endregion