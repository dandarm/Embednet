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


# region myGCN
class GCN(torch.nn.Module):
    #def __init__(self, neurons_per_layer, node_features=1, num_classes=2, autoencoder=False, put_batchnorm=False, mode='classification'):
    def __init__(self, config_class):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.neurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        #self.last_layer_neurons = self.neurons_per_layer[-1]
        self.last_layer_neurons = self.conf['model']['neurons_last_linear']
        self.node_features_dim = self.conf['model']['node_features_dim']
        self.num_classes = self.config_class.num_classes_ER()
        self.autoencoder = self.conf['model']['autoencoder']
        self.put_batchnorm = self.conf['model']['put_batchnorm']
        self.last_layer_dense = self.conf['model']['last_layer_dense']
        self.final_pool_aggregator = self.conf['model']['final_pool_aggregator']

        self.convs = torch.nn.ModuleList()
        self.leakys = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        #self.pools = torch.nn.ModuleList()
        if self.put_batchnorm:
            print('Batch Normalization included')
            self.batchnorms = torch.nn.ModuleList()

        rangeconv_layers = range(len(self.neurons_per_layer) - 1)
        for i in rangeconv_layers:
            self.convs.append(GCNConv(self.neurons_per_layer[i], self.neurons_per_layer[i + 1]))
            self.leakys.append(LeakyReLU(0.03))
            #self.pools.append( TopKPooling(neurons_per_layer[i+1], ratio=0.8) )
            if self.put_batchnorm:
                self.batchnorms.append(BatchNorm1d(self.neurons_per_layer[i + 1]))
            i += 1

        if self.final_pool_aggregator:
            self.mean_pool = MeanAggregation()

        if self.last_layer_dense:
            #il primo è uguale all'ultimo dei GCN layer
            self.linears.append(Linear(self.neurons_per_layer[-1], self.last_layer_neurons[0]))
            self.dropouts.append(torch.nn.Dropout(p=0.5))
            for j in range(0, len(self.last_layer_neurons)-1):
                lin = Linear(self.last_layer_neurons[j], self.last_layer_neurons[j+1])
                self.linears.append(lin)
                drop = torch.nn.Dropout(p=0.5)
                self.dropouts.append(drop)
        else:
            self.lin = None

        # provo con gli oggetti per il grafo
        self.drop = torch.nn.Dropout(p=0.5) #F.dropout

    def n_layers_gcn(self, x, edge_index):
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            # pooling layer after each GCN
            # print(f"Gconv{i}")
            if self.put_batchnorm and i < (len(self.batchnorms) - 1):
                x = self.batchnorms[i](x)
                # print(f"batchnorm{i}")
            x = self.leakys[i](x)
            # print(f"leakyrelu{i}")
            # x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
        return x

    def n_layers_linear(self, x):
        for i, layer in enumerate(self.linears):
            x = self.dropouts[i](x)
            x = layer(x)
        return x

    def forward(self, x, edge_index, batch=None, graph_embedding=False, node_embedding=False):
        x = self.n_layers_gcn(x, edge_index)

        if node_embedding or self.autoencoder:
            return x

        #print(f"before aggregator {x.shape}")
        if self.final_pool_aggregator:
            # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            x = self.mean_pool(x, batch)
            # assert x == x2, "Col piffero che sono uguali"
            #print(f"after aggregator {x.shape}")

        if graph_embedding or not self.last_layer_dense:
            return x

        # 3. Apply a final classifier
        # print(f"Dropout + Linear ")
        #x = self.drop(x)
        #x = self.lin(x)
        if self.last_layer_dense:
            x = self.n_layers_linear(x)
        #print(f"after linear: {x.shape}")
        return x

# endregion

# region imposta i pesi della rete

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


import matplotlib.pyplot as plt
def view_parameters(model):
    i = 0
    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            classname = m.__class__.__name__
            #print(f"classname: {classname} \n\t parameters: {list(m.parameters())} \n \n")
            array = []
            for e in list(m.parameters()):
                array.append(e.cpu().detach().numpy())
            for a in array:
                plt.hist(a.flatten(), alpha=0.5);

            plt.title(f"{classname}_{i}")
            i += 1
            plt.show()
        # print(f"classname: {classname} \n\t parameters: {list(m.parameters())} \n\t bias: {m.bias} \n\t weights: {m.lin.weight} \n \n \n")


def new_parameters(model, method=Inits.xavier_uniform, const_value=1.0):
    new_par = []

    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            shape = m.lin.weight.shape
            custom_weight = torch.empty(shape)
            if method is Inits.kaiming_normal:
                gain = torch.nn.init.calculate_gain(nonlinearity='relu', param=None)  # nonlinearity – the non-linear function (nn.functional name)
                torch.nn.init.kaiming_normal_(custom_weight, mode='fan_out', nonlinearity='relu')
            elif method == Inits.kaiming_uniform:
                torch.nn.init.kaiming_uniform_(custom_weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif method == Inits.uniform:
                torch.nn.init.uniform_(custom_weight, a=0, b=1)  # a: lower_bound, b: upper_bound
            elif method == Inits.normal:
                torch.nn.init.normal_(custom_weight, mean=0, std=1)
            elif method == Inits.constant:
                torch.nn.init.constant_(custom_weight, value=const_value)
            elif method == Inits.eye:
                torch.nn.init.eye_(custom_weight)
            elif method == Inits.dirac:
                torch.nn.init.dirac_(custom_weight)  # tensor deve essere almeno 3D
            elif method == Inits.xavier_uniform:
                torch.nn.init.xavier_uniform_(custom_weight, gain=1.0)
            elif method == Inits.xavier_normal:
                torch.nn.init.xavier_normal_(custom_weight, gain=1.0)
            elif method == Inits.trunc_normal:
                torch.nn.init.trunc_normal_(custom_weight, mean=0.0, std=1.0, a=- 2.0, b=2.0)
            elif method == Inits.orthogonal:
                torch.nn.init.orthogonal_(custom_weight, gain=1)
            elif method == Inits.sparse:
                torch.nn.init.sparse_(custom_weight, sparsity=0.5, std=0.01)  # The fraction of elements in each column to be set to zero
            elif method == 'esn':
                W = ESN_property(shape)
                custom_weight = torch.Tensor(W)

            new_par.append(custom_weight)
    return new_par



def ESN_property(array_shape):
    #N, N = array_shape
    #W = np.random.rand(N, N) - 0.5
    W = np.random.normal(0, 1, array_shape)
    W[np.random.rand(*array_shape) < 0.5] = 0
    #print(W)
    if W.ndim == 1:
        W2 = W[:,np.newaxis]
    if W.shape[0] != W.shape[1]: # e se è vero l'if sopra sarà vero anche questo
        W2 = np.dot(W.T, W)
    else:
        W2 = W
    autovalori = np.linalg.eig(W2)[0]
    radius = np.max(np.abs(autovalori))#.reshape(array_shape))))
    print(radius)
    W = W * (0.95 / radius)
    return W

def modify_parameters(model, new_par, device=torch.device('cuda')):
    i = 0
    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            shape = m.lin.weight.shape
            # custom_weight = torch.empty(shape)
            # 4torch.nn.init.kaiming_normal_(custom_weight, mode='fan_out', nonlinearity='relu')
            par = torch.nn.parameter.Parameter(new_par[i]).to(device)
            m.lin.weight.data = par
            i += 1

# endregion


# region MODELLI AUTOENCODER

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def GAEGCNEncoder(neurons_per_layer, node_features=1, num_classes=2, autoencoder=True, put_batchnorm=True):
    model = GAE(GCN(neurons_per_layer, node_features, num_classes, autoencoder, put_batchnorm))
    return model

# endregion