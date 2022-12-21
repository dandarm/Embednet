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

# region myGCN
class GCN(torch.nn.Module):
    #def __init__(self, neurons_per_layer, node_features=1, num_classes=2, autoencoder=False, put_batchnorm=False, mode='classification'):
    def __init__(self, config_class):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.GCNneurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        self.neurons_last_linear = self.conf['model']['neurons_last_linear']
        self.node_features_dim = self.conf['model']['node_features_dim']
        self.freezeGCNlayers = self.conf['model'].get('freezeGCNlayers', False)

        self.autoencoder = self.conf['model']['autoencoder']
        self.put_batchnorm = self.conf['model']['put_batchnorm']
        self.put_dropout = self.conf['model'].get('put_dropout', False)
        self.last_linear = self.conf['model']['last_layer_dense']
        self.final_pool_aggregator = self.conf['model']['final_pool_aggregator']

        self.convs = torch.nn.ModuleList()
        self.leakys = torch.nn.ModuleList()
        if self.last_linear:
            self.linears = torch.nn.ModuleList()
        if self.put_dropout:
            self.dropouts = torch.nn.ModuleList()
        #self.pools = torch.nn.ModuleList()
        if self.put_batchnorm:
            self.batchnorms = torch.nn.ModuleList()

        ###########   COSTRUISCO L'ARCHITETTURA      ###############
        ############################################################
        rangeconv_layers = range(len(self.GCNneurons_per_layer) - 1)
        for i in rangeconv_layers:
            self.convs.append(GCNConv(self.GCNneurons_per_layer[i], self.GCNneurons_per_layer[i + 1]))
            self.leakys.append(LeakyReLU(0.03))
            #self.pools.append( TopKPooling(neurons_per_layer[i+1], ratio=0.8) )
            #i += 1

        if self.final_pool_aggregator:
            self.mean_pool = MeanAggregation()

        if self.last_linear:
            if self.put_dropout:
                self.dropouts.append(torch.nn.Dropout(p=0.5))
            if self.put_batchnorm:
                self.batchnorms.append(BatchNorm1d(self.GCNneurons_per_layer[-1]))
            #il primo è uguale all'ultimo dei GCN layer
            self.linears.append(Linear(self.GCNneurons_per_layer[-1], self.neurons_last_linear[0]))
            if self.put_batchnorm:
                self.batchnorms.append(BatchNorm1d(self.neurons_last_linear[0]))
            for j in range(0, len(self.neurons_last_linear) - 1):
                lin = Linear(self.neurons_last_linear[j], self.neurons_last_linear[j + 1])
                self.linears.append(lin)
                if self.put_dropout:
                    self.dropouts.append(torch.nn.Dropout(p=0.5))
                if self.put_batchnorm:
                    self.batchnorms.append(BatchNorm1d(self.neurons_last_linear[j + 1]))
            self.last_relu = LeakyReLU(0.03)

        if self.freezeGCNlayers:
            self.freeze_gcn_layers()

    def freeze_gcn_layers(self):
        #for param in self.parameters():
        #    param.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.GCNConv):
                for p in m.parameters():
                    p.requires_grad = False


    def n_layers_gcn(self, x, edge_index):
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            # print(f"Gconv{i}")
            x = self.leakys[i](x)
            # print(f"leakyrelu{i}")
            # x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
        return x

    def n_layers_linear(self, x):
        if self.put_batchnorm:
            x = self.batchnorms[0](x)
        for i, layer in enumerate(self.linears):
            if self.put_dropout:
                x = self.dropouts[i](x)
            x = layer(x)
            x = self.last_relu(x)
            if self.put_batchnorm:  # and i < (len(self.batchnorms) - 1):
                x = self.batchnorms[i+1](x)

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

        if graph_embedding or not self.last_linear:
            return x

        # 3. Apply a final classifier
        if self.last_linear:
            x = self.n_layers_linear(x)
        #print(f"after linear: {x.shape}")
        return x

# endregion

# region imposta i pesi della rete




import matplotlib.pyplot as plt
def view_parameters(model, nomefile=None, verbose=False):
    """
    posso anche accedere col nome dell'attributo della classe che ho scelto io:
        self.convs = torch.nn.ModuleList()
        self.leakys = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
    :param model:
    :param verbose:
    :return:
    """
    k = 0
    num_gcn_layers = len(model.conf['model']['GCNneurons_per_layer']) - 1
    #fig, axs = plt.subplots(nrows=num_gcn_layers//4, ncols=4, figsize=(20, 4))
    fig, axs = plt.subplots(nrows=1, ncols=num_gcn_layers, figsize=(20, 4))
    for m in model.modules():
        classname = m.__class__.__name__
        if isinstance(m, nn.GCNConv):
            if verbose:
                print("GCN:")
                print(f"classname: {classname} \n\t parameters: {list(m.parameters())} \n \n")
            array = []
            for e in list(m.parameters()):
                array.append(e.cpu().detach().numpy())
            for a in array:
                #axs[k//4,k%4].hist(a.flatten(), alpha=0.5);
                axs[k].hist(a.flatten(), alpha=0.5, bins=50);
                axs[k].set_title(f"{classname}_{k}")

            k += 1
        else:
            if verbose:
                print("non GCN:")
                print(f"classname: {classname} \n\t parameters: {list(m.parameters())} \n \n")
    plt.suptitle(f"Init: {model.conf['model']['init_weights']}")
    plt.tight_layout()
    if nomefile:
        plt.savefig(nomefile)
    plt.show()
        # print(f"classname: {classname} \n\t parameters: {list(m.parameters())} \n\t bias: {m.bias} \n\t weights: {m.lin.weight} \n \n \n")
    if verbose:
        for name, param in model.named_parameters():
            print(name, f"freezed? {not param.requires_grad}")

def new_parameters(model, method=Inits.xavier_uniform, const_value=1.0):
    new_par = []

    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            shape = m.lin.weight.shape
            custom_weight = torch.empty(shape)
            custom_weight = get_weights_from_init_method(custom_weight, method, const_value=const_value)

            new_par.append(custom_weight)
    return new_par


def get_weights_from_init_method(custom_weight, method, const_value=1.0):
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
        W = ESN_property(custom_weight.shape)
        custom_weight = torch.Tensor(W)
    return custom_weight


def new_parameters_linears(model, method=Inits.kaiming_normal):
    new_par = []
    for m in model.linears:
        shape = m.weight.shape
        custom_weight = torch.empty(shape)
        if method is Inits.kaiming_normal:
            torch.nn.init.kaiming_normal_(custom_weight, mode='fan_out', nonlinearity='relu')
        elif method == Inits.kaiming_uniform:
            torch.nn.init.kaiming_uniform_(custom_weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
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

def modify_parameters_linear(model, new_par, device=torch.device('cuda')):
    i = 0
    for m in model.linears:
        par = torch.nn.parameter.Parameter(new_par[i]).to(device)
        m.weight.data = par
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