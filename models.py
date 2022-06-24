import torch
from torch.nn import Linear, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GCNConv, GAE, VGAE, TopKPooling
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    #def __init__(self, neurons_per_layer, node_features=1, num_classes=2, autoencoder=False, put_batchnorm=False, mode='classification'):
    def __init__(self, config_class):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.neurons_per_layer = self.conf['model']['neurons_per_layer']
        self.last_layer_neurons = self.neurons_per_layer[-1]
        self.node_features_dim = self.conf['model']['node_features_dim']
        self.num_classes = self.config_class.num_classes_ER()
        self.autoencoder = self.conf['model']['autoencoder']
        self.put_batchnorm = self.conf['model']['put_batchnorm']
        self.last_layer_dense = self.conf['model']['last_layer_dense']

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        if self.put_batchnorm:
            print('Batch Normalization included')
            self.batchnorms = torch.nn.ModuleList()

        if self.last_layer_dense:
            rangeconv_layers = range(len(self.neurons_per_layer) - 2)
            self.lin = Linear(self.neurons_per_layer[-2], self.last_layer_neurons)  # num_classes corrisponde a last_layer_neurons
        else:
            rangeconv_layers = range(len(self.neurons_per_layer) - 1)
            self.lin = None
        for i in rangeconv_layers:
            self.convs.append(GCNConv(self.neurons_per_layer[i], self.neurons_per_layer[i + 1]))
            #self.pools.append( TopKPooling(neurons_per_layer[i+1], ratio=0.8) )
            if self.put_batchnorm:
                self.batchnorms.append(BatchNorm1d(self.neurons_per_layer[i + 1]))
            i += 1

        self.leaky = LeakyReLU(0.03)

    def embeddings(self, x, edge_index, batch=None):

        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
            # pooling layer after each GCN
            # print(f"Gconv{i}")
            if self.put_batchnorm and i < (len(self.batchnorms) - 1):
                x = self.batchnorms[i](x)
                # print(f"batchnorm{i}")
            # x = x.relu()
            x = self.leaky(x)
            # print(f"leakyrelu{i}")
            #x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)

        if not self.autoencoder:
            # Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        return x

    def forward(self, x, edge_index, batch=None, embedding=False):
        x = self.embeddings(x, edge_index, batch)

        if not self.autoencoder and not embedding and self.last_layer_dense:
            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            # print(f"Dropout + Linear ")

        return x


# class GCNEmbed(GCN):
#     def __init__(self, model, neurons_per_layer):
#         super().__init__(neurons_per_layer)
#         self.num_neurons_last_layer = neurons_per_layer[-1]
#
#         self.convs = model._modules['convs']
#         # self.conv2 = model._modules['conv2']
#         # self.conv3 = model._modules['conv3']
#
#     def forward(self, x, edge_index, batch):
#         x = self.embeddings(x, edge_index, batch)
#
#         return x


# for autoencoder

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