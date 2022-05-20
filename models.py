import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class GCN(torch.nn.Module):
    def __init__(self, neurons_per_layer, node_features=1, num_classes=2):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)

        self.neurons_per_layer = neurons_per_layer
        self.num_neurons_last_layer = neurons_per_layer[-1]
        self.node_features = node_features
        self.num_classes = num_classes

        self.convs = torch.nn.ModuleList()
        for i in range(len(neurons_per_layer)-1):
            self.convs.append( GCNConv(neurons_per_layer[i], neurons_per_layer[i+1]) )
            #self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
            #self.conv3 = GCNConv(self.hidden_channels, self.num_neurons_last_layer)
            i += 1
        if self.num_neurons_last_layer > 1:
            self.lin = Linear(self.num_neurons_last_layer, self.num_classes)
        else:
            self.lin = None
        
    def embeddings(self, x, edge_index, batch):
        for layer in self.convs:
            x = layer(x, edge_index)
            x = x.relu()

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x

    def forward(self, x, edge_index, batch):
        x = self.embeddings(x, edge_index, batch)

        if self.num_neurons_last_layer > 1:
            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
        
        return x
    
    
class GCNEmbed(GCN):
    def __init__(self, model, neurons_per_layer):
        super().__init__(neurons_per_layer)
        self.num_neurons_last_layer = neurons_per_layer[-1]
        
        self.convs = model._modules['convs']
        #self.conv2 = model._modules['conv2']
        #self.conv3 = model._modules['conv3']
        
    
    def forward(self, x, edge_index, batch):
        x = self.embeddings(x, edge_index, batch)
        
        return x
    

# class GCN1n(GCN):
#     def __init__(self, neurons_per_layer, node_features=1):
#         super(GCN, self).__init__()
#
#         self.neurons_per_layer = neurons_per_layer
#         self.node_features = node_features
#
#         self.conv1 = GCNConv(self.node_features, self.hidden_channels)
#         self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
#         self.conv3 = GCNConv(self.hidden_channels, 1)
#
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#
#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#         return x