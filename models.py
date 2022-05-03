import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, node_features = 1, num_classes = 2):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        
        self.hidden_channels = hidden_channels
        self.node_features = node_features
        self.num_classes = num_classes
        
        self.conv1 = GCNConv(self.node_features, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels) # cambiare quì la dimensione dell'ultimo layer in funzione di un parametro
        self.lin = Linear(self.hidden_channels, self.num_classes)
        
    def embeddings(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x
        

    def forward(self, x, edge_index, batch):
        x = self.embeddings(x, edge_index, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
    
class GCNEmbed(GCN):
    def __init__(self, model, hidden_channels=64, node_features = 1, num_classes = 2):
        super().__init__(hidden_channels, node_features, num_classes)  # devo ricopiare i parametri?
        
        self.conv1 = model._modules['conv1']
        self.conv2 = model._modules['conv2']
        self.conv3 = model._modules['conv3']
        
    
    def forward(self, x, edge_index, batch):
        x = self.embeddings(x, edge_index, batch)
        
        return x