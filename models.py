import numpy as np
from enum import Enum
import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn.norm.graph_norm import GraphNorm
from torch.nn import ReLU, LeakyReLU, Hardsigmoid, Tanh, ELU, Hardtanh, Sigmoid, Softsign, Identity
from torch_geometric.nn import GCNConv, GAE, VGAE, TopKPooling
#from torch_geometric.nn import global_mean_pool
from torch_geometric import nn
from torch_geometric.nn.aggr.basic import MeanAggregation
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models.autoencoder import InnerProductDecoder

from config_valid import Inits

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# region myGCN
class extended4test_GCN(GCNConv):
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        #print(min(edge_weight))
        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out += self.bias

        return edge_weight


class GCN(torch.nn.Module):
    def __init__(self, config_class, **kwargs):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.GCNneurons_per_layer = self.conf['model']['GCNneurons_per_layer']
        self.neurons_last_linear = self.conf['model']['neurons_last_linear']
        self.node_features_dim = self.conf['model']['node_features_dim']
        self.freezeGCNlayers = self.conf['model'].get('freezeGCNlayers', False)

        self.put_batchnorm = self.conf['model']['put_batchnorm']
        self.put_graphnorm = self.conf['model'].get('put_graphnorm')
        self.put_dropout = self.conf['model'].get('put_dropout', False)
        self.last_linear = self.conf['model']['last_layer_dense']
        self.final_pool_aggregator = self.conf['model']['final_pool_aggregator']

        self.convs = torch.nn.ModuleList()
        self.act_func = torch.nn.ModuleList()
        if self.last_linear:
            self.linears = torch.nn.ModuleList()
        else:
            self.linears = []
        if self.put_dropout:
            self.dropouts = torch.nn.ModuleList()
        #self.pools = torch.nn.ModuleList()
        if self.put_batchnorm or self.put_graphnorm:
            self.GCNbatchnorms = torch.nn.ModuleList()
            self.batchnorms = torch.nn.ModuleList()

        activation_function = get_activ_func_from_config(self.conf['model'].get('activation'))
        self.last_act_func = get_activ_func_from_config(self.conf['model'].get('last_layer_activation'))

        self.dataset_degree_prob_infos = kwargs.get('dataset_degree_prob_infos')

        ###########   COSTRUISCO L'ARCHITETTURA      ###############
        ############################################################
        rangeconv_layers = range(len(self.GCNneurons_per_layer) - 1)
        norm = self.conf['model'].get('normalized_adj')
        for i in rangeconv_layers:
            if self.conf['model'].get('my_normalization_adj'):
                self.convs.append(GCN_custom_norm(self.GCNneurons_per_layer[i], self.GCNneurons_per_layer[i + 1], self.dataset_degree_prob_infos, normalize=False))
            else:
                self.convs.append(GCNConv(self.GCNneurons_per_layer[i], self.GCNneurons_per_layer[i + 1], normalize=norm))
            if self.put_batchnorm:
                self.GCNbatchnorms.append(BatchNorm1d(self.GCNneurons_per_layer[i]))
            elif self.put_graphnorm:
                self.GCNbatchnorms.append(GraphNorm(self.GCNneurons_per_layer[i]))
            self.act_func.append(activation_function)
            #self.pools.append( TopKPooling(neurons_per_layer[i+1], ratio=0.8) )
            #i += 1

        if self.final_pool_aggregator:
            self.mean_pool = MeanAggregation()


        if self.last_linear:
            if self.put_batchnorm:
                self.batchnorms.append(BatchNorm1d(self.GCNneurons_per_layer[-1]))
            if self.put_dropout:
                self.dropouts.append(torch.nn.Dropout(p=0.5))
            #il primo è uguale all'ultimo dei GCN layer
            self.linears.append(Linear(self.GCNneurons_per_layer[-1], self.neurons_last_linear[0]))
            for j in range(0, len(self.neurons_last_linear) - 1):
                if self.put_batchnorm:
                    self.batchnorms.append(BatchNorm1d(self.neurons_last_linear[j]))
                if self.put_dropout:
                    self.dropouts.append(torch.nn.Dropout(p=0.5))
                lin = Linear(self.neurons_last_linear[j], self.neurons_last_linear[j + 1])
                self.linears.append(lin)

            self.last_linears__activation = activation_function

        if self.freezeGCNlayers:
            self.freeze_gcn_layers()


    def freeze_gcn_layers(self):
        #for param in self.parameters():
        #    param.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.GCNConv):
                for p in m.parameters():
                    p.requires_grad = False


    def n_layers_gcn(self, x, edge_index, edge_weight_normalized=None):
        for i, layer in enumerate(self.convs[:-1]):
            x = layer(x, edge_index, edge_weight_normalized)
            if self.put_batchnorm and i > 1:
                x = self.GCNbatchnorms[i](x)
            x = self.act_func[i](x)
            # x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
        x = self.convs[-1](x, edge_index, edge_weight_normalized)
        # così posso specificare una activation func specifica per l'ultimo layer
        x = self.last_act_func(x)
        return x

    def n_layers_linear(self, x):

        for i, layer in enumerate(self.linears):
            x = layer(x)  # spostando questo prima dei batchnorm potrei avere un errore per come son odefiniti i batchnorm,
            # cioè con il numero dei neuroni all' i-esimo batchnorm
            if self.put_batchnorm and i < (len(self.batchnorms) - 1):
                x = self.batchnorms[i](x)
            if self.put_dropout:
                x = self.dropouts[i](x)
            x = self.last_linears__activation(x)

        return x

    # for hidden_size in self.neurons_per_layer:
    #     encoder_layers.append(nn.Linear(prev_size, hidden_size))
    #     encoder_layers.append(activation_function)
    #     prev_size = hidden_size
    #
    # self.encoder = torch.nn.Sequential(*encoder_layers)

    def forward(self, x, edge_index, batch=None, graph_embedding=False, node_embedding=False, edge_weight_normalized=None):
        x = self.n_layers_gcn(x, edge_index, edge_weight_normalized)

        if (node_embedding or self.config_class.autoencoding) and not graph_embedding:
            return x

        #print(f"before aggregator {x.shape}")
        if self.final_pool_aggregator:
            # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            x = self.mean_pool(x, batch)
            # assert x == x2, "Col piffero che sono uguali"
            #print(f"after aggregator {x.shape}")

        if graph_embedding or not self.last_linear:
            return x

        if self.last_linear:
            x = self.n_layers_linear(x)
        #print(f"after linear: {x.shape}")
        return x

class extended4test_myGCNencoder(GCN):
    def forward(self, x, edge_index, **kwargs):
        edge_weigths = self.convs[0](x, edge_index)
        return edge_weigths


class GCN_custom_norm(GCNConv):
    def __init__(self, in_channels, out_channels, degree_prob_infos, normalize):
        super(GCN_custom_norm, self).__init__(in_channels, out_channels, normalize=normalize)

        self.degree_prob_dict = degree_prob_infos[0]
        self.tot_links_per_graph = degree_prob_infos[1]
        self.average_links_per_graph = degree_prob_infos[2]


    def forward(self, x, edge_index, edge_weight):
        # x è la matrice delle features dei nodi
        # edge_index è la matrice di adiacenza (nodo di partenza, nodo di arrivo)

        """ Calcola la matrice dei gradi D
        row, col = edge_index
        node_degrees = row.bincount(minlength=x.size(0))
        probabilities = torch.tensor([1.0 / self.degree_prob_dict.get(int(d), 1.0) for d in node_degrees], device=torch.device('cuda'))
        distance_from_mean = torch.abs(node_degrees - self.average_links_per_graph) / self.tot_links_per_graph
        weights = torch.sqrt(1.0 / (distance_from_mean / probabilities))

        edge_weight_normalized = weights[row] * (edge_weight if edge_weight is not None else 1.0) * weights[col]
        """
        # D = torch.diag(weights)

        # Controllo se edge_weight è None e in tal caso impostare tutti i pesi a 1
        # if edge_weight is None:
        #     edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=x.device)

        # A = torch.sparse_coo_tensor(edge_index, edge_weight, (x.size(0), x.size(0)))
        # A_norm = torch.mm(torch.mm(D, A.to_dense()), D)


        return super(GCN_custom_norm, self).forward(x, edge_index, edge_weight)


# endregion

class CappedReLU(torch.nn.Module):
    def __init__(self):
        super(CappedReLU, self).__init__()
    def forward(self, x):
        return torch.clamp(x, 0, 1)

def get_activ_func_from_config(activation_function_string):
    ''' non la metto nel Config per non dover importare le funzioni di pytorch nel modulo di Config'''
    if activation_function_string == 'ELU':
        activation_function = ELU()
    elif activation_function_string == "RELU":
        activation_function = ReLU()
    elif activation_function_string == "CappedRELU":
        activation_function = CappedReLU()
    elif activation_function_string == "Hardtanh":
        activation_function = Hardtanh(0, 1)
    elif activation_function_string == "Tanh":
        activation_function = Tanh()
    elif activation_function_string == "Softsign":
        activation_function = Softsign()
    elif "*Tanh" in activation_function_string:
        factor = int(activation_function_string.split("*")[0])
        print(f"Tangente iperbolica con fattore {factor}")
        activation_function = lambda x: Tanh()(x) * factor
    elif "*Softsign" in activation_function_string:
        factor = int(activation_function_string.split("*")[0])
        print(f"Softsign con fattore {factor}")
        activation_function = lambda x: Softsign()(x) * factor
    elif activation_function_string == "LeakyRELU":
        activation_function = LeakyReLU(0.05)
    elif activation_function_string == "Sigmoid":
        activation_function = Sigmoid()
    elif activation_function_string == 'Identity':
        activation_function = Identity()
    else:
        raise "Errore nella funzione di attivazione specificata nel config."
    return activation_function


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

def get_parameters(model_layers):
    layers = []
    for lin in model_layers: #xp.trainer.model.linears:
        pars = []
        for par in lin.parameters():
            par = par.detach().cpu().numpy()
            # print(par.flatten().shape)
            pars.extend(par.flatten())
        #print(len(pars))
        layers.append(pars)
    return layers

def get_param_labels(model_layers):
    labels = []
    for i, lin in enumerate(model_layers):  # xp.trainer.model.linears:
        labels.append(f"{lin.__class__.__name__}_{i}")
    return labels

def view_weights_gradients(model):
    gradients = []
    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            ##m.lin.weight.retain_grad()
            gradients.append(m.lin.weight.grad)
    return gradients

def new_parameters(model, method=Inits.xavier_uniform, const_value=1.0, **kwargs):
    new_par = []
    for m in model.modules():
        if isinstance(m, nn.GCNConv):
            shape = m.lin.weight.shape
            custom_weight = torch.empty(shape)
            custom_weight = get_weights_from_init_method(custom_weight, method, const_value=const_value, **kwargs)

            new_par.append(custom_weight)
    return new_par


def get_weights_from_init_method(custom_weight, method, const_value=1.0, **kwargs):
    if kwargs.get('modified_gain_for_UNnormalized_adj'):
        n_params = torch.tensor(custom_weight.numel())
        gain = 1 / torch.sqrt(n_params.float())
    else:
        gain = 1.0
    print(f"Gain for init_weights: {gain}")
    if method is Inits.kaiming_normal:
        gain = torch.nn.init.calculate_gain(nonlinearity='relu', param=None)  # nonlinearity – the non-linear function (nn.functional name)
        torch.nn.init.kaiming_normal_(custom_weight, mode='fan_out', nonlinearity='relu')
    elif method == Inits.kaiming_uniform:
        torch.nn.init.kaiming_uniform_(custom_weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif method == Inits.uniform:
        torch.nn.init.uniform_(custom_weight, a=0, b=1)  # a: lower_bound, b: upper_bound
    elif method == Inits.normal:
        torch.nn.init.normal_(custom_weight, mean=0.0, std=0.4)
    elif method == Inits.constant:
        torch.nn.init.constant_(custom_weight, value=const_value)
    elif method == Inits.eye:
        torch.nn.init.eye_(custom_weight)
    elif method == Inits.dirac:
        torch.nn.init.dirac_(custom_weight)  # tensor deve essere almeno 3D
    elif method == Inits.xavier_uniform:
        torch.nn.init.xavier_uniform_(custom_weight, gain=gain)
    elif method == Inits.xavier_normal:
        torch.nn.init.xavier_normal_(custom_weight, gain=gain)  #gain=1.0
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


# region non li sto usando

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

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
def simpleautoencoder(num_features, out_channels):
    model = GAE(GCNEncoder(num_features, out_channels))
    return model

# endregion


# region MODELLI AUTOENCODER  uso solo i seguenti al momento

class AutoencoderGCN(GCN):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = None  # meglio chiamarlo più propriamente: self.GAE
        self.convs = encoder.convs
        self.linears = encoder.linears
        # self.__dict__.update(dic_attr)

    def set_decoder(self, encoder, decoder=None):
        if decoder is not None:
            # necessario per attaccare correttamente il decoder al GAE
            self.decoder = GAE(encoder, decoder=decoder)
        else:
            self.decoder = GAE(encoder)

    #def forward(self, x, edge_index, batch=None, graph_embedding=False, node_embedding=False):
    #    return self.decoder(x, edge_index, batch, graph_embedding, node_embedding)
    def encode(self, x, edge_index, batch, node_embedding=False, graph_embedding=False, edge_weight_normalized=None):
        return self.decoder.encode(x, edge_index, batch, graph_embedding, node_embedding, edge_weight_normalized)
    def test(self, z, pos_edge_label_index, neg_edge_label_index):
        return self.decoder.test(z, pos_edge_label_index, neg_edge_label_index)
    def recon_loss(self, z, pos_edge_label_index, neg_edge_index=None):
        return self.decoder.recon_loss(z, pos_edge_label_index, neg_edge_index)
    def forward_all(self, z, sigmoid: bool = False):
        output = self.decoder.decoder.forward_all(z, sigmoid=sigmoid)
        return output

    @classmethod
    def from_parent_instance(cls, dic_attr, parent_instance):
        #return cls(dic_attr, **parent_instance.__dict__)
        return cls(encoder=parent_instance, #config_class=parent_instance.config_class,
                   **parent_instance.__dict__,)


# Definiamo la classe dell'autoencoder
class AutoencoderMLP(torch.nn.Module):
    def __init__(self, config_class):
        super(AutoencoderMLP, self).__init__()

        self.config_class = config_class
        self.conf = self.config_class.conf

        in_size = self.conf['graph_dataset']['Num_nodes'][0]
        self.input_size = in_size * in_size  # tutti gli elementi della matrice
        self.neurons_per_layer = self.conf['mlp_ae_model']['neurons_per_layer']
        # devo comunque definire un layer latente intermedio di embedding per avere embedding:dim
        self.embedding_dim = min(self.neurons_per_layer)
        activation_function = get_activ_func_from_config(self.conf['mlp_ae_model'].get('activation'))
        last_act_func = get_activ_func_from_config(self.conf['mlp_ae_model'].get('last_layer_activation'))

        encoder_layers = []
        decoder_layers = []
        prev_size = self.input_size

        # Costruiamo l'encoder
        for hidden_size in self.neurons_per_layer:
            encoder_layers.append(nn.Linear(prev_size, hidden_size))
            encoder_layers.append(activation_function)
            prev_size = hidden_size

        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Costruiamo il decoder (simmetrico all'encoder)
        for hidden_size in reversed(self.neurons_per_layer):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(activation_function)
            prev_size = hidden_size

        decoder_layers.append(nn.Linear(prev_size, self.input_size))
        decoder_layers.append(last_act_func)

        self.decoder = torch.nn.Sequential(*decoder_layers)

        if self.conf['model']['autoencoder_fullMLP_CM']:
            self.final_layer = self.cm_layer
        else:
            self.final_layer = self.linear_layer

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x

    def linear_layer(self, x):
        return x

    def cm_layer(self, z):
        z = z.reshape(z.shape[0], int(np.sqrt(z.shape[-1])), int(np.sqrt(z.shape[-1])))
        res = []
        for v in z:
            adj = torch.matmul(v, v.t())
            value = adj / (1 + adj)
            value.fill_diagonal_(0)
            res.append(v)
        # check_nans(value, z)
        out = torch.stack(res)
        out = out.reshape(out.shape[0], -1)
        return out

class ConfModelDecoder(InnerProductDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, z, edge_index, sigmoid=False):
        # sigmoid non vene mai usato
        sigmoid = False
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        value = value / (1 + value)
        return value

    def forward_all(self, z, sigmoid=False):
        # sigmoid non vene mai usato
        sigmoid = False
        adj = torch.matmul(z, z.t())
        value = adj / (1 + adj)
        value.fill_diagonal_(0)
        #check_nans(value, z)
        return value


class MLPDecoder(torch.nn.Module):
    def __init__(self, config, in_dim, out_dim, **kwargs):
        """
        :param in_dim:  deve essere la dim delle feture del nodo
        :param out_dim:  deve essere il numero di nodi
        in modo che il prodotto venga una matrice num_nodi X num_nodi
        """
        super().__init__(**kwargs)
        self.conf = config
        # prendo i neuroni dal livello linear che usavo per classification
        self.neurons_per_layer = self.conf['model']['neurons_last_linear']
        # prendo le activ.f. dedicate mlp_ae_model
        activation_function = get_activ_func_from_config(self.conf['mlp_ae_model'].get('activation'))
        last_act_func = get_activ_func_from_config(self.conf['mlp_ae_model'].get('last_layer_activation'))

        decoder_layers = []
        prev_size = in_dim
        for hidden_size in reversed(self.neurons_per_layer):
            decoder_layers.append(nn.Linear(prev_size, hidden_size))
            decoder_layers.append(activation_function)
            prev_size = hidden_size
        decoder_layers.append(nn.Linear(prev_size, out_dim))
        decoder_layers.append(last_act_func)
        self.decoder_layers = torch.nn.Sequential(*decoder_layers)

    def forward_all(self, z, sigmoid=False):
        # devo tenere sigmoid per compatibilità con le altre classi
        z = self.decoder_layers(z)
        return z


class MLPCMDecoder(MLPDecoder):
    def __init__(self, config, in_dim, out_dim, **kwargs):
        super().__init__(config, in_dim, out_dim, **kwargs)
        # Inizializzazione aggiuntiva se necessaria

    def forward_all(self, z, sigmoid=False):
        # Prima applica le operazioni di MLPDecoder
        z = super().forward_all(z)

        # Poi esegui la normalizzazione come in ConfModelDecoder
        adj = torch.matmul(z, z.t())
        value = adj / (1 + adj)
        value.fill_diagonal_(0)  # Assicurati che la diagonale sia 0
        return value


def check_nans(input_array, embedding_z):
    nans = torch.isnan(input_array).any()
    if nans:
        print("NANNNNNNS")
    non_finiti = torch.logical_not(torch.isfinite(input_array))
    if non_finiti.any():
        print("non finiti")
        print(input_array)
        print(f"Embedding z chi erano  {embedding_z}")
        print(non_finiti.sum().item() / input_array.numel())
        # adjusted_pred_adj[adjusted_pred_adj != adjusted_pred_adj] = 0.5
        # print(adjusted_pred_adj)
        print("\n")




# endregion