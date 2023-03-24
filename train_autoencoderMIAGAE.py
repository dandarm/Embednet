import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from models import GCN, AutoencoderGCN, view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear
from model_MIAGAE import AutoencoderMIAGAE
from Dataset_autoencoder import DatasetAutoencoder, DatasetAutoencoderReady

class Trainer_AutoencoderMIAGAE(Trainer):
    def __init__(self, config_class, verbose=False):
        super().__init__(config_class, verbose)
        
        # TODO.... poi posso impostare il criterion dentro il config
        self.criterion_autoenc = nn.MSELoss()
        self.num_nodes_per_graph = self.conf['graph_dataset']['Num_nodes']
        self.num_graphs = self.conf['graph_dataset']['Num_grafi_per_tipo'] * self.config_class.num_classes
        if isinstance(self.num_nodes_per_graph, list):
            self.num_nodes_per_graph = self.num_nodes_per_graph[0]


    def init_GCN(self, init_weights_gcn=None, init_weights_lin=None, verbose=False):
        """
        Returns the GCN model given the class of configurations
        :param config_class:
        :param verbose:
        :return:
        """
        if verbose: print("Initialize model")
        if self.config_class.conf['device'] == 'gpu':
            device = torch.device('cuda')
        else:
            device = "cpu"

        model = AutoencoderMIAGAE(self.config_class)
        model.to(device)
        
        #if init_weights_gcn is not None:
        #    modify_parameters(model, init_weights_gcn)
        #if init_weights_lin is not None:
        #    modify_parameters_linear(model, init_weights_lin)
        if verbose:
            print(model)
        return model
    
    #def init_dataset():
    #    real_dataset

    def load_dataset(self, dataset, parallel=False):  # dataset è di classe GeneralDataset
        print("Loading Dataset...")
        self.dataset = DatasetAutoencoderReady(self.percentage_train, self.batch_size, self.device, self.config_class, dataset)
        
    
        
              
    def train(self):
        self.model.train()
        running_loss = 0
        for data in self.dataset.train_loader:    
            total_batch_z, _, _, _ = self.model(data)
            loss_on_features = self.criterion_autoenc(total_batch_z, data.x)
            
            #adjusted_pred_adj = self.calc_inner_prod_for_batches(total_batch_z, self.num_nodes_per_graph)
            #input_adj = to_dense_adj(data.edge_index, data.batch)            
            #loss_on_recon_edges = self.criterion_autoenc(adjusted_pred_adj, input_adj)
            
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            # self.scheduler.step()
            running_loss += loss_on_features.item()
        return running_loss / self.dataset.train_len

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for data in loader:                
                total_batch_z, _, _, _ = self.model(data)
                loss_on_features = self.criterion_autoenc(total_batch_z, data.x)
                #adjusted_pred_adj = self.calc_inner_prod_for_batches(total_batch_z, self.num_nodes_per_graph)
                #input_adj = to_dense_adj(data.edge_index, data.batch)
                #loss_on_recon_edges = self.criterion_autoenc(adjusted_pred_adj, input_adj)
                running_loss += loss_on_features.item()

        return running_loss / self.dataset.test_len

    def get_embedding(self, loader, type_embedding='both'):
        self.model.eval()
        graph_embeddings_array = []
        node_embeddings_array = []
        node_embeddings_array_id = []
        final_output = []
        
        with torch.no_grad():
            for data in loader:
                if type_embedding == 'graph':
                    out = self.model.encode(data.x, data.edge_index, data.batch, graph_embedding=True)
                    to = out.detach().cpu().numpy()
                    graph_embeddings_array.extend(to)
                elif type_embedding == 'node':
                    out = self.model.encode(data.x, data.edge_index, data.batch, node_embedding=True)
                    to = out.detach().cpu().numpy()
                    node_embeddings_array.extend(to)
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
                    node_out = self.model.encode(data.x, data.edge_index, data.batch, node_embedding=True)
                    to = node_out.detach().cpu().numpy()
                    #print(f"node emb size: {to.nbytes}")
                    node_embeddings_array.extend(to)
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                    graph_out = self.model.encode(data.x, data.edge_index, data.batch, graph_embedding=True)
                    to = graph_out.detach().cpu().numpy()
                    #print(f"graph emb size: {to.nbytes}")
                    graph_embeddings_array.extend(to)

                    final_output.extend([None]*len(to))

        graph_embeddings_array = np.array(graph_embeddings_array)
        node_embeddings_array = np.array(node_embeddings_array)
        return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output
    
    def get_recon_adjs(self, loader):
        self.model.eval()
        adjs_list = []
        feats = []
        with torch.no_grad():
            for data in loader:
                total_batch_z = self.model.encode(data.x, data.edge_index, data.batch) 
                adjusted_pred_adj = self.calc_inner_prod_for_batches(total_batch_z, self.num_nodes_per_graph, self.num_graphs)
                recon_adjs = adjusted_pred_adj.detach().cpu().numpy()
                adjs_list.extend(recon_adjs)
                
                #anche per le feature x
                for i in range(0, len(data.x), self.num_nodes_per_graph):
                    x = data.x[i:i+self.num_nodes_per_graph]
                    x = x.detach().cpu().numpy()
                    feats.append(x)
                
        return np.array(adjs_list), np.array(feats)

    def calc_metric(self, loader):
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                z = self.model.encode(data.x, data.edge_index, data.batch)
                auc, ap = self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

        return auc  #, ap
