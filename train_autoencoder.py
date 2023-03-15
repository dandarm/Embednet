import numpy as np
import torch
from train import Trainer
from models import GCN, AutoencoderGCN, view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear
from Dataset_autoencoder import DatasetAutoencoder

class Trainer_Autoencoder(Trainer):
    def __init__(self, config_class, verbose=False):
        super().__init__(config_class, verbose)


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

        encoder = GCN(self.config_class)
        model = AutoencoderGCN.from_parent_instance(dic_attr="dict_attr", parent_instance=encoder)
        model.set_decoder(encoder)

        model.to(device)
        if init_weights_gcn is not None:
            modify_parameters(model, init_weights_gcn)
        if init_weights_lin is not None:
            modify_parameters_linear(model, init_weights_lin)
        if verbose:
            print(model)
        return model

    def train(self):
        self.model.train()
        running_loss = 0

        for data in self.dataset.train_loader:
            z = self.model.encode(data.x, data.edge_index, data.batch)
            loss = self.model.recon_loss(z, data.pos_edge_label_index)
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            # self.scheduler.step()
            running_loss += loss.item()
        return running_loss / self.dataset.train_len

    def load_dataset(self, dataset, parallel=False):  # dataset è di classe GeneralDataset
        print("Loading Dataset...")
        self.dataset = DatasetAutoencoder.from_super_instance(self.percentage_train, self.batch_size, self.device, self.config_class, dataset)
        self.dataset.prepare(self.shuffle_dataset, parallel)

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for data in loader:
                z = self.model.encode(data.x, data.edge_index, data.batch)
                auc, ap = self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
                #loss = self.model.recon_loss(z, data.pos_edge_label_index)
                #running_loss += loss.item()
                #print(f"out and target shape")
                #print(emb.shape, target.shape)

        graph_embeddings_array, node_embeddings_array, _, final_output = self.get_embedding(loader)

        #calcola la PCA
        #obj = PCA(graph_embeddings_array)
        #var_exp, _, _ = obj.get_ex_var()
        #var_exp = torch.as_tensor(np.array(var_exp))
        #var_exp = self.myExplained_variance(emb, target)  # sul singolo batch
        return running_loss / self.dataset.test_len, 0, graph_embeddings_array, node_embeddings_array

    def get_embedding(self, loader, type_embedding='both'):
        self.model.eval()
        graph_embeddings_array = []
        node_embeddings_array = []
        node_embeddings_array_id = []
        final_output = []

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

    def accuracy(self, loader):
        return None
