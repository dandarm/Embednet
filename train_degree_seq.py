import numpy as np
import torch
from torch.nn import MSELoss, BCELoss, Module
from train_autoencoder_inductive import Trainer_Autoencoder
from models import GCN, AutoencoderGCN, ConfModelDecoder, MLPDecoder
from Metrics import Metrics
class Trainer_Degree_Sequence(Trainer_Autoencoder):
    def __init__(self, config_class, verbose=False, rootsave="."):
        super().__init__(config_class, verbose, rootsave)
        #self.name_of_metric = ["auc", "pr_auc", "f1_score", "euclid"]
        self.name_of_metric = ["BCE"]

        self.criterion = MSELoss()
        self.bce = BCELoss()

    def calc_loss(self, adjusted_pred_adj, input_adj):
        degree_sequence_output = torch.sum(adjusted_pred_adj, dim=1)
        degree_sequence_input = torch.sum(input_adj, dim=1)
        loss = self.criterion(degree_sequence_output, degree_sequence_input)
        return loss

    def init_decoder(self, encoder, model):
        #if self.config_class.conf['model']['autoencoder']:
        #    model.set_decoder(encoder)
        #elif self.config_class.conf['model'].get('autoencoder_confmodel'):
        model.set_decoder(encoder, ConfModelDecoder())

    def calc_metric(self, actual_node_class, embeddings):
        """calcoliamo solo la BCE come nella funzione test del train_autoencoder_inductive"""

        input_adjs = np.array([g.input_adj_mat for g in embeddings])#.ravel().squeeze()
        adjusted_pred_adj = np.array([g.pred_adj_mat for g in embeddings])#.ravel().squeeze()

        adjusted_pred_adj_r = adjusted_pred_adj.ravel()
        input_adj_r = input_adjs.ravel()
        if not torch.is_tensor(adjusted_pred_adj_r):
            adjusted_pred_adj_r = torch.tensor(adjusted_pred_adj_r)
        if not torch.is_tensor(input_adj_r):
            input_adj_r = torch.tensor(input_adj_r)
        total_bce_loss = self.bce(adjusted_pred_adj_r, input_adj_r)

        metriche = Metrics(BCE=total_bce_loss)

        return metriche


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = BCELoss()
        self.mse = MSELoss()

    def forward(self, outputs, targets, auxiliary_targets):
        # Calcola BCE loss
        bce = self.bce(outputs, targets)
        # Calcola MSE loss
        mse = self.mse(outputs, auxiliary_targets)
        # Somma le due loss
        loss = bce + mse
        return loss



class Trainer_BCEMSE(Trainer_Autoencoder):
    def __init__(self, config_class, verbose=False, rootsave="."):
        super().__init__(config_class, verbose, rootsave)

        self.name_of_metric = ["BCE", "MSE"]
        #self.criterion = CustomLoss()
        self.bce = BCELoss()
        self.mse = MSELoss()

    def init_decoder(self, encoder, model):
        if self.config_class.conf['model']['autoencoder']:
            model.set_decoder(encoder)
        elif self.config_class.conf['model'].get('autoencoder_confmodel'):
            model.set_decoder(encoder, ConfModelDecoder())
    def calc_loss(self, adjusted_pred_adj, input_adj):

        N = input_adj.shape[0]
        #MSEloss
        degree_sequence_output = torch.sum(adjusted_pred_adj, dim=1)
        degree_sequence_input = torch.sum(input_adj, dim=1)
        loss_mse = self.mse(degree_sequence_output, degree_sequence_input)

        #BCELoss
        adjusted_pred_adj_r = adjusted_pred_adj.ravel()
        input_adj_r = input_adj.ravel()
        #self.debug_ATen(adjusted_pred_adj, input_adj)
        loss_bce = self.bce(adjusted_pred_adj_r, input_adj_r)

        # Somma le due loss
        loss = loss_bce + loss_mse
        return loss

    def calc_metric(self, actual_node_class, embeddings):
        """calcolo la BCE e la MSE che sono le due stesse componenti della loss"""

        input_adj = np.array([g.input_adj_mat for g in embeddings])#.ravel().squeeze()
        adjusted_pred_adj = np.array([g.pred_adj_mat for g in embeddings])#.ravel().squeeze()

        if not torch.is_tensor(adjusted_pred_adj):
            adjusted_pred_adj = torch.tensor(adjusted_pred_adj)
        if not torch.is_tensor(input_adj):
            input_adj = torch.tensor(input_adj)
        adjusted_pred_adj_r = adjusted_pred_adj.ravel()
        input_adj_r = input_adj.ravel()

        total_bce_loss = self.bce(adjusted_pred_adj_r, input_adj_r)

        #input_seq = np.array(actual_node_class).ravel()
        #pred_seq = np.array([g.out_degree_seq for g in embeddings]).ravel().squeeze()
        degree_sequence_output = torch.sum(adjusted_pred_adj, dim=1)
        degree_sequence_input = torch.sum(input_adj, dim=1)
        msevalue = self.mse(degree_sequence_output, degree_sequence_input)

        metriche = Metrics(BCE=total_bce_loss, MSE=msevalue)

        return metriche