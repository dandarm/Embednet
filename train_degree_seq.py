import torch
from torch.nn import MSELoss, BCELoss
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
        model.set_decoder(encoder, ConfModelDecoder())

    def calc_metric(self, loader):
        """calcoliamo solo la BCE come nella funzione test del train_autoencoder_inductive"""
        self.model.eval()
        running_loss = 0
        num_nodes_batch = [len(data.x) for data in loader.dataset]

        i = 0
        num_batches = 0
        with torch.no_grad():
            for data in loader:
                total_batch_z, adjusted_pred_adj, input_adj = self.encode_decode_inputadj(
                    data, i, loader.batch_size, num_nodes_batch)

                adjusted_pred_adj_r = adjusted_pred_adj.ravel()
                input_adj_r = input_adj.ravel()
                loss = self.bce(adjusted_pred_adj_r, input_adj_r)

                running_loss += loss.item()
                i += loader.batch_size
                num_batches += 1

        metriche = Metrics(BCE=running_loss / num_batches)

        return metriche