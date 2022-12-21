import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from train import Trainer
from config_valid import TrainingMode


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum(axis=0)

def softmax_memsafe(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=0)


class Inspect():
    def __init__(self, model, trainer):
        self.target_shuffled = None
        self.total_loss = None
        self.total_loss_mia = None
        self.targets = []
        self.graph_emb = []
        self.output = []

        self.model = model
        self.model.eval()
        self.trainer = trainer
        self.mode = trainer.config_class.modo
        #print(f"Loss function usata: {self.trainer.criterion}")

    def results(self):
        self.get_loss()
        print(f"total_loss    : {self.total_loss}")
        print(f"total_loss mia: {self.total_loss_mia}")

        #self.plot_embedding_targets(self.graph_emb, self.targets)

        print("Loss with shuffled targets")
        self.get_loss(shuffled=True)
        print(f"total_loss:     {self.total_loss}")
        print(f"total_loss mia: {self.total_loss_mia}")
        #self.plot_embedding_targets(self.graph_emb, self.target_shuffled.to('cpu'))

    def get_loss(self, shuffled=False):

        #all_data_loader = DataLoader(self.trainer.dataset.dataset_pyg, batch_size=self.trainer.dataset.bs, shuffle=False)
        graph_emb = []
        targets = []
        running_loss = 0
        running_loss_mia = 0

        for data in self.trainer.dataset.test_loader:
            out = self.model(data.x, data.edge_index, data.batch)
            target = self.trainer.correct_shape(data.y)

            if shuffled:
                target = self.shuffle_target(target)
                self.target_shuffled = target
                target = torch.unsqueeze(torch.Tensor(target), dim=1).to('cuda')

            loss = self.trainer.criterion(out, target)

            if self.mode == TrainingMode.mode2:
                out_cpu = np.array(out.cpu().detach().numpy()).flatten()
                target_cpu = np.array(target.cpu().detach().numpy()).flatten()
                loss_mia = BinaryCrossEntropy(target_cpu, sigmoid(out_cpu))

            graph_emb.extend(out_cpu)
            targets.extend(target_cpu)

            running_loss += loss.item()
            running_loss_mia += loss_mia

        self.total_loss = running_loss / self.trainer.dataset.test_len
        self.total_loss_mia = running_loss_mia / self.trainer.dataset.test_len
        return

    # def reshape_tensors(self, graph_emb_gpu, targets_gpu):
    #     if self.mode == TrainingMode.mode2:
    #         self.graph_emb_gpu = torch.reshape(torch.Tensor(graph_emb_gpu), (-1, 1))
    #         self.targets_gpu = torch.reshape(torch.Tensor(targets_gpu), (-1, 1))
    #     elif self.mode == TrainingMode.mode1:
    #         self.graph_emb_gpu = torch.stack(graph_emb_gpu)
    #         self.targets_gpu = torch.stack(targets_gpu)
    #     return

    def shuffle_target(self, target):
        x = list(enumerate(target))
        random.shuffle(x)
        indices, target_shuffled = zip(*x)
        return target_shuffled

    # def get_loss_pytorch_shuffled_targets(self):
    #     device = 'cuda'
    #     x = list(enumerate(self.targets_gpu))
    #     random.shuffle(x)
    #     indices, target_shuffled = zip(*x)
    #     if self.mode == TrainingMode.mode2:
    #         self.target_shuffled = torch.reshape(torch.Tensor(target_shuffled), (-1, 1)).to(device)
    #     elif self.mode == TrainingMode.mode1:
    #         self.target_shuffled = torch.stack(target_shuffled)
    #
    #     return self.trainer.criterion(self.graph_emb_gpu.to(device), self.target_shuffled) / self.trainer.dataset.test_len


    def plot_embedding_targets(self, graph_emb, targets):
        emb_targ = list(zip(graph_emb, targets))

        if self.mode == TrainingMode.mode1:
            distinct_labels = np.unique(targets, axis=0)
            emb_targ_0 = [e[0] for e in emb_targ if (e[1] == distinct_labels[0]).all()]
            emb_targ_1 = [e[0] for e in emb_targ if (e[1] == distinct_labels[1]).all()]
        elif self.mode == TrainingMode.mode2:
            emb_targ_0 = [e[0] for e in emb_targ if e[1] == 0]
            emb_targ_1 = [e[0] for e in emb_targ if e[1] == 1]

        plt.hist(emb_targ_0, bins=30);
        plt.hist(emb_targ_1, bins=30);
        plt.show()