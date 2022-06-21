import random
import datetime
import os
from time import time
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter as WriterX
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import tensorflow as tf

from pytorchtools import EarlyStopping
from metrics import ExplainedVarianceMetric
from TorchPCA import PCA
from utils import add_histogram




class Trainer():

    def __init__(self, model,
                 config):  # learning_rate, epochs, batch_size, layers, neurons, last_layer_neurons, criterion, device=None):
        self.model = model
        self.config = config

        self.lr = config['training']['learning_rate']
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        self.layers = len(config['model']['neurons_per_layer'])
        self.neurons = config['model']['neurons_per_layer']
        self.last_layer_neurons = config['model']['neurons_per_layer'][-1]

        if config['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"

        # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr , )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                          amsgrad=False)
        decayRate = 0.96
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        criterion = config['training']['criterion']
        if criterion == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()

        print(self.criterion)

        self.dataset = None
        self.myExplained_variance = ExplainedVarianceMetric(dimension=self.last_layer_neurons)

    def set_model(self, new_model):
        self.model = new_model

    def correct_shape(self, y):
        # if isinstance(self.model, GCN1n):
        if self.last_layer_neurons == 1:
            target = y.unsqueeze(1).float()
        else:
            target = y
        return target

    def train(self):
        self.model.train()
        running_loss = 0

        for data in self.dataset.train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = self.correct_shape(data.y)
            loss = self.criterion(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            # self.scheduler.step()
            running_loss += loss.item()
        return running_loss / self.dataset.train_len

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        embeddings_array = []

        for data in loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = self.correct_shape(data.y)
            loss = self.criterion(out, target)  # Compute the loss.
            running_loss += loss.item()
            emb = self.model(data.x, data.edge_index, data.batch, embedding=True)
            embeddings_array.extend(emb.cpu().detach().numpy())
            #print(f"out and target shape")
            #print(emb.shape, target.shape)

        #calcola la PCA
        embeddings_array = np.array(embeddings_array)
        obj = PCA(embeddings_array)
        var_exp, _, _ = obj.get_ex_var()
        #var_exp = torch.as_tensor(np.array(var_exp))
        #var_exp = self.myExplained_variance(emb, target)  # sul singolo batch
        return running_loss / self.dataset.test_len, var_exp, embeddings_array

    def take_embedding(self, loader):
        self.model.eval()
        embeddings_array = []
        for data in loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch, embedding=True)
            embeddings_array.extend(out)
        return embeddings_array

    def accuracy(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)
            target = self.correct_shape(data.y)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == target).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def load_dataset(self, dataset_list, labels, percentage_train=0.7):
        self.dataset = Dataset(dataset_list, labels, percentage_train, self.batch_size, self.device, self.config)
        self.dataset.prepare()

    def launch_training(self):
        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        neurons_str = str(self.neurons).replace(', ', '-').strip('[').strip(']')
        expstr = f"lr-{self.lr}_epochs{self.epochs}_bs{self.batch_size}_neurons-{neurons_str}"
        LOG_DIR = f"runs/{expstr}_{nowstr}"
        print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)

        log_dir_variance = f"runs/ExpVar_{nowstr}"
        #writer_variance = SummaryWriter(log_dir_variance)
        # summary_variance = tf.summary.create_file_writer(log_dir_variance)

        # best_loss = 0 # for early stopping
        # early_stopping = EarlyStopping(patience=200, delta=0.02, initial_delta=0.2)
        # il seguente serve da quando è stata introdotta la batchnorm
        # early_stopping = EarlyStopping(patience=540, delta=0.01, initial_delta=0.04)
        # con task di regression
        early_stopping = EarlyStopping(patience=100, delta=0.00000, initial_delta=0.00004, minvalue=0.00002)

        train_loss_list = []
        test_loss_list = []
        print(f"Run training for {self.epochs} epochs")
        with tf.compat.v1.Graph().as_default():
            summary_writer = tf.compat.v1.summary.FileWriter(log_dir_variance)
            for epoch in range(self.epochs):
                train_loss = self.train()
                test_loss, var_exp, embeddings_array = self.test(self.dataset.test_loader)
                #expvar = self.myExplained_variance.compute()
                # add explained variance to tensorboard
                add_histogram(summary_writer, "Explained variance", var_exp, step=epoch)

                writer.add_scalar("Train Loss", train_loss, epoch)
                # writer.add_scalar("Train accuracy", train_acc, epoch)
                writer.add_scalar("Test Loss", test_loss, epoch)
                # writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
                # writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
                # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                # train_acc_list.append(train_acc)
                # test_acc_list.append(test_acc)
                if epoch % 100 == 0:
                    print(f'Epoch: {epoch}\tTest loss: {test_loss}')  # \t Explained Variance: {var_exp}')
                    #print(f'Embeddings: {embeddings_array}')
                    #print('\n')

                # if test_loss > best_loss:  # check for early stopping
                #    best_loss = test_loss
                early_stopping(test_loss)
                if early_stopping.early_stop:
                    print("Early stopping!!!")
                    break

        writer.flush()
        #writer_variance.flush()
        self.myExplained_variance.reset()
        return train_loss_list, test_loss_list  # , train_acc_list, test_acc_list


class Dataset():

    def __init__(self, dataset_list, labels, percentage_train, batch_size, device, config):
        self.dataset = dataset_list  # rename in dataset_list
        self.dataset_pyg = None
        self.labels = labels
        self.len_data = len(self.dataset)
        self.tt_split = int(self.len_data * percentage_train)
        self.train_dataset = None
        self.train_len = None
        self.test_dataset = None
        self.test_len = None
        self.bs = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.config = config
        self.last_neurons = config['model']['neurons_per_layer'][-1]
        self.transform4ae = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True,
                                              split_labels=True, add_negative_train_samples=False)

    def convert_G(self, g_i):
        g, i = g_i
        # aggiungo i metadati x e y per l'oggetto Data di PYG
        nodi = list(g.nodes)
        for n in nodi:
            g.nodes[n]["x"] = [1.0]

        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        # print(f'type_graph {type_graph}')
        if self.last_neurons == 1:
            tipo = torch.float
        else:
            tipo = torch.long
        pyg_graph.y = torch.tensor([type_graph], dtype=tipo)

        return pyg_graph

    def convert_G_autoencoder(self, g_i):
        pyg_graph = self.convert_G(g_i)
        pyg_graph, _, _ = self.transform4ae(pyg_graph)
        return pyg_graph

    def process_each(self, each_list):
        each_pyg = []
        for i, g in enumerate(each_list):
            if not self.config['model']['autoencoder']:
                pyg_graph = self.convert_G((g, i))
            else:
                pyg_graph = self.convert_G_autoencoder((g, i))
            each_pyg.append(pyg_graph)
        return each_pyg

    def nx2pyg(self, graph_list_nx):
        dataset_pyg = []
        total = len(graph_list_nx)
        """
        with Pool(processes=12) as p:
            total = len(graph_list_nx)
            with tqdm(total=total) as pbar:
                for pyg_graph in p.imap_unordered(self.convert_G, zip(graph_list_nx, range(total)) ):
                    pbar.update()
                    dataset_pyg.append(pyg_graph)

        # process the test list elements in parallel
        pool = Pool(processes=32)
        dataset_pyg = pool.map(self.convert_G, zip(graph_list_nx, range(total)))

        """
        i = 0
        for g in tqdm(graph_list_nx, total=len(graph_list_nx)):
            pyg_graph = self.convert_G((g, i))
            dataset_pyg.append(pyg_graph)
            i += 1

        """
        from joblib import Parallel, delayed
        def process(i):
            return i * i

        results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
        print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        """

        for pyg_graph in dataset_pyg:
            pyg_graph = pyg_graph.to(self.device)

        return dataset_pyg

    def prepare(self):
        starttime = time()
        self.dataset_pyg = self.nx2pyg(self.dataset)
        durata = time() - starttime
        print(f"Tempo impiegato: {durata}")

        # shuffle before train test split
        x = list(enumerate(self.dataset_pyg))
        random.shuffle(x)
        indices, self.dataset_pyg = zip(*x)
        self.labels = self.labels[list(indices)]
        # cambio l'ordine anche al dataset di grafi nx (non uso la numpy mask)
        self.dataset = [self.dataset[i] for i in list(indices)]

        self.train_dataset = self.dataset_pyg[:self.tt_split]
        self.test_dataset = self.dataset_pyg[self.tt_split:]
        self.train_len = len(self.train_dataset)
        self.test_len = len(self.test_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False)

        """
        for step, data in enumerate(self.train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()
        """