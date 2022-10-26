import random
import datetime
import os
from time import time
from tqdm import tqdm
from multiprocessing import Pool
from config_valid import Config

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter as WriterX
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import tensorflow as tf
from torchmetrics import Accuracy

from pytorchtools import EarlyStopping
from metrics import ExplainedVarianceMetric
from TorchPCA import PCA

from utils_tf import add_histogram
from config_valid import TrainingMode



class Trainer():

    def __init__(self, model, config_class, verbose=False):
        self.model = model
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.lr = self.conf['training']['learning_rate']
        self.epochs = self.conf['training']['epochs']
        self.batch_size = self.conf['training']['batch_size']

        self.last_layer_neurons = self.config_class.get_mode()['last_neuron']
        self.mode = self.conf['training']['mode']    # 'classification'  or 'regression'  or 'unsupervised'

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"

        self.set_optimizer(model)

        self.criterion = self.config_class.get_mode()['criterion']
        # if criterion == 'MSELoss':
        #     self.criterion = torch.nn.MSELoss()
        # elif criterion == 'CrossEntropy':
        #     self.criterion = torch.nn.CrossEntropyLoss()
        if verbose:
            print(self.criterion)

        self.dataset = None
        #self.myExplained_variance = ExplainedVarianceMetric(dimension=self.last_layer_neurons)
        self.last_accuracy = None

    def set_optimizer(self, model):
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr , )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=False)
        decayRate = 0.96
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

    def reinit_conf(self, config_class):
        self.config_class = config_class
        self.conf = self.config_class.conf

        self.lr = self.conf['training']['learning_rate']
        self.epochs = self.conf['training']['epochs']
        self.batch_size = self.conf['training']['batch_size']

        self.last_layer_neurons = self.config_class.get_mode()['last_neuron']
        self.mode = self.conf['training']['mode']  # 'classification'  or 'regression'  or 'unsupervised'

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"

    def reinit_model(self, new_model):
        self.model = new_model
        self.model.to(self.device)
        self.set_optimizer(self.model)

    def correct_shape(self, y):
        if self.last_layer_neurons == 1:
            target = y.unsqueeze(1)#.float()
        else:
            target = y
        return target

    def train(self):
        self.model.train()
        running_loss = 0

        for data in self.dataset.train_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = data.y
            target = self.correct_shape(data.y)
            #target = data.y.unsqueeze(1).float()  # TODO: modificato
            #print(f'target corrected {target}')
            #print(f'out: {out}')
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
            target = data.y
            target = self.correct_shape(data.y)
            #target = data.y.unsqueeze(1).float()  # TODO: modificato
            #print(f'target corrected {target}')
            #print(f'out: {out}')
            loss = self.criterion(out, target)  # Compute the loss.
            running_loss += loss.item()
            emb = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
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

    def take_embedding(self, loader, type_embedding='graph'):
        self.model.eval()
        graph_embeddings_array = []
        node_embeddings_array = []
        node_embeddings_array_id = []
        for data in loader:
            if type_embedding == 'graph':
                out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                graph_embeddings_array.extend(out)
            elif type_embedding == 'node':
                out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                node_embeddings_array.extend(out)
                node_embeddings_array_id.extend(data.id)
            elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
                node_out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                node_embeddings_array.extend(node_out)
                node_embeddings_array_id.extend(data.id)
                graph_out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                graph_embeddings_array.extend(graph_out)

        return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id


    def accuracy(self, loader):
        self.model.eval()
        accuracy_class = Accuracy()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)
            target = data.y
            #print(out, out.shape)
            #print(target, target.shape)
            #target = self.correct_shape(data.y)
            if not self.config_class.modo == TrainingMode.mode2:
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                label = target.argmax(dim=1)
                correct += int((pred == label).sum())
            else:
                #pred = out.argmax(dim=1)
                #label = target.argmax(dim=-1)
                #print(out.flatten())
                #print(target)
                correct += self.binary_accuracy(target, out.flatten())


            out2 = out.to(torch.device('cuda'))
            target2 = target.to(torch.device('cuda'), dtype=torch.int16)
            #print(self.model.device())
            #print(target.to(self.device, dtype=torch.int16).get_device())

            #correct += accuracy_class(out2.cpu(), target2.cpu())
            #correct += int((out == target).sum())  # Check against ground-truth labels.

        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def binary_accuracy(self, y_true, y_prob):
        assert y_true.ndim == 1, "dim not 1"
        assert y_true.size() == y_prob.size(), "size non equal"
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item()# / y_true.size(0)

    def load_dataset(self, dataset, percentage_train=0.7, parallel=False):  # dataset è di classe GeneralDataset
        self.dataset = Dataset.from_super_instance(percentage_train, self.batch_size, self.device, self.config_class, dataset)
        self.dataset.prepare(parallel)

    def launch_training(self, verbose=False):
        test_loss, var_exp, embeddings_array = self.test(self.dataset.test_loader)
        print(f'Before training Test loss: {test_loss}')

        if self.epochs == 0:
            return None, None

        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        neurons_str = self.config_class.layer_neuron_string()
        expstr = f"lr-{self.lr}_epochs{self.epochs}_bs{self.batch_size}_neurons-{neurons_str}"
        LOG_DIR = f"runs/{expstr}_{nowstr}"
        if verbose:
            print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)

        log_dir_variance = f"runs/ExpVar_{nowstr}"
        #writer_variance = SummaryWriter(log_dir_variance)
        # summary_variance = tf.summary.create_file_writer(log_dir_variance)

        # best_loss = 0 # for early stopping
        early_stopping = EarlyStopping(patience=self.conf['training']['earlystop_patience'],
                                       delta=0.00001,
                                       initial_delta=0.0004,
                                       minvalue=0.0002)
        # il seguente serve da quando è stata introdotta la batchnorm
        # early_stopping = EarlyStopping(patience=540, delta=0.01, initial_delta=0.04)
        # con task di regression
        #early_stopping = EarlyStopping(patience=self.conf['training']['earlystop_patience'],
        #                              delta=0.00000,
        #                              initial_delta=0.00004,
        #                              minvalue=0.00002)



        train_loss_list = []
        test_loss_list = []
        print(f"Run training for {self.epochs} epochs")
        with tf.compat.v1.Graph().as_default():
            summary_writer = tf.compat.v1.summary.FileWriter(log_dir_variance)
            epoch = 0
            for epoch in range(self.epochs):
                train_loss = self.train()
                test_loss, var_exp, embeddings_array = self.test(self.dataset.test_loader)
                test_acc = self.accuracy(self.dataset.test_loader)
                #expvar = self.myExplained_variance.compute()
                # add explained variance to tensorboard
                add_histogram(summary_writer, "Explained variance", var_exp, step=epoch)

                writer.add_scalar("Train Loss", train_loss, epoch)
                writer.add_scalar("Test accuracy", test_acc, epoch)
                writer.add_scalar("Test Loss", test_loss, epoch)
                # writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
                # writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
                # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                # train_acc_list.append(train_acc)
                # test_acc_list.append(test_acc)
                print_each_step = self.conf['logging']['train_step_print']
                if epoch % print_each_step == 0 and verbose:
                    print(f'Epoch: {epoch}\tTest loss: {test_loss}')  # \t Explained Variance: {var_exp}')
                    #print(f'Embeddings: {embeddings_array}')
                    #print('\n')

                # if test_loss > best_loss:  # check for early stopping
                #    best_loss = test_loss
                early_stopping(test_loss)
                if early_stopping.early_stop:
                    print("Early stopping!!!")
                    break
            print(f'Epoch: {epoch}\tTest loss: {test_loss} \t\t FINE TRAINING')

        writer.flush()
        #writer_variance.flush()
        #self.myExplained_variance.reset()
        self.last_accuracy = test_acc
        return train_loss_list, test_loss_list  # , train_acc_list, test_acc_list, last accuracy


class GeneralDataset:
    def __init__(self, dataset_list, labels, **kwarg):
        self.dataset_list = dataset_list
        self.labels = labels
        self.original_class = kwarg.get('original_class')

        #calcola il max degree
        #[dataset_list]

class Dataset(GeneralDataset):

    def __init__(self, percentage_train, batch_size, device, config_class, dataset_list, labels, original_class):
        super().__init__(dataset_list, labels, original_class=original_class)
        #self.dataset_list = dataset_list  # rename in dataset_list
        self.dataset_pyg = None
        #self.labels = labels
        self.len_data = len(self.dataset_list)
        self.tt_split = int(self.len_data * percentage_train)
        self.train_dataset = None
        self.train_len = None
        self.test_dataset = None
        self.test_len = None
        self.bs = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.config_class = config_class
        self.config = config_class.conf
        self.last_neurons = self.config_class.lastneuron
        self.transform4ae = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True,
                                              split_labels=True, add_negative_train_samples=False)

    @classmethod
    def from_super_instance(cls, percentage_train, batch_size, device, config_class, super_instance):
        return cls(percentage_train, batch_size, device, config_class, **super_instance.__dict__)

    def convert_G(self, g_i):
        g, i = g_i
        # aggiungo i metadati x e y per l'oggetto Data di PYG
        nodi = list(g.nodes)
        for n in nodi:
            g.nodes[n]["x"] = [1.0]
            g.nodes[n]["id"] = [n]

        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        #if len(type_graph) == 1:
        #    type_graph = [type_graph]
        #print(f'type_graph {type_graph}')
        # if self.config.modo == TrainingMode.mode1 or self.config.modo == TrainingMode.mode2:
        #     tipo = torch.long
        # if self.last_neurons == 1:  # TODO: cambiare anche qui
        #     tipo = torch.float
        # else:
        #     tipo = torch.long
        #
        # if self.config['graph_dataset']['regular']:

        tipo = torch.float
        # else:
        #     tipo = torch.float
        pyg_graph.y = torch.tensor(np.array([type_graph]), dtype=tipo)
        #print(pyg_graph.y)

        return pyg_graph

    def convert_G_random_feature(self, g_i):
        g, i = g_i
        nodi = list(g.nodes)
        for n in nodi:
            r = np.random.randn() + 1.0
            g.nodes[n]["x"] = [r]
        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        pyg_graph.y = torch.tensor([type_graph], dtype=torch.float)
        return pyg_graph

    def convert_G_autoencoder(self, g_i):
        pyg_graph = self.convert_G(g_i)
        pyg_graph, _, _ = self.transform4ae(pyg_graph)
        return pyg_graph

    # def process_each(self, each_list):
    #     each_pyg = []
    #     for i, g in enumerate(each_list):
    #         if self.config['model']['autoencoder']:
    #             pyg_graph = self.convert_G_autoencoder((g, i))
    #         elif self.config['graph_dataset']['random_node_feat']:
    #             print("randommmm featureeeeee!!!!!!!!!!!")
    #             pyg_graph = self.convert_G_random_feature((g, i))
    #         else:
    #             pyg_graph = self.convert_G((g, i))
    #         each_pyg.append(pyg_graph)
    #     return each_pyg

    def nx2pyg(self, graph_list_nx, parallel=False):
        dataset_pyg = []
        total = len(graph_list_nx)
        """
        with Pool(processes=12) as p:
            total = len(graph_list_nx)
            with tqdm(total=total) as pbar:
                for pyg_graph in p.imap_unordered(self.convert_G, zip(graph_list_nx, range(total)) ):
                    pbar.update()
                    dataset_pyg.append(pyg_graph)
        """

        if parallel:
            #process the test list elements in parallel
            with Pool(processes=2) as pool:
                dataset_pyg = pool.map(self.convert_G, zip(graph_list_nx, range(total)))

        else:
            i = 0
            for g in tqdm(graph_list_nx, total=len(graph_list_nx)):
                if self.config['model']['autoencoder']:
                    pyg_graph = self.convert_G_autoencoder((g, i))
                elif self.config['graph_dataset']['random_node_feat']:
                    pyg_graph = self.convert_G_random_feature((g, i))
                else:
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
        #starttime = time()
        for pyg_graph in dataset_pyg:
            pyg_graph = pyg_graph.to(self.device)
        #durata = time() - starttime
        #print(f"Tempo impiegato per spostare su GPU: {durata}")

        return dataset_pyg

    def prepare(self, parallel=False):
        starttime = time()
        self.dataset_pyg = self.nx2pyg(self.dataset_list, parallel)
        durata = time() - starttime
        print(f"Tempo impiegato: {durata}")

        # shuffle before train test split
        x = list(enumerate(self.dataset_pyg))
        random.shuffle(x)
        indices, self.dataset_pyg = zip(*x)
        self.labels = self.labels[list(indices)]
        # cambio l'ordine anche al dataset di grafi nx (non uso la numpy mask)
        self.dataset_list = [self.dataset_list[i] for i in list(indices)]
        # e cambio l'ordine anche alle orginal class nel caso regression con discrete distrib
        #if self.config['training']['mode'] == 'mode3' and not self.config['graph_dataset']['continuous_p']:
        self.original_class = [self.original_class[i] for i in list(indices)]

        self.train_dataset = self.dataset_pyg[:self.tt_split]
        self.test_dataset = self.dataset_pyg[self.tt_split:]
        self.train_len = len(self.train_dataset)
        self.test_len = len(self.test_dataset)

        #print(self.train_dataset[0].y, self.train_len)
        #print(self.test_dataset[0].y, self.test_len)

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

    def sample_dummy_data(self):
        whole_data = self.dataset_pyg
        all_data_loader = DataLoader(whole_data, batch_size=self.bs, shuffle=False)
        batch = next(iter(all_data_loader))
        return batch


