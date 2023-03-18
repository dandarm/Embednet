import datetime
import os
import copy
from time import time
from tqdm import tqdm

from config_valid import Config

import numpy as np
import torch
from torch.nn import Softmax
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter as WriterX

#import tensorflow as tf
from torchmetrics import Accuracy
from sklearn.metrics import f1_score

from pytorchtools import EarlyStopping
from metrics import ExplainedVarianceMetric
from TorchPCA import PCA

#from utils_tf import add_histogram
from config_valid import TrainingMode
from models import GCN, view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear
from Dataset import Dataset, GeneralDataset
from graph_generation import GenerateGraph
from plot_model import plot_model



class Trainer():

    def __init__(self, config_class, verbose=False):
        self.config_class = config_class
        self.config_class.valid_conf()
        self.conf = None
        self.model = None
        self.model_checkpoint = None
        self.best_model = None

        self.percentage_train = None
        self.lr = None
        self.epochs = None
        self.batch_size = None
        self.last_layer_neurons = None
        self.unique_train_name = None
        #self.mode = self.conf['training']['mode']    # 'classification'  or 'regression'  or 'unsupervised'
        self.epochs_checkpoint = None        
        self.shuffle_dataset = None
        self.save_best_model = None

        self.device = None

        self.criterion = None
        # if criterion == 'MSELoss':
        #     self.criterion = torch.nn.MSELoss()
        # elif criterion == 'CrossEntropy':
        #     self.criterion = torch.nn.CrossEntropyLoss()
        #if verbose:
        #    print(self.criterion)

        self.softmax = Softmax(dim=1)

        self.gg = None  # class generate_graph
        self.dataset = None
        #self.myExplained_variance = ExplainedVarianceMetric(dimension=self.last_layer_neurons)
        self.last_metric_value = None
        self.metric_list = None
        self.f1score_list = None
        #self.auc_score_list = None
        self.test_loss_list = None
        self.train_loss_list = None
        self.last_epoch = None

        self.graph_embedding_per_epoch = []
        self.node_embedding_per_epoch = []
        self.output_per_epoch = []
        self.model_LINEAR_pars_per_epoch = []
        self.model_GCONV_pars_per_epoch = []

        self.reinit_conf(config_class)


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
        model = GCN(self.config_class)
        model.to(device)
        if init_weights_gcn is not None:
            modify_parameters(model, init_weights_gcn)
        if init_weights_lin is not None:
            modify_parameters_linear(model, init_weights_lin)
        if verbose:
            print(model)
        return model

    def set_optimizer(self, model):
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr , )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          weight_decay=0,
                                          amsgrad=False)
        decayRate = 0.96
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

    def reinit_conf_file(self, config_file):
        config_c = Config(config_file)
        self.reinit_conf(config_c)
    def reinit_conf(self, config_class):
        self.config_class = config_class
        self.config_class.valid_conf()
        self.conf = self.config_class.conf

        self.percentage_train = self.conf['training']['percentage_train']
        self.lr = self.conf['training']['learning_rate']
        self.epochs = self.conf['training']['epochs']
        self.batch_size = self.conf['training']['batch_size']
        self.last_layer_neurons = self.config_class.get_mode()['last_neuron']
        #self.mode = self.conf['training']['mode']  # 'classification'  or 'regression'  or 'unsupervised'
        self.unique_train_name = self.config_class.unique_train_name

        self.epochs_checkpoint = self.conf['training'].get('epochs_model_checkpoint')
        self.shuffle_dataset = self.conf['training']['shuffle_dataset']
        self.save_best_model = self.conf['training']['save_best_model']

        self.criterion = self.config_class.get_mode()['criterion']

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"

    def load_model(self, model):
        self.model = model
        #self.model.to(self.device)
        self.set_optimizer(self.model)

    def init_dataset(self, parallel=True, verbose=False):
        """
        la classe GenerateGraph contiene un dataset in formato NetworkX
        :param parallel:
        :param verbose:
        :return:
        """
        self.gg = GenerateGraph(self.config_class)
        self.gg.initialize_dataset(parallel=parallel)  # istanzia il dataset networkx

        # alcuni controlli
        # print(self.gg.node_label)
        # print(self.gg.scalar_label)

    def load_dataset(self, dataset, parallel=False):  # dataset è di classe GeneralDataset
        print("Loading Dataset...")
        self.dataset = Dataset.from_super_instance(self.percentage_train, self.batch_size, self.device, self.config_class, dataset)
        self.dataset.prepare(self.shuffle_dataset, parallel)

    def init_all(self, parallel=True, verbose=False):
        """
        Inizializza modello e datasest
        :param parallel:
        :param verbose: se True ritorna il plot object del model
        :return:
        """
        init_weigths_method = self.config_class.init_weights_mode
        w = new_parameters(self.init_GCN(), init_weigths_method)
        model = self.init_GCN(init_weights_gcn=w, verbose=verbose)
        self.load_model(model)
        self.init_dataset(parallel=parallel, verbose=verbose)
        self.load_dataset(self.gg.dataset, parallel=False)  # parallel false perché con load_from_networkx non c'è nulla da fare...
        if verbose:
            batch = self.dataset.sample_dummy_data()
            plot = plot_model(self.model, batch)
            return plot




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
            del loss
            del out
        return running_loss / self.dataset.train_len

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        for data in loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = data.y
            target = self.correct_shape(data.y)
            #target = data.y.unsqueeze(1).float()  # TODO: modificato
            #print(f'target shape {target.shape}')
            #print(f'out: shape {out.shape}')
            loss = self.criterion(out, target)  # Compute the loss.
            running_loss += loss.item()
            #print(f"out and target shape")
            #print(emb.shape, target.shape)
            del loss
            del out

        #graph_embeddings_array, node_embeddings_array, _, final_output = self.get_embedding(loader)

        #calcola la PCA
        #obj = PCA(graph_embeddings_array)
        #var_exp, _, _ = obj.get_ex_var()
        #var_exp = torch.as_tensor(np.array(var_exp))
        #var_exp = self.myExplained_variance(emb, target)  # sul singolo batch

        return running_loss / self.dataset.test_len#, var_exp#, graph_embeddings_array, node_embeddings_array

    def take_embedding_all_data(self, type_embedding='both'):
        all_data_loader = self.dataset.get_all_data_loader()
        return self.get_embedding(all_data_loader, type_embedding=type_embedding)
    def take_embedding_test(self, type_embedding='both'):
        return self.get_embedding(self.dataset.test_loader, type_embedding=type_embedding)
    def take_embedding_train(self, type_embedding='both'):
        return self.get_embedding(self.dataset.train_loader, type_embedding=type_embedding)

    def get_embedding(self, loader, type_embedding='both'):
        self.model.eval()
        graph_embeddings_array = []
        node_embeddings_array = []
        node_embeddings_array_id = []
        final_output = []
        with torch.no_grad():
            for data in loader:
                if type_embedding == 'graph':
                    out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                    to = out.detach().cpu().numpy()
                    graph_embeddings_array.extend(to)
                elif type_embedding == 'node':
                    out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                    to = out.detach().cpu().numpy()
                    node_embeddings_array.extend(to)
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
                    node_out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                    to = node_out.detach().cpu().numpy()
                    #print(f"node emb size: {to.nbytes}")
                    node_embeddings_array.extend(to)
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                    graph_out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                    to = graph_out.detach().cpu().numpy()
                    #print(f"graph emb size: {to.nbytes}")
                    graph_embeddings_array.extend(to)

                    out = self.model(data.x, data.edge_index, data.batch)
                    to = out.detach().cpu().numpy()
                    final_output.extend(to)

        graph_embeddings_array = np.array(graph_embeddings_array)
        node_embeddings_array = np.array(node_embeddings_array)
        return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output

    def take_embedding_gpuversion(self, loader, type_embedding='both'):
        print("take_embedding_gpuversion")
        self.model.eval()
        node_embeddings_array_id = []
        graph_embeddings_array = torch.empty((1,1), device=torch.device('cuda'))
        node_embeddings_array = torch.empty((1,1), device=torch.device('cuda'))
        final_output = torch.empty((1, 1), device=torch.device('cuda'))
        with torch.no_grad():
            for data in loader:
                if type_embedding == 'graph':
                    out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                    graph_embeddings_array = torch.cat((graph_embeddings_array, out))
                elif type_embedding == 'node':
                    out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                    node_embeddings_array = torch.cat((node_embeddings_array, out))
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
                    node_out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                    #print(f"node_out shape : {node_out.shape}")
                    node_embeddings_array = torch.cat((node_embeddings_array, node_out))
                    #node_embeddings_array_id.extend(data.id) TODO: rimettere
                    graph_out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
                    #print(f"graph_out shape : {graph_out.shape}")
                    graph_embeddings_array = torch.cat((graph_embeddings_array, graph_out))

                    out = self.model(data.x, data.edge_index, data.batch)
                    final_output = torch.cat((final_output, out))

        return graph_embeddings_array[1:], node_embeddings_array[1:], node_embeddings_array_id, final_output[1:]


    def calc_metric(self, loader):
        self.model.eval()
        # accuracy_class = Accuracy()
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
                correct += self.binary_accuracy(target, out.flatten())

            #f1 = self.calc_f1score(target.detach().cpu(), out.detach().cpu())  TODO:  riprovare a chiamare prima detach e poi cpu


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

    def calc_f1score(self, y_true, y_pred):
        if self.config_class.modo == TrainingMode.mode2:
            pred = y_pred > 0.5
            label = y_true
        else:
            y_pred = self.softmax(y_pred)
            pred = torch.argmax(y_pred, dim=1)  # Use the class with highest probability.
            label = torch.argmax(y_true, dim=1)
        #print(y_true, y_pred)
        f1 = f1_score(label, pred, average=None)
        #print(f"f1:{f1}")
        return f1


    def launch_training(self, verbose=0):
        self.train_loss_list = []
        self.test_loss_list = []
        self.metric_list = []
        self.f1score_list = []
        self.graph_embedding_per_epoch = []
        self.node_embedding_per_epoch = []
        self.output_per_epoch = []
        self.model_LINEAR_pars_per_epoch = []
        self.model_GCONV_pars_per_epoch = []

        test_loss = self.test(self.dataset.test_loader)
        if self.dataset.all_data_loader is None:
            all_data_loader = DataLoader(self.dataset.dataset_pyg, batch_size=self.dataset.bs, shuffle=False)
        else:
            all_data_loader = self.dataset.all_data_loader
        alldata_loss = self.test(all_data_loader)
        all_graph_embeddings_array, all_node_embeddings_array, _, _ = self.get_embedding(all_data_loader)

        # Calcola la metrica (accuracy o auc etc...)
        if self.config_class.modo != TrainingMode.mode3:  # and not self.config_class.conf['model']['autoencoder']:
            metric_value = self.calc_metric(self.dataset.test_loader)

        if verbose > 1:
            print(f'Before training Test loss: {test_loss}')
            print(f"test accuracy iniziale: {metric_value}")
            print(f'Before training Training + Test loss: {alldata_loss}')


        self.test_loss_list.append(test_loss)
        self.metric_list.append(metric_value)
        #self.f1score_list.append(test_f1score)
        self.last_metric_value = metric_value
        self.last_epoch = 0
        self.graph_embedding_per_epoch.append(all_graph_embeddings_array)
        self.node_embedding_per_epoch.append(all_node_embeddings_array)

        if self.epochs == 0:
            return

        # SETUP TENSORBOARD folder
        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        LOG_DIR = f"runs/{self.unique_train_name}__{nowstr}"
        if verbose > 0: print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)
        writerX = WriterX(LOG_DIR)

        log_dir_variance = f"runs/ExpVar_{nowstr}"
        #writer_variance = SummaryWriter(log_dir_variance)
        # summary_variance = tf.summary.create_file_writer(log_dir_variance)

        # EARLY STOPPING SETUP
        best_loss = 100  # for model saving
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


        if verbose > 0: print(f"Run training for {self.epochs} epochs")
        #with tf.compat.v1.Graph().as_default():
        #summary_writer = tf.compat.v1.summary.FileWriter(log_dir_variance) TODO: CALCOLO DELLA pca TEMPORANEAMENTE SOSPESO
        epoch = 0
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            train_loss = self.train()
            test_loss = self.test(self.dataset.test_loader)
            writer.add_scalar("Train Loss", train_loss, epoch)
            writer.add_scalar("Test Loss", test_loss, epoch)

            if self.config_class.modo != TrainingMode.mode3:  # and not self.config_class.conf['model']['autoencoder']:
                metric_value = self.calc_metric(self.dataset.test_loader)
                writer.add_scalar("Test metric", metric_value, epoch)

            # prendo l'embedding a ogni epoca
            if self.conf['training'].get('every_epoch_embedding'):
                #######  _____________take embedding  (switch tra gpuversion e normale)
                graph_embeddings_array, node_embeddings_array, _, final_output = self.get_embedding(all_data_loader)
                self.graph_embedding_per_epoch.append(graph_embeddings_array)
                self.node_embedding_per_epoch.append(node_embeddings_array)
                # prendo anche l'output nel caso in cui abbiamo il layer denso finale
                self.output_per_epoch.append(final_output)
                # prendo pure i weights del modello
                if self.conf['model']['last_layer_dense']:
                    self.model_LINEAR_pars_per_epoch.append(get_parameters(self.model.linears))
                self.model_GCONV_pars_per_epoch.append(get_parameters(self.model.convs))

            #expvar = self.myExplained_variance.compute()
            # add explained variance to tensorboard
            #add_histogram(summary_writer, "Explained variance", var_exp, step=epoch)  TODO: CALCOLO DELLA pca TEMPORANEAMENTE SOSPESO


            #for i, v in enumerate(test_f1score):
            #    writer.add_scalar(f"Test F1 Score/{i}", v, epoch)
            #writerX.add_scalars("Test F1 Score", {f"Class_{i}": v for i, v in enumerate(test_f1score)}, epoch)   TODO: rimettere!
            # writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
            # writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            self.train_loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)
            self.metric_list.append(metric_value)
            #self.f1score_list.append(test_f1score)

            # train_acc_list.append(train_acc)
            # test_acc_list.append(test_acc)

            print_each_step = self.conf['logging']['train_step_print']
            if epoch % print_each_step == 0:
                if verbose > 1: print(f'Epoch: {epoch}\tTest loss: {test_loss}')  # \t Explained Variance: {var_exp}')

            # save model
            if epoch in self.epochs_checkpoint:
                self.model_checkpoint = copy.deepcopy(self.model)
            if test_loss < best_loss:  # check for save best model
                best_loss = test_loss
                if verbose:
                    print(best_loss)
                if self.save_best_model:
                    self.best_model = copy.deepcopy(self.model)

            early_stopping(test_loss)
            if early_stopping.early_stop:
                #if verbose > 0:
                print("Early stopping!!!")
                break

        if verbose > 0: print(f'Epoch: {epoch}\tTest loss: {test_loss} \t\tBest test loss: {best_loss} FINE TRAINING')

        print(f"test accuracy finale: {metric_value}")
        writer.flush()
        #writer_variance.flush()
        #self.myExplained_variance.reset()
        self.last_metric_value = self.metric_list[-1]
        self.last_epoch = epoch

        # riporto su cpu gli embedding per epoca
        #print("Take to cpu the embedding array ")
        #self.graph_embedding_per_epoch = [a.squeeze().cpu().detach().numpy() for a in self.graph_embedding_per_epoch]
        #self.node_embedding_per_epoch = [a.squeeze().cpu().detach().numpy() for a in self.node_embedding_per_epoch]
        #self.output_per_epoch = [a.squeeze().cpu().detach().numpy() for a in self.output_per_epoch]
        return


