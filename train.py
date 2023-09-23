import datetime
import os
import sys
import traceback
import copy
from pathlib import Path
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import multiprocessing

import numpy as np
import torch
from torch.nn import Softmax
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter as WriterX
from pytorchtools import EarlyStopping
#import tensorflow as tf
from torchmetrics import Accuracy
from sklearn.metrics import f1_score

from metrics import ExplainedVarianceMetric
from TorchPCA import PCA

#from utils_tf import add_histogram
from config_valid import Config, TrainingMode, GraphType
from models import GCN, view_parameters, get_parameters, new_parameters, modify_parameters, Inits, modify_parameters_linear, get_param_labels
from Dataset import Dataset, GeneralDataset
from Dataset_autoencoder import DatasetReady
from graph_generation import GenerateGraph
from take_real_data import TakeRealDataset
from plot_model import plot_model
from Metrics import Metrics
from plot_funcs import Data2Plot, DataAutoenc2Plot, plot_metrics, save_ffmpeg
from embedding import Embedding, Embedding_autoencoder



class Trainer():

    def __init__(self, config_class, verbose=False, rootsave="."):
        self.config_class = config_class
        self.config_class.valid_conf()
        self.conf = None
        self.model = None
        self.model_checkpoint = None
        self.best_model = None

        self.rootsave = rootsave
        self.verbose = verbose

        #self.percentage_train = None
        self.lr = None
        self.epochs = None
        #self.batch_size = None
        self.last_layer_neurons = None
        self.embedding_dimension = None
        self.unique_train_name = None
        #self.mode = self.conf['training']['mode']    # 'classification'  or 'regression'  or 'unsupervised'
        self.epochs_checkpoint = None        
        self.shuffle_dataset = None
        self.save_best_model = None

        self.device = None

        self.name_of_metric = "accuracy"
        self.softmax = Softmax(dim=1)

        self.gg = None  # class generate_graph
        self.dataset = None
        #self.myExplained_variance = ExplainedVarianceMetric(dimension=self.last_layer_neurons)
        self.last_metric_value = None
        #self.metric_list = None
        self.metric_obj_list = None
        self.metric_obj_list_train = []
        self.metric_obj_list_test = []
        self.test_loss_list = []
        self.train_loss_list = []
        self.last_epoch = None

        self.init_conf(config_class)


        # CONFIGURAZIONI BASE PER OGNI EVENTUALE TRIAL
        self.run_path = self.create_runpath_dir()
        # scrivo la lista delle epoche sulla quale esegui i calcoli
        self.epochs_list_points = self.conf["training"].get("epochs_list_points")
        self.epochs_list = self.make_epochs_list_for_embedding_tracing(self.epochs_list_points)
        #self.mapping_of_snap_epochs = {snapshot_epochs: sequential_epochs for
        #                sequential_epochs, snapshot_epochs in enumerate(self.epochs_list)}

        self.total_node_emb_dim = multiprocessing.Array('f', range(self.epochs+1))
        self.total_graph_emb_dim = multiprocessing.Array('f', range(self.epochs+1))
        for i in range(self.epochs):
            self.total_node_emb_dim[i] = -1.0
            self.total_graph_emb_dim[i] = -1.0

    def create_runpath_dir(self):
        run_path = self.rootsave / self.unique_train_name
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        return run_path

    def make_epochs_list_for_embedding_tracing(self, list_points):
        """
        Restituisce una lista di valori interi che non può avere num elementi = a list_points
        perché la sequenza logaritmica che va arrotondata all'intero produce dei duplicati
        :param list_points: numero di punti per la sequenza logaritmica
        :return: Una lista di interi che si riferiscono alle epoche alle quali salvare gli embedding.
        Può avere una lunghezza inferiore a list_points
        """
        if self.epochs == 0:
            return [0]
        endlog = np.log2(self.epochs)
        end3 = np.cbrt(self.epochs)
        end2 = np.sqrt(self.epochs)
        if list_points > self.epochs:
            raise Exception('Chiesto un numero di epochs_list maggiore del numero di epoche da config')
        #list_points = min(list_points, self.trainer.epochs)
        #logarray = np.round(np.logspace(1., endlog, num=list_points, base=2)).astype(int)
        cubicarray = np.power(np.linspace(1., end3, num=list_points), 3)
        squarearray = np.power(np.linspace(1., end2, num=list_points), 2)
        #lista = np.unique(np.concatenate((np.arange(0, 10, 2), logarray)))[:-1]
        #lista = np.unique(np.round(cubicarray)).astype(int)[:-1]
        lista = np.unique(np.round(cubicarray)).astype(int)#[:-1]
        if self.verbose:
            print(f"epoch list {lista} ")
        return np.concatenate(([0], lista))  # aggiungo anche l'elemento 0 per i plot snapshot

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
        if self.conf['training'].get('optimizer') == "SGD":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=float(self.lr), momentum=0.9)
        elif self.conf['training'].get('optimizer') == "ADAM":
            # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr , )
            self.optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr),
                                              betas=(0.9, 0.999),
                                              eps=1e-08,
                                              weight_decay=0,
                                              amsgrad=False)
        elif self.conf['training'].get('optimizer') == 'ADAMW':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.lr))  # gli altri valori default sono gli stessi

        decayRate = 0.96
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

    def reinit_conf_file(self, config_file):
        config_c = Config(config_file)
        self.init_conf(config_c)
    def init_conf(self, config_class):
        self.config_class = config_class
        self.config_class.valid_conf()
        self.conf = self.config_class.conf

        #self.percentage_train = self.conf['training']['percentage_train']
        self.lr = self.conf['training']['learning_rate']
        self.epochs = self.conf['training']['epochs']
        #self.batch_size = self.conf['training']['batch_size']
        self.last_layer_neurons = self.config_class.get_mode()['last_neuron']
        #self.mode = self.conf['training']['mode']  # 'classification'  or 'regression'  or 'unsupervised'
        self.unique_train_name = self.config_class.unique_train_name

        self.epochs_checkpoint = self.conf['training'].get('epochs_model_checkpoint')
        self.shuffle_dataset = self.conf['training']['shuffle_dataset']
        self.save_best_model = self.conf['training']['save_best_model']

        # CONFIGURA LA LOSS
        self.criterion = self.config_class.get_loss()
        if getattr(self.criterion, "reduction") == 'none':
            self.is_weighted = True  # self.config_class.conf['training'].get('weigths4unbalanced_dataset')
        else:
            self.is_weighted = False
        #if self.verbose:
        print(self.criterion)
        print(f"loss reduction: {getattr(self.criterion, 'reduction')}, -> is_weighted: {self.is_weighted}")

        if self.conf['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"

    def load_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.set_optimizer(self.model)
        if hasattr(self.model, 'convs'):
            self.embedding_dimension = self.model.convs[-1].out_channels
        else:
            self.embedding_dimension = self.model.embedding_dim
        if self.verbose: print(f"Optimizer: {self.optimizer}")

    def init_dataset(self, parallel=True, verbose=False):
        """
        la classe GenerateGraph contiene un dataset in formato NetworkX
        Se è un dataset reale viene creato un diverso gestore del dataset
        :param parallel:
        :param verbose:
        :return:
        """
        if not self.config_class.conf['graph_dataset'].get('real_dataset'):
            self.gg = GenerateGraph(self.config_class, self.verbose)
            self.gg.initialize_dataset(parallel=parallel)  # instanzia il dataset networkx
        else:
            # TakeRealDataset si occupa di ottenere il dataset reale
            real_dataset_taker = TakeRealDataset(self.config_class, self.verbose)
            real_dataset_taker.get_dataset()  # imposta gg.dataset
            self.gg = real_dataset_taker.gg

    def load_dataset(self, dataset, parallel=False):  # dataset è di classe GeneralDataset
        print("Loading Dataset...")
        if not self.config_class.conf['graph_dataset'].get('real_dataset'):            
            self.dataset = Dataset.from_super_instance(self.config_class, dataset, verbose=self.verbose)
            self.dataset.prepare(self.shuffle_dataset, parallel)
        else:
            if self.config_class.conf['graph_dataset']['real_data_name'] == 'REDDIT-BINARY':
                # con questo dataset abbiamo già una lista di oggetti Data
                self.dataset = DatasetReady(self.config_class, dataset)
            elif self.config_class.conf['graph_dataset']['real_data_name'] == 'BACI':
                self.dataset = Dataset.from_super_instance(self.config_class, dataset)
                self.dataset.prepare(self.shuffle_dataset, parallel)

    def init_all(self, parallel=True, verbose=False, path_model_toload=None):
        """
        Inizializza modello e datasest
        :param parallel:
        :param verbose: se True ritorna il plot object del model
        :return:
        """
        init_weigths_method = self.config_class.init_weights_mode
        w = new_parameters(self.init_GCN(), init_weigths_method)
        model = self.init_GCN(init_weights_gcn=w, verbose=verbose)
        if path_model_toload is None:
            self.load_model(model)
        else:
            model.load_state_dict(torch.load(path_model_toload))
            model.eval()
            self.load_model(model)
        
        self.init_dataset(parallel=parallel, verbose=verbose)
        self.load_dataset(self.gg.dataset, parallel=False)  # parallel false perché con load_from_networkx non c'è nulla da fare...

        if self.conf.get("plot").get("plot_model"):
            batch = self.dataset.sample_dummy_data()
            plot = plot_model(self.model, batch, self.unique_train_name)
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
        num_batches = 0
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
            num_batches += 1
            del loss
            del out
        return running_loss / num_batches

    def test(self, loader):
        self.model.eval()
        running_loss = 0
        num_batches = 0
        for data in loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = data.y
            target = self.correct_shape(data.y)
            #target = data.y.unsqueeze(1).float()  # TODO: modificato
            #print(f'target shape {target.shape}')
            #print(f'out: shape {out.shape}')
            loss = self.criterion(out, target)  # Compute the loss.
            running_loss += loss.item()
            num_batches += 1
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

        return running_loss / num_batches

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
                    #node_embeddings_array_id.extend(data.id)
                elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
                    node_out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
                    to = node_out.detach().cpu().numpy()
                    #print(f"node emb size: {to.nbytes}")
                    node_embeddings_array.extend(to)
                    #node_embeddings_array_id.extend(data.id)
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

    # def take_embedding_gpuversion(self, loader, type_embedding='both'):
    #     print("take_embedding_gpuversion")
    #     self.model.eval()
    #     node_embeddings_array_id = []
    #     graph_embeddings_array = torch.empty((1,1), device=torch.device('cuda'))
    #     node_embeddings_array = torch.empty((1,1), device=torch.device('cuda'))
    #     final_output = torch.empty((1, 1), device=torch.device('cuda'))
    #     with torch.no_grad():
    #         for data in loader:
    #             if type_embedding == 'graph':
    #                 out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
    #                 graph_embeddings_array = torch.cat((graph_embeddings_array, out))
    #             elif type_embedding == 'node':
    #                 out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
    #                 node_embeddings_array = torch.cat((node_embeddings_array, out))
    #                 #node_embeddings_array_id.extend(data.id) TODO: rimettere
    #             elif type_embedding == 'both':  # quì ho garantito che i graph embedding sono ordinati come i node_embedding
    #                 node_out = self.model(data.x, data.edge_index, data.batch, node_embedding=True)
    #                 #print(f"node_out shape : {node_out.shape}")
    #                 node_embeddings_array = torch.cat((node_embeddings_array, node_out))
    #                 #node_embeddings_array_id.extend(data.id) TODO: rimettere
    #                 graph_out = self.model(data.x, data.edge_index, data.batch, graph_embedding=True)
    #                 #print(f"graph_out shape : {graph_out.shape}")
    #                 graph_embeddings_array = torch.cat((graph_embeddings_array, graph_out))
    #
    #                 out = self.model(data.x, data.edge_index, data.batch)
    #                 final_output = torch.cat((final_output, out))
    #
    #     return graph_embeddings_array[1:], node_embeddings_array[1:], node_embeddings_array_id, final_output[1:]


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

            #f1 = self.calc_f1score(target.detach().cpu(), out.detach().cpu())


            out2 = out.to(torch.device(self.device))
            target2 = target.to(torch.device(self.device), dtype=torch.int16)
            #print(self.model.device())
            #print(target.to(self.device, dtype=torch.int16).get_device())

            #correct += accuracy_class(out2.cpu(), target2.cpu())
            #correct += int((out == target).sum())  # Check against ground-truth labels.

        metriche = Metrics(accuracy=correct / len(loader.dataset))
        return metriche

    def binary_accuracy(self, y_true, y_prob, threshold=0.5):
        assert y_true.ndim == 1, "dim not 1"
        assert y_true.size() == y_prob.size(), "size non equal"
        y_prob = y_prob > threshold
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
        #self.metric_obj_list = []
        self.metric_obj_list_train = []
        self.metric_obj_list_test = []
        animation_files = []

        print("Epoca 0...")
        test_loss = self.test(self.dataset.test_loader)
        self.train_loss_list.append(test_loss)
        self.test_loss_list.append(test_loss)
        print("Prima snapshot...")   # self.epochs if self.epochs > 0 else 1,
        self.produce_traning_snapshot(0, False, [])
        file_path = self.run_path / f"_epoch0.png"
        animation_files.append(file_path)

        #self.f1score_list.append(test_f1score)
        #self.last_metric_value = self.metric_list[-1]
        self.last_epoch = 0
        # siamo bravi fin quì 🤪

        if self.epochs == 0:
            return

        # region SETUP TENSORBOARD folder
        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        LOG_DIR = f"runs/{self.unique_train_name}__{nowstr}"
        if verbose > 0: print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)
        writerX = WriterX(LOG_DIR)

        log_dir_variance = f"runs/ExpVar_{nowstr}"
        #writer_variance = SummaryWriter(log_dir_variance)
        # summary_variance = tf.summary.create_file_writer(log_dir_variance)
        # endregion


        # EARLY STOPPING SETUP
        best_loss = 100  # for model saving
        best_epoch = 0
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


        parallel_processes_save_images = []

        for epoch in tqdm(range(1, self.epochs+1), total=self.epochs):

            if epoch % 1 == 0:
                test_loss = self.test(self.dataset.test_loader)
                writer.add_scalar("Test Loss", test_loss, epoch)
                #writer.add_scalar(f"Test {self.name_of_metric}", metric_object.get_metric(self.name_of_metric), epoch)
            # VOGLIO AGGIORNARE I PESI COL TRAINING DOPO AVER CALCOLATO LA LOSS DI TEST,
            # ALTRIMENTI LA LOSS DI TEST SARÀ SEMPRE PIÙ AVVANTAGGIATA RISPETTO ALLA LOSS DI TRAINING
            # PERCHÈ ARRIVEREBBE DOPO L'AGGIORNAMENTO APPUNTO
            train_loss = self.train()
            writer.add_scalar("Train Loss", train_loss, epoch)


            #for i, v in enumerate(test_f1score):
            #    writer.add_scalar(f"Test F1 Score/{i}", v, epoch)
            #writerX.add_scalars("Test F1 Score", {f"Class_{i}": v for i, v in enumerate(test_f1score)}, epoch)   TODO: rimettere!
            # writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
            # writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            self.train_loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)

            if self.conf['training'].get('every_epoch_embedding'):
                if epoch in self.epochs_list:
                    parallel = True
                    self.produce_traning_snapshot(epoch, parallel, parallel_processes_save_images)
                    file_path = self.run_path / f"_epoch{epoch}.png"
                    animation_files.append(file_path)


            #expvar = self.myExplained_variance.compute()
            # add explained variance to tensorboard
            #add_histogram(summary_writer, "Explained variance", var_exp, step=epoch)  TODO: CALCOLO DELLA pca TEMPORANEAMENTE SOSPESO


            # print_each_step = self.conf['logging']['train_step_print']
            # if epoch % print_each_step == 0:
            #     if verbose > 1: print(f'Epoch: {epoch}\tTest loss: {test_loss}')  # \t Explained Variance: {var_exp}')

            # region salva il self.best_model
            #if epoch in self.epochs_checkpoint:
            #    self.model_checkpoint = copy.deepcopy(self.model)

            if epoch % 10000 == 0:
                if test_loss < best_loss:  # check for save best model
                    best_loss = test_loss
                    best_epoch = epoch
                    self.best_model = copy.deepcopy(self.model)
                torch.save(self.best_model.state_dict(), self.run_path / "model")

            # endregion

            # TODO: rimettere l'early stopping!  check per performance
            # early_stopping(test_loss)
            # if early_stopping.early_stop:
            #     #if verbose > 0:
            #     print("Early stopping!!!")
            #     break

        print(f"best loss: {round(best_loss, 3)} at epoch {best_epoch}")
        #if verbose > 0:
        print(f'Epoch: {epoch}\tTest loss: {round(test_loss, 3)} \t\tTrain loss: {round(train_loss, 3)} FINE TRAINING')


        writer.flush()
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), self.run_path / "model")

        #writer_variance.flush()
        #self.myExplained_variance.reset()
        self.last_epoch = epoch

        # aspetto per sicurezza che tutti i processi di salvataggio immagini siano finiti
        for p in parallel_processes_save_images:
            p.join()
        sys.stdout.flush()

        # salvo le animazioni
        if self.conf['training'].get('every_epoch_embedding') and self.epochs != 0:
            self.epochs_list = np.insert(self.epochs_list, 0, 0)
            self.save_all_animations(animation_files, self.epochs_list)

        if not self.config_class.conf['training']['every_epoch_embedding']:
            self.produce_traning_snapshot(self.last_epoch, False, [], last_plot=True)

        return

    def produce_traning_snapshot(self, epoch, parallel, parallel_processes_save_images, **kwargs):
        if self.conf['training'].get('calculate_metrics'):
            #metric_object = self.calc_metric(self.dataset.all_data_loader)
            metric_object_train = self.calc_metric(self.dataset.train_loader)
            metric_object_test = self.calc_metric(self.dataset.test_loader)
            self.metric_obj_list_train.append(metric_object_train)
            self.metric_obj_list_test.append(metric_object_test)

        emb_pergraph_test, emb_pergraph_train, all_embeddings_arrays = \
            self.provide_all_embeddings_and_metrics(give_emb_train=True, give_emb_test=True,
                                                    #metric_object=metric_object,
                                                    metric_object_test=metric_object_test,
                                                    metric_object_train=metric_object_train)
        model_weights = self.provide_model_weights()
        if parallel:
            p = multiprocessing.Process(target=self.save_image_at_epoch,
                                        args=(all_embeddings_arrays, model_weights, epoch,
                                              emb_pergraph_train, emb_pergraph_test,
                                              self.run_path), kwargs=kwargs)
            p.start()
            parallel_processes_save_images.append(p)
        else:
            self.save_image_at_epoch(all_embeddings_arrays, model_weights, epoch,
                                     emb_pergraph_train, emb_pergraph_test,
                                     self.run_path, **kwargs)
        return metric_object_train, metric_object_test

    def save_all_animations(self, animation_files, epochs_list):
        nomefile = self.run_path / str(self.unique_train_name)

        self.save_gif_snapshots(animation_files, nomefile)

        # degree seq
        #self.save_degree_seq_animation(epochs_list)

        new_files = self.save_mp4_snapshots(animation_files, epochs_list, nomefile)

        # ora cancello le singole snapshots
        self.delete_list_of_files(new_files)

    def save_degree_seq_animation(self, epochs_list):
        degims = []
        degree_files = [self.run_path / f"Degree_seq.{self.unique_train_name}_{epoch}.png" for epoch in epochs_list]
        for f in degree_files:
            degims.append(imageio.imread(f))
        imageio.mimwrite(self.run_path / f"Degree_seq.{self.unique_train_name}.gif", degims, duration=0.1)
        self.delete_list_of_files(degree_files)

    def delete_list_of_files(self, files, keep_last=True):
        for f in files[:-1]:
            try:
                os.remove(f)
            except FileNotFoundError as e:
                print(f"Non è riuscito a rimuovere qualche file: {e}")
        if keep_last:
            os.rename(files[-1], self.run_path / "last_epoch.png")

    def save_mp4_snapshots(self, animation_files, epochs_list, nomefile):
        # salvo video in mp4
        # devo rinominare i file in modo sequenziale altrimenti si blocca
        radice = "_epoch"
        mapping = {old: new for new, old in enumerate(epochs_list)}
        new_files = []
        for i, f in enumerate(animation_files):
            old = str(f).split('.')[0].split('/')[-1].replace(radice, '')
            # print(f"file: {str(f)}, old: {old}")
            new_file = self.run_path / f"{radice}{mapping[int(old)]}.png"
            try:
                os.rename(f, new_file)
                new_files.append(new_file)
            except FileNotFoundError as e:
                print(e)

        # chiamo FFMPEG
        save_ffmpeg(self.run_path / radice, nomefile)
        return new_files

    def save_gif_snapshots(self, animation_files, nomefile):
        ims = []
        try:
            for f in animation_files:
                ims.append(imageio.imread(f))
        except FileNotFoundError as e:
            print(f"Errore in save animations: {e}")
        imageio.mimwrite(nomefile.with_suffix(".gif"), ims, duration=0.1)
        # non cancello i file perché servono a salvare anche il video


    def provide_model_weights(self):
        if self.conf['model']['last_layer_dense']:
            model_pars = get_parameters(self.model.convs) + get_parameters(self.model.linears)
            param_labels = get_param_labels(self.model.convs) + get_param_labels(self.model.linears)
        else:
            model_pars = get_parameters(self.model.convs)
            param_labels = get_param_labels(self.model.convs)
        model_weights = model_pars, param_labels
        return model_weights

    def provide_all_embeddings_and_metrics(self, give_emb_train=False, give_emb_test=False, metric_object=None, metric_object_test=None, metric_object_train=None):
        """

        :param give_emb_train:
        :param give_emb_test:
        :param metric_object:
        :param metric_object_test:  serve solo per calcolare la soglia con cui ottenere la binary adjacency matrix
        :param metric_object_train: serve solo per calcolare la soglia con cui ottenere la binary adjacency matrix
        :return:
        """
        if self.config_class.autoencoding:
            all_embeddings_arrays = self.get_embedding_autoencoder(self.dataset.all_data_loader)

            """NON devo piu calcolare i valori a soglia"""
            # if metric_object is not None:
            #     [e.calc_thresholded_values(threshold=metric_object.get_metric("soglia")) for e in all_embeddings_arrays]
            # else:
            #     [e.calc_thresholded_values(threshold=0.5) for e in all_embeddings_arrays]

            if give_emb_train:
                emb_pergraph_train = self.get_embedding_autoencoder(self.dataset.train_loader)
            if give_emb_test:
                emb_pergraph_test = self.get_embedding_autoencoder(self.dataset.test_loader)

            # if metric_object_train is not None:
            #     [e.calc_thresholded_values(threshold=metric_object_train.get_metric("soglia")) for e in emb_pergraph_train]
            # else:
            #     [e.calc_thresholded_values(threshold=0.5) for e in emb_pergraph_train]
            # if metric_object_test is not None:
            #     [e.calc_thresholded_values(threshold=metric_object_test.get_metric("soglia")) for e in emb_pergraph_test]
            # else:
            #     [e.calc_thresholded_values(threshold=0.5) for e in emb_pergraph_test]

        else:
            graph_embeddings_array, node_embeddings_array, _, final_output = self.get_embedding(self.dataset.all_data_loader)
            all_embeddings_arrays = graph_embeddings_array, node_embeddings_array, final_output
        return emb_pergraph_test, emb_pergraph_train, all_embeddings_arrays

    def save_image_at_epoch(self, embedding_arrays, model_weights_and_labels, epoch,
                            emb_pergraph_train=None, emb_pergraph_test=None,
                            path=".", **kwargs):
        # DEVO SEPARARE LA RACCOLTA DEGLI EMBEDDING DAL MULTIPROCESSING, ma mi sa che no nsi chiamare lo spostamento dalla gpu da dentro un processo parallelo
        if self.config_class.autoencoding:
            embedding_class = Embedding_autoencoder(embedding_arrays, config_c=self.config_class, dataset=self.dataset)
            embedding_class.get_metrics(self.embedding_dimension)
            data = DataAutoenc2Plot(embedding_class, dim=self.embedding_dimension,
                                    config_class=self.config_class,
                                    emb_pergraph_train=emb_pergraph_train,
                                    emb_pergraph_test=emb_pergraph_test)
        else:
            graph_embeddings_array, node_embeddings_array, final_output = embedding_arrays
            embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, self.dataset, self.config_class, final_output)
            embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
            embedding_class.separate_embedding_by_classes()  # riempie node_emb_perclass e graph_emb_perclass
            embedding_class.get_metrics(self.embedding_dimension)
            data = Data2Plot(embedding_class.emb_perclass, dim=self.embedding_dimension, config_class=self.exp_config)

        # prendo pure i weights del modello
        model_pars, param_labels = model_weights_and_labels

        # node_emb_dims = embedding_class.node_emb_dims
        # graph_emb_dims = embedding_class.graph_emb_dims
        #p = self.mapping_of_snap_epochs[epoch]
        self.total_node_emb_dim[epoch] = embedding_class.total_node_emb_dim
        self.total_graph_emb_dim[epoch] = embedding_class.total_graph_emb_dim
        node_correlation = embedding_class.total_node_correlation  # node_correlation_per_class
        graph_correlation = embedding_class.total_graph_correlation  # graph_correlation_per_class

        try:
            (all_range_epochs_list, metric_epoch_list,
             metric_obj_list_test, metric_obj_list_train,
             testll, total_graph_emb_dim, total_node_emb_dim, trainll,
             intr_dim_epoch_list) = self.handle_lists_for_plots(epoch, kwargs)

            fig = plot_metrics(data, self.embedding_dimension,
                               testll, all_range_epochs_list,
                               total_node_emb_dim, total_graph_emb_dim,
                               model_pars, param_labels,
                               node_correlation, graph_correlation, sequential_colors=True,
                               showplot=False, last_epoch=self.epochs_list[-1], metric_name=self.name_of_metric,
                               long_string_experiment=self.config_class.long_string_experiment,
                               metric_obj_list_train=metric_obj_list_train,
                               metric_obj_list_test=metric_obj_list_test,
                               train_loss_list=trainll,
                               x_axis_log=self.conf.get("plot").get("x_axis_log"),
                               metric_epoch_list=metric_epoch_list,
                               plot_reconstructed_degree_scatter=True,
                               intr_dim_epoch_list=intr_dim_epoch_list,
                               **kwargs)

            if kwargs.get("notsave"):
                fig.show()
            else:
                file_name = path / f"_epoch{epoch}"
                plt.savefig(file_name)
                fig.clf()
                plt.cla()
                plt.clf()
                if not self.conf['training']['every_epoch_embedding']: # quì ho soltanto una immagine all'inizio e una allafine
                    # salvo solo lultima immagine e la rinomino:
                    os.rename(f"{file_name}.png", path / f"{self.unique_train_name}.png")

            # salvo la sequneza di grado
            #if not kwargs.get("notsave"):
            #    data.plot_output_degree_sequence(ax=axes[0][1], path / f"Degree_seq.{self.unique_train_name}_{epoch}.png")

        except Exception as e:
            print(f"Immagine {epoch} non completata")
            print(e)
            # traceback.print_stack()
            # print(traceback.format_exc())
            print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))

        return

    def handle_lists_for_plots(self, epoch, kwargs):
        metric_epoch_list = self.epochs_list[:np.where(self.epochs_list == epoch)[0][0] + 1]
        if not kwargs.get("unico_plot"):
            testll = self.test_loss_list
            trainll = self.train_loss_list
            all_range_epochs_list = range(epoch + 1)
            metric_obj_list_train = self.metric_obj_list_train
            metric_obj_list_test = self.metric_obj_list_test
            total_node_emb_dim = self.total_node_emb_dim
            total_graph_emb_dim = self.total_graph_emb_dim
            intr_dim_epoch_list = range(self.epochs_list[-1]+1)

            if kwargs.get("last_plot"):
                metric_epoch_list = [0, self.epochs_list[-1]]
                # intr_dim_epoch_list = [0, self.epochs_list[-1]]
                # total_node_emb_dim = [self.total_node_emb_dim[0], self.total_node_emb_dim[-1]]
                # total_graph_emb_dim = [self.total_graph_emb_dim[0], self.total_graph_emb_dim[-1]]

        else:  # quando voglio fare un solo plot, cioè quando carico un modello preaddestrato
            testll = [0]
            trainll = [0]
            all_range_epochs_list = [0]
            # metric_epoch_list = [1]
            metric_obj_list_train = [self.metric_obj_list_train[0]]
            metric_obj_list_test = [self.metric_obj_list_test[0]]
            total_node_emb_dim = [self.total_node_emb_dim[0]]
            total_graph_emb_dim = [self.total_graph_emb_dim[0]]

        return all_range_epochs_list, metric_epoch_list, metric_obj_list_test, metric_obj_list_train, testll, total_graph_emb_dim, total_node_emb_dim, trainll, intr_dim_epoch_list


