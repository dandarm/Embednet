import datetime
import os
import traceback
import copy
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

import multiprocessing

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
from config_valid import Config, TrainingMode
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

        self.percentage_train = None
        self.lr = None
        self.epochs = None
        self.batch_size = None
        self.last_layer_neurons = None
        self.embedding_dimension = None
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

        self.name_of_metric = "accuracy"

        self.softmax = Softmax(dim=1)

        self.gg = None  # class generate_graph
        self.dataset = None
        #self.myExplained_variance = ExplainedVarianceMetric(dimension=self.last_layer_neurons)
        self.last_metric_value = None
        #self.metric_list = None
        self.metric_obj_list = None
        self.test_loss_list = None
        self.train_loss_list = None
        self.last_epoch = None

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
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr),
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
        self.embedding_dimension = self.model.convs[-1].out_channels
        print(f"Optimizer: {self.optimizer}")

    def init_dataset(self, parallel=True, verbose=False):
        """
        la classe GenerateGraph contiene un dataset in formato NetworkX
        Se è un dataset reale viene creato un diverso gestore del dataset
        :param parallel:
        :param verbose:
        :return:
        """
        if not self.config_class.conf['graph_dataset'].get('real_dataset'):
            self.gg = GenerateGraph(self.config_class)
            self.gg.initialize_dataset(parallel=parallel)  # instanzia il dataset networkx
        else:
            # TakeRealDataset si occupa di ottenere il dataset reale
            real_dataset_taker = TakeRealDataset(self.config_class, verbose)
            real_dataset_taker.get_dataset()  # imposta gg.dataset
            self.gg = real_dataset_taker.gg

    def load_dataset(self, dataset, parallel=False):  # dataset è di classe GeneralDataset
        print("Loading Dataset...")
        if not self.config_class.conf['graph_dataset'].get('real_dataset'):            
            self.dataset = Dataset.from_super_instance(self.percentage_train, self.batch_size, self.device, self.config_class, dataset)
            self.dataset.prepare(self.shuffle_dataset, parallel)
        else:
            if self.config_class.conf['graph_dataset']['real_data_name'] == 'REDDIT-BINARY':
                # con questo dataset abbiamo già una lista di oggetti Data
                self.dataset = DatasetReady(self.percentage_train, self.batch_size, self.device, self.config_class, dataset)
            elif self.config_class.conf['graph_dataset']['real_data_name'] == 'BACI':
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


            out2 = out.to(torch.device('cuda'))
            target2 = target.to(torch.device('cuda'), dtype=torch.int16)
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


    def launch_training(self, epochs_list=None, verbose=0):
        self.train_loss_list = []
        self.test_loss_list = []
        #self.metric_obj_list = []
        self.metric_obj_list_train = []
        self.metric_obj_list_test = []

        animation_files = []

        test_loss = self.test(self.dataset.test_loader)
        # if self.dataset.all_data_loader is None:
        #     all_data_loader = DataLoader(self.dataset.dataset_pyg, batch_size=self.dataset.bs, shuffle=False)
        # else:
        #     all_data_loader = self.dataset.all_data_loader
        #alldata_loss = self.test(all_data_loader)
        self.train_loss_list.append(0)
        self.test_loss_list.append(test_loss)

        self.produce_traning_snapshot(0, self.epochs if self.epochs > 0 else 1, False, [])
        file_path = self.rootsave / f"_epoch0.png"
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

        for epoch in tqdm(range(1, self.epochs), total=self.epochs):
            train_loss = self.train()
            writer.add_scalar("Train Loss", train_loss, epoch)

            if epoch % 100 == 0:
                test_loss = self.test(self.dataset.test_loader)
                writer.add_scalar("Test Loss", test_loss, epoch)
                #writer.add_scalar(f"Test {self.name_of_metric}", metric_object.get_metric(self.name_of_metric), epoch)

            #for i, v in enumerate(test_f1score):
            #    writer.add_scalar(f"Test F1 Score/{i}", v, epoch)
            #writerX.add_scalars("Test F1 Score", {f"Class_{i}": v for i, v in enumerate(test_f1score)}, epoch)   TODO: rimettere!
            # writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
            # writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
            # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            self.train_loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)

            if self.conf['training'].get('every_epoch_embedding'):
                if epoch in epochs_list:
                    parallel = True
                    self.produce_traning_snapshot(epoch, epochs_list[-1],
                                                parallel, parallel_processes_save_images)
                    file_path = self.rootsave / f"_epoch{epoch}.png"
                    animation_files.append(file_path)


            #expvar = self.myExplained_variance.compute()
            # add explained variance to tensorboard
            #add_histogram(summary_writer, "Explained variance", var_exp, step=epoch)  TODO: CALCOLO DELLA pca TEMPORANEAMENTE SOSPESO


            # print_each_step = self.conf['logging']['train_step_print']
            # if epoch % print_each_step == 0:
            #     if verbose > 1: print(f'Epoch: {epoch}\tTest loss: {test_loss}')  # \t Explained Variance: {var_exp}')

            # region salva il self.best_model
            if epoch in self.epochs_checkpoint:
                self.model_checkpoint = copy.deepcopy(self.model)
            if test_loss < best_loss:  # check for save best model
                best_loss = test_loss
                if verbose:
                    print(best_loss)
                if self.save_best_model:
                    self.best_model = copy.deepcopy(self.model)
            # endregion

            early_stopping(test_loss)
            if early_stopping.early_stop:
                #if verbose > 0:
                print("Early stopping!!!")
                break


        if verbose > 0:
            print(f'Epoch: {epoch}\tTest loss: {test_loss} \t\tBest test loss: {best_loss} FINE TRAINING')


        writer.flush()
        #writer_variance.flush()
        #self.myExplained_variance.reset()
        #self.last_metric_value = self.metric_list[-1]
        self.last_epoch = epoch

        # aspetto per sicurezza che tutti i processi di salvataggio immagini siano finiti
        for p in parallel_processes_save_images:
            p.join()

        # salvo le animazioni
        if self.conf['training'].get('every_epoch_embedding') and self.epochs != 0:
            self.save_all_animations(animation_files, epochs_list)

        if not self.config_class.conf['training']['every_epoch_embedding']:
            self.produce_traning_snapshot(self.last_epoch, self.last_epoch, False, [])

        return

    def produce_traning_snapshot(self, epoch, last_epoch, parallel, parallel_processes_save_images):
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
                                        args=(all_embeddings_arrays, model_weights, epoch, last_epoch,
                                              emb_pergraph_train, emb_pergraph_test,
                                              self.rootsave))
            p.start()
            parallel_processes_save_images.append(p)
        else:
            self.save_image_at_epoch(all_embeddings_arrays, model_weights, epoch, last_epoch,
                                     emb_pergraph_train, emb_pergraph_test,
                                     self.rootsave)
        return  # metric_object_train, metric_object_test

    def save_all_animations(self, animation_files, epochs_list):
        nomefile = self.rootsave / str(self.unique_train_name)
        ims = []
        try:
            for f in animation_files:
                ims.append(imageio.imread(f))
        except FileNotFoundError as e:
            print(e)
        imageio.mimwrite(nomefile.with_suffix(".gif"), ims, duration=0.1)

        # degree seq
        degims = []
        degree_files = [self.rootsave / f"Degree_seq.{self.unique_train_name}_{epoch}.png" for epoch in epochs_list]
        for f in degree_files:
            degims.append(imageio.imread(f))
        imageio.mimwrite(self.rootsave / f"Degree_seq.{self.unique_train_name}.gif", degims, duration=0.1)
        for f in degree_files:
            os.remove(f)

        # salvo video in mp4
        # devo rinominare i file in modo sequenziale altrimenti si blocca
        radice = "_epoch"  # TODO: cambiare in "nomefile"
        mapping = {old: new for new, old in enumerate(epochs_list)}
        new_files = []
        for i, f in enumerate(animation_files):
            old = str(f).split('.')[0].split('/')[-1].replace(radice, '')
            # print(f"file: {str(f)}, old: {old}")
            new_file = self.rootsave / f"{radice}{mapping[int(old)]}.png"
            # print(new_file)
            try:
                os.rename(f, new_file)
                new_files.append(new_file)
            except FileNotFoundError as e:
                print(e)

        save_ffmpeg(self.rootsave / radice, nomefile)
        # files = new_files

        for f in new_files:
            os.remove(f)

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
            if metric_object is not None:
                [e.calc_thresholded_values(threshold=metric_object.get_metric("soglia")) for e in all_embeddings_arrays]
            else:
                [e.calc_thresholded_values(threshold=0.5) for e in all_embeddings_arrays]

            if give_emb_train:
                emb_pergraph_train = self.get_embedding_autoencoder(self.dataset.train_loader)
            if give_emb_test:
                emb_pergraph_test = self.get_embedding_autoencoder(self.dataset.test_loader)

            if metric_object_train is not None:
                [e.calc_thresholded_values(threshold=metric_object_train.get_metric("soglia")) for e in emb_pergraph_train]
            else:
                [e.calc_thresholded_values(threshold=0.5) for e in emb_pergraph_train]
            if metric_object_test is not None:
                [e.calc_thresholded_values(threshold=metric_object_test.get_metric("soglia")) for e in emb_pergraph_test]
            else:
                [e.calc_thresholded_values(threshold=0.5) for e in emb_pergraph_test]

        else:
            graph_embeddings_array, node_embeddings_array, _, final_output = self.get_embedding(self.dataset.all_data_loader)
            all_embeddings_arrays = graph_embeddings_array, node_embeddings_array, final_output
        return emb_pergraph_test, emb_pergraph_train, all_embeddings_arrays

    def save_image_at_epoch(self, embedding_arrays, model_weights_and_labels, epoch, last_epoch,
                            emb_pergraph_train=None, emb_pergraph_test=None,
                            metric_obj_list_train=None, metric_obj_list_test=None,  rootsave="."):
        # DEVO SEPARARE LA RACCOLTA DEGLI EMBEDDING DAL MULTIPROCESSING
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
        node_intrinsic_dimensions_total = embedding_class.total_node_emb_dim
        graph_intrinsic_dimensions_total = embedding_class.total_graph_emb_dim
        node_correlation = embedding_class.total_node_correlation  # node_correlation_per_class
        graph_correlation = embedding_class.total_graph_correlation  # graph_correlation_per_class
        # costruisco e salvo l'immagine: per ora sincrono, poi lancerò un thread asincrono

        try:
            fig = plot_metrics(data, self.embedding_dimension,
                               self.test_loss_list[:epoch], range(epoch),
                               node_intrinsic_dimensions_total, graph_intrinsic_dimensions_total,
                               model_pars, param_labels,
                               node_correlation, graph_correlation, sequential_colors=True,
                               showplot=False, last_epoch=last_epoch, metric_name=self.name_of_metric,
                               long_string_experiment=self.config_class.long_string_experiment,
                               metric_obj_list_train=self.metric_obj_list_train[:epoch],
                               metric_obj_list_test=self.metric_obj_list_test[:epoch],
                               train_loss_list=self.train_loss_list[:epoch])
            file_name = self.rootsave / f"_epoch{epoch}"
            plt.savefig(file_name)
            fig.clf()
            plt.cla()
            plt.clf()
            if not self.conf['training']['every_epoch_embedding']:
                # salvo solo lultima immagine e la rinomino:
                os.rename(f"{file_name}.png", self.rootsave / f"{self.unique_train_name}.png")

            # salvo la sequneza di grado
            data.plot_output_degree_sequence(self.rootsave / f"Degree_seq.{self.unique_train_name}_{epoch}.png")

        except Exception as e:
            print(f"Immagine {epoch} non completata")
            print(e)
            # traceback.print_stack()
            # print(traceback.format_exc())
            print(''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))

        return #file_path


