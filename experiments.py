import imageio
from io import BytesIO
import os
import time

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import animation
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
#torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from multiprocessing import Pool, Process, Manager
import torch_geometric.transforms as T

from models import GAEGCNEncoder, view_parameters, get_param_labels, new_parameters, modify_parameters, new_parameters_linears, modify_parameters_linear
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding
import plot_funcs
from plot_funcs import scatter_node_emb, plot_graph_emb_1D, plot_node_emb_1D, save_ffmpeg, Data2Plot, plot_weights_multiple_hist
from plt_parameters import get_colors_to_cycle_rainbow8, get_colors_to_cycle_rainbowN, get_colors_to_cycle_sequential
from config_valid import Config
from GridConfigurations import GridConfigurations
from graph_generation import GraphType
from utils import array_wo_outliers

# per usare il trainer e il config nei processi paralleli
graph_embedding_per_epoch = []
node_embedding_per_epoch = []
model_linear_pars_per_epoch = []
model_gconv_pars_per_epoch = []
param_labels = []
absmin = 1000000
absmax = -1000000
dataset = None
loss_list = []
exp_config = None
output_per_epoch = []
accuracy_list = []
dataset_type = None
embedding_dimension = None
trainmode = None
sequential_colors = False
bounds_for_plot = []
node_intrinsic_dimensions_perclass = []
graph_intrinsic_dimensions_perclass = []
node_intrinsic_dimensions_total = []
graph_intrinsic_dimensions_total = []

num_classes = None
data4video = []

def all_seeds():
    os.environ['PYTHONHASHSEED'] = str(0)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)  # anche da richiamare tra una chiamata e l'altra del training
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    #torch.use_deterministic_algorithms(True)


class Experiments():
    def __init__(self, config_file, diz_trials, rootsave, config_class=None):
        self.eseguito=False
        self.config_file = config_file
        if config_class:
            self.config_class = config_class
        else:
            self.config_class = Config(self.config_file)

        self.rootsave = rootsave
        self.trainer = Trainer(self.config_class)
        self.gc = None
        self.diz_trials = diz_trials

        all_seeds()

        if self.diz_trials:
            self.gc = GridConfigurations(self.config_class, self.diz_trials)
            self.gc.make_configs()

        # risultati
        #self.graph_embedding_per_epoch = []
        #self.node_embedding_per_epoch = []

    def GS_simple_experiments(self):

        k = 0
        for c in self.gc.configs:
            print(f'Run {k + 1} \t\t exp name: {c.unique_train_name}')
            # all_seeds()
            self.trainer.reinit_conf(c)
            self.just_train()
            embedding_class = self.embedding()

            #fill_df_with_results(self.gc.config_dataframe, k, None, None, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
            k += 1

    def stesso_init_diversi_dataset(self):
        self.GS_same_weight_inits_different_datasets(test_same_training=False)
    def GS_same_weight_inits_different_datasets(self):
        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = self.trainer.init_GCN()
        saved_initial_weights_lin = new_parameters_linears(modello_base)
        saved_initial_weights_gcn = new_parameters(modello_base, method=self.config_class.init_weights_mode)

        for i, c in enumerate(self.gc.configs):
            print(f'Run {i + 1}/{len(self.gc.configs)}')
            all_seeds()
            self.trainer.reinit_conf(c)

            model = self.trainer.init_GCN(saved_initial_weights_gcn, saved_initial_weights_lin)
            self.trainer.load_model(model)

            # TODO: verificare che vengono effettivamente diversi i dataset perché ho resettatto i seed subito prima
            self.trainer.init_dataset()
            self.trainer.load_dataset(self.trainer.gg.dataset)

            self.trainer.launch_training()

            embedding_class = self.embedding()
            fill_df_with_results(self.gc.config_dataframe, i, None, None, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)

    def diversi_init_weights_stesso_dataset(self, parallel=True):  #, metodi, ripetizioni):
        self.GS_different_weight_inits( train_with_same_dataset=True, test_same_training=False, parallel_take_result=parallel)

    def GSdiversi_init_weights_diversi_dataset(self, dataset_key1, initw_key2=None, parallel_take_result=True):
        global graph_embedding_per_epoch
        global node_embedding_per_epoch
        global dataset
        global loss_list
        global exp_config

        k = 0
        for el in self.diz_trials[dataset_key1]:  # come fosse for d in dataset_distinti
            cs1 = [(i, c) for i, c in enumerate(self.gc.configs) if c.conf['graph_dataset']['list_exponents'] == el]
            print(f"Run {k + 1}: \t {el}")
            # devo comfigurare il trainer con uno di questi config per poter inizializzare il dataset
            # inoltre serve pure per il modello di base, per avere l'architettura dei parametri da impostare
            # per l'architettura del modello mi serve il numero delle classi, la prendo in uno dei config col dato dataset
            self.trainer.reinit_conf(cs1[0][1])

            self.trainer.init_dataset()
            self.trainer.load_dataset(self.trainer.gg.dataset)

            modello_base = self.trainer.init_GCN()
            saved_initial_weights_lin = new_parameters_linears(modello_base)  # indifferente se epoch==0

            # inizializzazioni distinte
            metodi_distinti = set([c.conf['model']['init_weights'] for i, c in cs1])
            for m in metodi_distinti:
                cs2 = [(i, c) for i, c in cs1 if c.conf['model']['init_weights'] == m]
                print(f"Run {k + 1} \t metodo:{m}")

                for ord_config in cs2:
                    i, c = ord_config

                    init_weight_parameters = new_parameters(modello_base, method=c.init_weights_mode)
                    model = self.trainer.init_GCN(init_weight_parameters, saved_initial_weights_lin)
                    self.trainer.reinit_conf(c)
                    self.trainer.load_model(model)
                    self.trainer.launch_training()
                    embedding_class = self.embedding()

                    print("calcolo i risultati di interesse")
                    if c.conf['model']['freezeGCNlayers']:  # non ha senso guardare la correlazione iun funzione delle epoche di training
                        fill_df_with_results(self.gc.config_dataframe, i, None, None, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
                    else:
                        if parallel_take_result:
                            graph_embedding_per_epoch = self.trainer.graph_embedding_per_epoch
                            node_embedding_per_epoch = self.trainer.node_embedding_per_epoch
                            dataset = self.trainer.dataset
                            loss_list = self.trainer.test_loss_list
                            exp_config = c
                        avg_corr_classes, avg_tau_classes = self.get_corrs_training(parallel=parallel_take_result)
                        avg_corr_classes = np.array(avg_corr_classes).T
                        avg_tau_classes = np.array(avg_tau_classes).T

                        fill_df_with_results(self.gc.config_dataframe, i, avg_corr_classes, avg_tau_classes, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
                    k += 1

    def GS_different_weight_inits(self, train_with_same_dataset=False, test_same_training=False, parallel_take_result=True):
        global graph_embedding_per_epoch
        global node_embedding_per_epoch
        global dataset
        global loss_list
        global exp_config

        if train_with_same_dataset:
            print("Carico il dataset e lo tengo invariato per tutti i trial")
            self.trainer.init_dataset()
            self.trainer.load_dataset(self.trainer.gg.dataset)

        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = self.trainer.init_GCN()
        saved_initial_weights_lin = new_parameters_linears(modello_base)
        if test_same_training:
            saved_initial_weights_gcn = new_parameters(modello_base)

        # ordino tutte le inizializzazioni dello stesso tipo, per dover gestire bene il reset dei seed
        # gc.configs = sorted(gc.configs, key=lambda c: c.conf['model']['init_weights'])
        # meglio il seguente metodo
        metodi_distinti = set([c.conf['model']['init_weights'] for c in self.gc.configs])
        print(metodi_distinti)
        k = 0
        for m in metodi_distinti:
            print(m)
            cs = [(i, c) for i, c in enumerate(self.gc.configs) if c.conf['model']['init_weights'] == m]
            all_seeds()
            # diversi training dell'unico parametro diverso ripescato random dalla stessa distribuzione
            for ord_config in cs:
                i, c = ord_config
                print(f"Run {k + 1} \t metodo:{c.conf['model']['init_weights']}")

                self.trainer.reinit_conf(c)
                if not train_with_same_dataset:
                    self.trainer.init_dataset()
                    self.trainer.load_dataset(self.trainer.gg.dataset)
                if test_same_training:
                    model = self.trainer.init_GCN(saved_initial_weights_gcn, saved_initial_weights_lin)
                else:
                    # questa chiamata non deve avere i seed resettati altrimenti otterrò gli stessi pesi e lo stesso training
                    init_weight_parameters = new_parameters(modello_base, method=c.init_weights_mode)
                    model = self.trainer.init_GCN(init_weight_parameters, saved_initial_weights_lin)


                self.trainer.load_model(model)

                self.trainer.launch_training()
                embedding_class = self.embedding()

                print("calcolo i risultati di interesse")
                if c.conf['model']['freezeGCNlayers']:  # non ha senso guardare la correlazione iun funzione delle epoche di training
                    fill_df_with_results(self.gc.config_dataframe, i, None, None, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
                else:
                    if parallel_take_result:
                        graph_embedding_per_epoch = self.trainer.graph_embedding_per_epoch
                        node_embedding_per_epoch = self.trainer.node_embedding_per_epoch
                        dataset = self.trainer.dataset
                        loss_list = self.trainer.test_loss_list
                        exp_config = c
                    avg_corr_classes, avg_tau_classes = self.get_corrs_training(parallel=parallel_take_result)
                    avg_corr_classes = np.array(avg_corr_classes).T
                    avg_tau_classes = np.array(avg_tau_classes).T

                    fill_df_with_results(self.gc.config_dataframe, i, avg_corr_classes, avg_tau_classes, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
                k+=1

    def diverse_classi_stesso_dataset(self, parallel_take_result):  #, metodi, ripetizioni):
        self.GS_different_classes(test_same_training=False)

    def GS_different_classes(self, test_same_training=False, parallel_take_result=False):
        global graph_embedding_per_epoch
        global node_embedding_per_epoch
        global dataset
        global loss_list
        global exp_config

        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = self.trainer.init_GCN()
        #saved_initial_weights_lin = new_parameters_linears(modello_base)
        saved_initial_weights_gcn = new_parameters(modello_base, method=self.trainer.config_class.init_weights_mode)
        if test_same_training:
            self.trainer.init_dataset()
            self.trainer.load_dataset(self.trainer.gg.dataset)

        k = 0
        for c in self.gc.configs:
            print(f'Run {k + 1}')
            self.trainer.reinit_conf(c)
            if not test_same_training:
                self.trainer.init_dataset()
                self.trainer.load_dataset(self.trainer.gg.dataset)
            model = self.trainer.init_GCN(saved_initial_weights_gcn) #, saved_initial_weights_lin)
            self.trainer.load_model(model)

            self.trainer.launch_training()
            embedding_class = self.embedding()

            print("calcolo i risultati di interesse")
            #exp_trainer = trainer
            #exp_config = c
            if parallel_take_result:
                graph_embedding_per_epoch = self.trainer.graph_embedding_per_epoch
                node_embedding_per_epoch = self.trainer.node_embedding_per_epoch
                dataset = self.trainer.dataset
                loss_list = self.trainer.test_loss_list
                exp_config = c

            avg_corr_classes, avg_tau_classes = self.get_corrs_training(parallel=parallel_take_result)
            avg_corr_classes = np.array(avg_corr_classes).T
            avg_tau_classes = np.array(avg_tau_classes).T

            fill_df_with_results(self.gc.config_dataframe, k, avg_corr_classes, avg_tau_classes, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
            #fill_embedding_df(self.gc.embedding_dataframe, trainer, k)
            k += 1

    def GS_different_linear_layers(self):
        """
        Presuppongo che i layer GCN siano FREEZED
        :return:
        """
        global graph_embedding_per_epoch
        global node_embedding_per_epoch
        global dataset
        global loss_list
        global exp_config

        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = self.trainer.init_GCN()
        saved_initial_weights_gcn = new_parameters(modello_base, method=self.trainer.config_class.init_weights_mode)
        # dataset sempre uguale
        self.trainer.init_dataset()
        self.trainer.load_dataset(self.trainer.gg.dataset)

        k = 0
        for c in self.gc.configs:
            print(f'Run {k + 1}')
            #all_seeds()
            self.trainer.reinit_conf(c)
            # i parametri linear ora sono sempre diversi perché sto prendendo un numero sempre diverso di neuroni e layers
            model = self.trainer.init_GCN(saved_initial_weights_gcn)  # , saved_initial_weights_lin)
            self.trainer.load_model(model)
            self.trainer.launch_training()
            embedding_class = self.embedding()

            fill_df_with_results(self.gc.config_dataframe, k, None, None, self.trainer.test_loss_list, self.trainer.accuracy_list, embedding_class)
            k += 1


    def just_train(self, parallel=True, verbose=False):
        #self.trainer = Trainer(self.config_class)
        self.trainer.init_all(parallel=parallel, verbose=verbose)
        self.trainer.launch_training()
    def embedding(self):
        if self.config_class.conf['training']['save_best_model']:
            self.trainer.model = self.trainer.best_model
        graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output = self.trainer.take_embedding_all_data()
        embedding_class = self.elaborate_embedding(graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output, self.trainer.model.linears)

        return embedding_class

    def elaborate_embedding(self, graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, output_array, modelparams):
        embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, self.trainer.dataset, self.trainer.config_class, output_array, modelparams)
        embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
        embedding_class.separate_embedding_by_classes()  # riempie node_emb_perclass e graph_emb_perclass
        #if not self.config_class.conf['graph_dataset']['continuous_p']:
        #    embedding_class.get_graph_emb_per_class()  # riempie

        return embedding_class

    def take_corr(self, epoca):
        graph_embeddings_array = self.trainer.graph_embedding_per_epoch[epoca]
        node_embeddings_array = self.trainer.node_embedding_per_epoch[epoca]
        node_embedding_class = self.elaborate_embedding(graph_embeddings_array, node_embeddings_array, [])
        # node_emb_pergraph, node_embedding_class = elaborate_embedding_pergraph(c, graph_embeddings_array, node_embeddings_array, [], t)
        # embs_by_class = separate_embedding_by_classes(node_emb_pergraph)

        avg_corr_classes = []
        avg_tau_classes = []
        for classe in node_embedding_class.emb_perclass:
            corrs = []
            kendall_tau = []
            for e in classe:
                e.get_correlation_with_degree_sequence()
                e.get_kendall_with_degree_sequence()
                corrs.append(e.correlation_with_degree)
                kendall_tau.append(e.kendall_with_degree)
            avg_corr_class0 = sum(corrs) / len(classe)
            avg_corr_classes.append(avg_corr_class0)
            avg_tau = sum(kendall_tau) / len(classe)
            avg_tau_classes.append(avg_tau)

        return avg_corr_classes, avg_tau_classes

    def get_corrs_training(self, parallel=True):
        print(self.trainer.last_epoch)
        if parallel:
            with Pool(processes=32) as pool:
                res = pool.map(parallel_take_corr, range(max(self.trainer.last_epoch, 1)))
                avg_corr_classes = [r[0] for r in res]
                avg_tau_classes = [r[1] for r in res]
        else:
            avg_corr_classes = []
            avg_tau_classes = []
            for e in range(self.trainer.last_epoch + 1):
                rr = self.take_corr(e)
                avg_corr_classes.append(rr[0])
                avg_tau_classes.append(rr[1])

        return avg_corr_classes, avg_tau_classes


    # region save video or gif

    def save_many_images_embedding(self, trainer, config_c, node_embeddings_array_id=[0]):
        for i in range(trainer.epochs):
            graph_embeddings_array = trainer.graph_embedding_per_epoch[i]
            node_embeddings_array = trainer.node_embedding_per_epoch[i]
            emb_perclass0, emb_perclass1 = self.elaborate_embeddings(config_c, graph_embeddings_array, trainer.model, node_embeddings_array, node_embeddings_array_id, trainer.test_loss_list, trainer)
            scatter_node_emb(emb_perclass0, emb_perclass1, trainer.accuracy_list[i], f"scatter_epoch{i}", show=False, close=True)
            # plot_graph_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy)


    def parallel_save_many_images_embedding(self, lista):
        with Pool(processes=12) as pool:
            pool.map(mylambda_save, lista)
        return

    def parallel_many_images_embedding(self, lista):
        with Pool(processes=30) as pool:
            bios = pool.map(mylambda_memory, lista)
        return bios

    def parallel_prepare_data4video_each_epoch(self, lista):
        with Pool(processes=30) as pool:
            res = pool.map(prepare_data4video_each_epoch, lista)
        return res

    def make_video(self, skip=1, fromfiles=True, isgif=False, both=True, custom_list=False, seq_colors=False):
        global output_per_epoch
        global loss_list
        global accuracy_list
        global sequential_colors
        global bounds_for_plot
        global model_linear_pars_per_epoch
        global model_gconv_pars_per_epoch
        global param_labels
        global absmin
        global absmax
        global data4video
        global node_intrinsic_dimensions_perclass
        global graph_intrinsic_dimensions_perclass
        global node_intrinsic_dimensions_total
        global graph_intrinsic_dimensions_total

        if both:
            isgif = True
            isvideo = True
            fromfiles = True

        if custom_list:
            end = np.log10(self.trainer.last_epoch)
            logarray = np.round(np.logspace(1., end, num=100)).astype(int)
            lista = np.unique(np.concatenate((np.arange(0, 10), logarray)))[:-1]
        else:
            lista = range(0, self.trainer.last_epoch, skip)

        output_per_epoch = self.trainer.output_per_epoch
        loss_list = self.trainer.test_loss_list
        accuracy_list = self.trainer.accuracy_list
        sequential_colors = seq_colors
        model_linear_pars_per_epoch = self.trainer.model_LINEAR_pars_per_epoch
        model_gconv_pars_per_epoch = self.trainer.model_GCONV_pars_per_epoch
        lin_param_labels = get_param_labels(self.trainer.model.linears)
        gcn_param_labels = get_param_labels(self.trainer.model.convs)
        param_labels = gcn_param_labels + lin_param_labels

        # calcolo i massimi e minimi di tutti i model pars
        absmin = []
        absmax = []
        for layers in model_linear_pars_per_epoch:
            absmin.append(np.min([np.min(par) for par in layers]))
            absmax.append(np.max([np.max(par) for par in layers]))
        for layers in model_gconv_pars_per_epoch:
            absmin.append(np.min([np.min(par) for par in layers]))
            absmax.append(np.max([np.max(par) for par in layers]))
        absmin = np.min(absmin)
        absmax = np.max(absmax)

        node_embedding_per_epoch_ = np.array(node_embedding_per_epoch)#.squeeze()
        graph_embedding_per_epoch_ = np.array(graph_embedding_per_epoch)#.squeeze()
        output_per_epoch_ = np.array(output_per_epoch)#.squeeze()

        print(node_embedding_per_epoch_.shape)
        if embedding_dimension > 1:
            bounds_for_plot.append(axis_bounds(node_embedding_per_epoch_))
            bounds_for_plot.append(axis_bounds(graph_embedding_per_epoch_))
            bounds_for_plot.append(axis_bounds(output_per_epoch_))
        else:
            bounds_for_plot = None


        #with Manager() as manager:
        print(f"Preparing data for make video...")
        start = time.time()
        #data4video = manager.list([None]*self.trainer.last_epoch)
        #intrinsic_dimensions = manager.list([None]*self.trainer.last_epoch)
        d4v_e_ids = self.parallel_prepare_data4video_each_epoch(lista)
        d4v = np.array([d for d,_,_,_,_ in d4v_e_ids])
        id_node_emb = np.array([node for _,node,_,_,_ in d4v_e_ids])
        id_graph_emb = np.array([graph for _, _, graph, _, _ in d4v_e_ids])
        id_tot_node_emb = np.array([totnode for _,_,_,totnode,_ in d4v_e_ids])
        id_tot_graph_emb = np.array([totgraph for _, _, _, _, totgraph in d4v_e_ids])

        data4video = [None]*self.trainer.last_epoch
        node_intrinsic_dimensions_perclass = [[None]*num_classes] * self.trainer.last_epoch
        graph_intrinsic_dimensions_perclass = [[None]*num_classes] * self.trainer.last_epoch
        node_intrinsic_dimensions_total = [None] * self.trainer.last_epoch
        graph_intrinsic_dimensions_total = [None] * self.trainer.last_epoch
        print(f"{len(id_node_emb)} è la len di id_node_emb")
        for i,l in enumerate(lista):
            data4video[l] = d4v[i]
            node_intrinsic_dimensions_perclass[l] = id_node_emb[i]
            graph_intrinsic_dimensions_perclass[l] = id_graph_emb[i]
            node_intrinsic_dimensions_total[l] = id_tot_node_emb[i]
            graph_intrinsic_dimensions_total[l] = id_tot_graph_emb[i]

        end = time.time()
        print(f"({round(end - start, 2)} s.)  ...data ready: make video...", end='')
        start = time.time()

        if fromfiles:
            self.parallel_save_many_images_embedding(lista)
            files = [f"scatter_epoch{i}.png" for i in lista]
        else:
            pictures = self.parallel_many_images_embedding(lista)
        end = time.time()
        print(f"({round(end - start, 2)} s.)")

        nomefile = self.trainer.unique_train_name
        print(nomefile)

        if isgif:
            if fromfiles:
                # files = [f"scatter_epoch{i}.png" for i in lista]
                ims = [imageio.imread(f) for f in files]
                imageio.mimwrite(nomefile + ".gif", ims, duration=0.1)
            else:
                ims = [imageio.imread(f) for f in pictures]
                imageio.mimwrite(nomefile, ims, duration=0.1)
        if isvideo:  # con ffmpeg per forza da file...o no?
            # devo rinominare i file in modo sequenziale altrimenti si blocca
            radice = "scatter_epoch"
            mapping = {old: new for new, old in enumerate(lista)}
            new_files = []
            for i, f in enumerate(files):
                old = f.replace(radice, '').split('.')[0]
                new_file = f"{radice}{mapping[int(old)]}.png"
                new_files.append(new_file)
                os.rename(f, new_file)

            save_ffmpeg(radice, nomefile)
            files = new_files

        if fromfiles:
            for f in files:
                os.remove(f)

        return nomefile

        # endregion
def prepare_data4video_each_epoch(i):
    graph_embeddings_array = graph_embedding_per_epoch[i]
    node_embeddings_array = node_embedding_per_epoch[i]
    output_array = output_per_epoch[i]

    embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, dataset, exp_config, output_array)
    embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
    embedding_class.separate_embedding_by_classes()
    embedding_class.get_metrics(embedding_dimension, trainmode)

    #intrinsic_dimensions[i] = embedding_class.graph_emb_dims
    data = Data2Plot(embedding_class.emb_perclass, dim=embedding_dimension)
    #data4video[i] = data
    #return data, embedding_class.graph_emb_dims
    node_emb_dims = embedding_class.node_emb_dims
    graph_emb_dims = embedding_class.graph_emb_dims
    total_node_emb_dim = embedding_class.total_node_emb_dim
    total_graph_emb_dim = embedding_class.total_graph_emb_dim
    return data, node_emb_dims, graph_emb_dims, total_node_emb_dim, total_graph_emb_dim

def parallel_take_corr(epoca):  # TODO: correggere errore quando chiama Embedding: 'Embedding' object has no attribute 'node_emb_perclass'
    graph_embeddings_array = graph_embedding_per_epoch[epoca]
    node_embeddings_array = node_embedding_per_epoch[epoca]
    embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, dataset, exp_config)
    embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
    embedding_class.separate_embedding_by_classes()  # riempie node_emb_perclass
    # node_emb_pergraph, node_embedding_class = elaborate_embedding_pergraph(c, graph_embeddings_array, node_embeddings_array, [], t)
    # embs_by_class = separate_embedding_by_classes(node_emb_pergraph)

    avg_corr_classes = []
    avg_tau_classes = []
    for classe in embedding_class.node_emb_perclass:
        corrs = []
        kendall_tau = []
        for e in classe:
            e.get_correlation_with_degree_sequence()
            e.get_kendall_with_degree_sequence()
            corrs.append(e.correlation_with_degree)
            kendall_tau.append(e.kendall_with_degree)
        avg_corr_class0 = sum(corrs) / len(classe)
        avg_corr_classes.append(avg_corr_class0)
        avg_tau = sum(kendall_tau) / len(classe)
        avg_tau_classes.append(avg_tau)

    return avg_corr_classes, avg_tau_classes

def mylambda_save(i):
    fig = mylambda_figure(i)
    plt.savefig(f"scatter_epoch{i}")
    fig.clf()
    plt.cla()
    plt.clf()
    plt.close("all")
    return

def mylambda_memory(i):
    fig = mylambda_figure(i)
    bio = BytesIO()
    fig.savefig(bio, format="png")
    plt.close()
    return bio
def mylambda_figure(i):
    data = data4video[i]
    model_linear_pars = model_linear_pars_per_epoch[i]
    model_gcn_pars = model_gconv_pars_per_epoch[i]
    model_pars = model_gcn_pars + model_linear_pars

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    if embedding_dimension == 1:
        data.plot(datatype='node_embedding', type='scatter', ax=axes[0][0], sequential_colors=sequential_colors, title="Node Embedding")
        if bounds_for_plot is not None: axes[0][0].axis(bounds_for_plot[0])
        data.plot(datatype='graph_embedding', type='histogram', ax=axes[0][1], sequential_colors=sequential_colors, title="Graph Embedding")
        if bounds_for_plot is not None: axes[0][1].axis(bounds_for_plot[1])
        data.plot(datatype='final_output', type='plot', ax=axes[0][2], sequential_colors=sequential_colors, title="Final Output")
        if bounds_for_plot is not None: axes[0][2].axis(bounds_for_plot[2])
    else:
        data.plot(datatype='node_embedding', type='plot', ax=axes[0][0], sequential_colors=sequential_colors, title="Node Embedding")
        #if bounds_for_plot is not None: axes[0][0].axis(bounds_for_plot[0])
        data.plot(datatype='graph_embedding', type='plot', ax=axes[0][1], sequential_colors=sequential_colors, title="Graph Embedding")
        #if bounds_for_plot is not None: axes[0][1].axis(bounds_for_plot[1])
        data.plot(datatype='final_output', type='plot', ax=axes[0][2], sequential_colors=sequential_colors, title="Final Output")
        #if bounds_for_plot is not None: axes[0][2].axis(bounds_for_plot[2])
    #scatter_node_emb(embedding_class.emb_perclass, ax=axes[0][0], show=False, close=False, sequential_colors=True)
    #plot_graph_emb_1D(embedding_class.emb_perclass, ax=axes[0][1], show=False, close=False, sequential_colors=True)
    #axes[0][2].hist(np.array(output_array).flatten(), bins=50);

    # plot Test loss e accuracy senza outlier
    # test loss
    loss_list_min, loss_list_max = min(array_wo_outliers(loss_list)), max(array_wo_outliers(loss_list))
    axes[1][0].plot(loss_list[:i], color='black', label='Test Loss')
    axes[1][0].plot(i, loss_list[i], 'ro')
    axes[1][0].set_ylim(loss_list_min, loss_list_max)
    axes[1][0].set_xlim(0, len(loss_list))
    #axes[1][0].set_ylabel('Test Loss')#, fontsize=16);

    # plot accuracy
    axt = axes[1][0].twinx()
    axt.plot(accuracy_list[:i], color='blue', label='Accuracy')
    axt.plot(i, accuracy_list[i], 'ro')
    axt.set_ylim(0,1)
    #axt.set_ylabel('Accuracy')#, fontsize=16);
    axt.set_xlim(0, len(accuracy_list))
    axt.set_yticklabels([])

    axes[1][0].legend(loc='lower left')
    axt.legend()

    # plot misure (correlazione, dimensionalità intrinseca, overlap degli embedding...)
    #custom_cycler = cycler(color=get_colors_to_cycle_sequential(len(graph_intrinsic_dimensions_perclass[0])))
    #axes[1][1].set_prop_cycle(custom_cycler)
    axes[1][1].plot(node_intrinsic_dimensions_total[:i], linestyle='None', marker='.', color='red', label='node id') #
    axes[1][1].plot(graph_intrinsic_dimensions_total[:i], linestyle='None', marker='.', color='blue', label='graph id')
    axes[1][1].set_xlim(0, len(graph_intrinsic_dimensions_total))
    axes[1][1].set_ylim(0, 3)
    axes[1][1].set_title(f"Intrinsic Dimensionality")
    axes[1][1].legend()

    plot_weights_multiple_hist(model_pars, param_labels, axes[1][2], absmin, absmax, sequential_colors=True)
    fig.suptitle(f"Network dynamics")
    return fig

def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    if bottom == top:
        bottom = bottom -1
        top = top + 1
    if left == right:
        left = left - 1
        right = right + 1
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]



def my_animated_figure(i):
    graph_embeddings_array = graph_embedding_per_epoch[i]
    node_embeddings_array = node_embedding_per_epoch[i]
    output_array = output_per_epoch[i]

    embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, dataset, exp_config, output_array)
    embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
    embedding_class.separate_embedding_by_classes()
    data = Data2Plot(embedding_class.emb_perclass, dim=embedding_dimension)
    data.set_data('node_embedding')
    Data2Plot.static_plot_for_animation(data.array2plot, data.class_labels)


def experiment_node_embedding(config_file):
    config_class = Config(config_file)
    trainer = Trainer(config_class)
    trainer.init_all(config_file)
    trainer.launch_training()
    graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output = trainer.take_embedding_all_data(type_embedding='both')

    for i in range(trainer.epochs):
        graph_embeddings_array = trainer.graph_embedding_per_epoch[i]
        node_embeddings_array = trainer.node_embedding_per_epoch[i]
        embs_by_class, node_emb_pergraphclass  = Experiments.elaborate_embeddings(config_class, graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, trainer)
        scatter_node_emb(embs_by_class, trainer.last_accuracy)
        plot_graph_emb_1D(embs_by_class, trainer.last_accuracy)







def fill_df_with_results(df, i, avg_corr_classes, avg_tau_classes, test_loss_list, accuracy_list, embedding_class):
    df.loc[i, ('risultati', 'data')] = pd.Timestamp.now()
    df.at[i, ('risultati', 'test_loss')] = test_loss_list
    df.at[i, ('risultati', 'test_accuracy')] = accuracy_list
    if avg_corr_classes:
        df.at[i, ('risultati', 'correlation_allclasses')] = avg_corr_classes.tolist()
    else:
        df.at[i, ('risultati', 'correlation_allclasses')] = None
    if avg_tau_classes:
        df.at[i, ('risultati', 'tau_allclasses')] = avg_tau_classes.tolist()
    else:
        df.at[i, ('risultati', 'tau_allclasses')] = None
    df.at[i, ('risultati', 'embedding_class')] = embedding_class

def fill_embedding_df(df, trainer):
    """
    riempie con i valori degli embedding , anche per ogni epoca
    :param df:
    :param trainer:
    :param i:
    :return:
    """
    #df = pd.DataFrame()  # data=xp.trainer.node_embedding_per_epoch)
    for i in range(len(trainer.node_embedding_per_epoch)):
        df[i] = trainer.node_embedding_per_epoch[i].flatten()


def verify_two_same_trainings(config_file, tentativi):
    config_class = Config(config_file)
    trainer = Trainer(config_class)

    # inizializzo dei parametri
    model = trainer.init_GCN()
    saved_initial_weights_gcn = new_parameters(model)
    saved_initial_weights_lin = new_parameters_linears(model)

    # inizializzo dataset
    trainer.init_dataset()
    trainer.load_dataset(trainer.gg.dataset)

    test_losses = []
    for i in range(tentativi):
        all_seeds()
        model = trainer.init_GCN(saved_initial_weights_gcn, saved_initial_weights_lin)
        trainer.load_model(model)
        trainer.launch_training()
        test_losses.append(trainer.test_loss_list)

    return test_losses


def experiment_node_emb_cm(config_file, methods, ripetiz=30):
    config_c = Config(config_file)
    trainer = Trainer(config_c)
    model, trainer = trainer.init_all(verbose=False)
    for method in methods:
        for i in range(ripetiz):
            model = trainer.init_GCN()
            new_par = new_parameters(model, method=method)
            model = trainer.init_GCN(new_par, init_weights_lin=None)
            trainer.load_model(model)

            trainer.launch_training(verbose=1)
            graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output = trainer.take_embedding_all_data()

            embeddings = Embedding(graph_embeddings_array, node_embeddings_array, trainer.dataset, config_c)
            node_emb_pergraphclass = embeddings.get_emb_per_graph()
            emb_perclass0 = [n for n in node_emb_pergraphclass if n.graph_label == 0]
            emb_perclass1 = [n for n in node_emb_pergraphclass if n.graph_label == 1]

            str_filename = f"node_embeddings_{method}_{i}.png"
            plot_node_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"scatter_embeddings_degree_{method}_{i}.png"
            scatter_node_emb(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"graph_embedding_{method}_{i}.png"
            plot_graph_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)


def autoencoder_embedding(config_class, dataset_grafi_nx, dataset_labels, list_p):
    conf = config_class.conf
    num_last_neurons = conf['model']['neurons_per_layer'][-1]
    if conf['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"

    model = GAEGCNEncoder(neurons_per_layer=conf['model']['neurons_per_layer'], put_batchnorm=conf['model']['put_batchnorm'])
    model.to(device)
    print(model)

    #variational = False
    trainer = Trainer_Autoencoder(model, config_class)
    print("Loading Dataset...")
    trainer.load_dataset(dataset_grafi_nx)
    train_loss_list, test_loss_list = trainer.launch_training()

    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False)
    embeddings_array = trainer.get_embedding(all_data_loader)
    embeddings_array = np.array([emb.cpu().detach().numpy() for emb in embeddings_array])
    # embeddings_array = model(batch.x, batch.edge_index, batch.batch).cpu().detach().numpy()
    embeddings = Embedding(embeddings_array, trainer.dataset.labels, test_loss_list, conf)
    #embeddings.calc_distances()

    return embeddings, trainer



  # quì faccio la prova per la mpl animation.FuncAnimation
        # fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        # x = np.linspace(0, 4, 1000)
        # y = np.sin(2 * np.pi * (x - 0.02))
        # sc, = axes[0][0].plot(x, y, marker="o", ls="")
        # plot_funcs.sc1 = sc
        # anim = animation.FuncAnimation(
        #     fig,
        #     init_func=None,
        #     func=my_animated_figure,
        #     frames=lista,
        #     interval=100,
        #     save_count=len(lista))
        # #plt.show()
        # nomefile = self.create_filename(exp_config)
        # anim.save(nomefile + ".gif")
        # return

        # SEPSARO IN DUE FUNZIONI PRIMA TUTTO IL CALCOLO (SPERANDO DI MANTENERLO PARALLELO
        # E POI IL MAKE VIDEO