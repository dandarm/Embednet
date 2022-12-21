import imageio
from io import BytesIO
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
#torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from multiprocessing import Pool
import torch_geometric.transforms as T

from models import GAEGCNEncoder, view_parameters, new_parameters, modify_parameters, new_parameters_linears, modify_parameters_linear
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding
from plot_funcs import scatter_node_emb, plot_graph_emb_1D, plot_node_emb_1D_perclass, save_ffmpeg
from config_valid import Config
from GridConfigurations import GridConfigurations
from utils import array_wo_outliers

# per usare il trainer e il config nei processi paralleli
graph_embedding_per_epoch = []
node_embedding_per_epoch = []
dataset = None
loss_list = []
exp_config = None
output_per_epoch = []
accuracy_list = []
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

        # risultati
        #self.graph_embedding_per_epoch = []
        #self.node_embedding_per_epoch = []

    def stesso_init_diversi_dataset(self):
        all_seeds()

        self.gc = GridConfigurations(self.config_class, self.diz_trials)
        self.gc.make_configs()
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

            self.trainer.init_dataset()
            self.trainer.load_dataset(self.trainer.gg.dataset)

            self.trainer.launch_training()

    def diversi_init_weights_stesso_dataset(self):  #, metodi, ripetizioni):
        global exp_config
        global exp_trainer

        all_seeds()

        config_class = Config(self.config_file)
        trainer = Trainer(config_class)
        exp_config = config_class
        exp_trainer = trainer


        #self.diz_trials = {'model.init_weights': metodi * ripetizioni}
        self.gc = GridConfigurations(config_class, self.diz_trials)
        self.gc.make_configs()
        self.GS_different_weight_inits(trainer, train_with_same_dataset=True, test_same_training=False)

    def GS_different_weight_inits(self, trainer, train_with_same_dataset=False, test_same_training=False):
        global exp_config
        global exp_trainer

        if train_with_same_dataset:
            print("Carico il dataset e lo tengo invariato per tutti i trial")
            trainer.init_dataset()
            trainer.load_dataset(trainer.gg.dataset)

        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = trainer.init_GCN()
        saved_initial_weights_lin = new_parameters_linears(modello_base)
        if test_same_training:
            saved_initial_weights_gcn = new_parameters(modello_base)

        # ordino tutte le inizializzazioni dello stesso tipo, per dover gestire bene il reset dei seed
        # gc.configs = sorted(gc.configs, key=lambda c: c.conf['model']['init_weights'])
        # meglio il seguente metodo
        metodi_distinti = set([c.conf['model']['init_weights'] for c in self.gc.configs])

        k = 0
        for m in metodi_distinti:
            cs = [(i, c) for i, c in self.gc.ordinal_configs if c.conf['model']['init_weights'] == m]
            all_seeds()
            # diversi training dell'unico parametro ripescato random dalla stessa distribuzione
            for ord_config in cs:
                i, c = ord_config
                print(f'Run {k + 1}')

                trainer.reinit_conf(c)
                if not train_with_same_dataset:
                    trainer.init_dataset()
                    trainer.load_dataset(trainer.gg.dataset)
                if test_same_training:
                    model = trainer.init_GCN(saved_initial_weights_gcn, saved_initial_weights_lin)
                else:
                    # questa chiamata non deve avere i seed resettati altrimenti otterrò gli stessi pesi e lo stesso training
                    init_weight_parameters = new_parameters(modello_base, method=c.init_weights_mode)
                    model = trainer.init_GCN(init_weight_parameters, saved_initial_weights_lin)


                trainer.load_model(model)

                trainer.launch_training()

                print("calcolo i risultati di interesse")
                exp_trainer = trainer
                exp_config = c
                avg_corr_classes, avg_tau_classes = self.get_corrs_training()
                avg_corr_classes = np.array(avg_corr_classes).T
                avg_tau_classes = np.array(avg_tau_classes).T

                fill_df_with_results(self.gc.config_dataframe, i, avg_corr_classes, avg_tau_classes, trainer.test_loss_list, trainer.accuracy_list)
                k+=1

    def diverse_classi_stesso_dataset(self, parallel_take_result):  #, metodi, ripetizioni):
        global exp_config
        global exp_trainer

        all_seeds()

        config_class = Config(self.config_file)
        self.trainer = Trainer(config_class)
        exp_config = config_class
        exp_trainer = self.trainer


        #self.diz_trials = {'model.init_weights': metodi * ripetizioni}
        self.gc = GridConfigurations(config_class, self.diz_trials)
        self.gc.make_configs()
        self.GS_different_classes(self.trainer, test_same_training=False)

    def GS_different_classes(self, trainer, test_same_training=False, parallel_take_result=False):
        global graph_embedding_per_epoch
        global node_embedding_per_epoch
        global dataset
        global loss_list
        global exp_config

        # modello di base per avere l'architettura dei parametri da impostare
        modello_base = trainer.init_GCN()
        #saved_initial_weights_lin = new_parameters_linears(modello_base)
        saved_initial_weights_gcn = new_parameters(modello_base, method=trainer.config_class.init_weights_mode)
        if test_same_training:
            trainer.init_dataset()
            trainer.load_dataset(trainer.gg.dataset)

        k = 0
        for c in self.gc.configs:
            print(f'Run {k + 1}')
            trainer.reinit_conf(c)
            if not test_same_training:
                trainer.init_dataset()
                trainer.load_dataset(trainer.gg.dataset)
            model = trainer.init_GCN(saved_initial_weights_gcn) #, saved_initial_weights_lin)
            trainer.load_model(model)

            trainer.launch_training()
            embedding_class = self.embedding()

            print("calcolo i risultati di interesse")
            #exp_trainer = trainer
            #exp_config = c
            if parallel_take_result:
                graph_embedding_per_epoch = trainer.graph_embedding_per_epoch
                node_embedding_per_epoch = trainer.node_embedding_per_epoch
                dataset = self.trainer.dataset
                loss_list = self.trainer.test_loss_list
                exp_config = self.trainer.config_class

            avg_corr_classes, avg_tau_classes = self.get_corrs_training(parallel=parallel_take_result)
            avg_corr_classes = np.array(avg_corr_classes).T
            avg_tau_classes = np.array(avg_tau_classes).T

            fill_df_with_results(self.gc.config_dataframe, k, avg_corr_classes, avg_tau_classes, trainer.test_loss_list, trainer.accuracy_list, embedding_class)
            #fill_embedding_df(self.gc.embedding_dataframe, trainer, k)
            k += 1

    def just_train(self):
        self.trainer = Trainer(self.config_class)
        self.trainer.init_all()
        self.trainer.launch_training()
    def embedding(self):
        if self.config_class.conf['training']['save_best_model']:
            self.trainer.model = self.trainer.best_model
        graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, final_output = self.trainer.take_embedding_all_data()
        embedding_class = self.elaborate_embedding(graph_embeddings_array, node_embeddings_array, node_embeddings_array_id)

        return embedding_class

    def elaborate_embedding(self, graph_embeddings_array, node_embeddings_array, node_embeddings_array_id):
        embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, self.trainer.dataset, self.trainer.config_class)
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
        for classe in node_embedding_class.node_emb_perclass:
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
        if parallel:
            with Pool(processes=32) as pool:
                res = pool.map(parallel_take_corr, range(self.trainer.last_epoch + 1))
                avg_corr_classes = [r[0] for r in res]
                avg_tau_classes = [r[1] for r in res]
        else:
            avg_corr_classes = []
            avg_tau_classes = []
            for e in range(exp_trainer.last_epoch + 1):
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

    def make_video(self, skip, fromfiles=True, isgif=False, both=True, custom_list=None):
        global output_per_epoch
        global loss_list
        global accuracy_list
        output_per_epoch = self.trainer.output_per_epoch
        loss_list = self.trainer.test_loss_list
        accuracy_list = self.trainer.accuracy_list

        if both:
            isgif = True
            isvideo = True
            fromfiles = True
        if custom_list:
            lista = custom_list
        else:
            lista = range(0, self.trainer.last_epoch, skip)
        if fromfiles:
            self.parallel_save_many_images_embedding(lista)
            files = [f"scatter_epoch{i}.png" for i in lista]
        else:
            pictures = self.parallel_many_images_embedding(lista)

        numnodi = exp_config.conf['graph_dataset']['Num_nodes']
        if isinstance(numnodi, list):
            numnodi = numnodi[0]
        numgrafi = exp_config.conf['graph_dataset']['Num_grafi_per_tipo'] * 2
        exps = exp_config.conf['graph_dataset']['list_exponents']
        if isinstance(exps, list):
            exps = f"{len(exps)}"
        percentuale_train = exp_config.conf['training']['percentage_train']
        layers = exp_config.conf['model']['GCNneurons_per_layer']
        denso = exp_config.conf['model']['last_layer_dense']
        modo = exp_config.conf['training']['mode']
        freezed = exp_config.conf['model']['freezeGCNlayers']
        nomefile = f"scatter_exp{exps}_nodi{numnodi}_grafi{numgrafi}_percent{percentuale_train}_{modo}_layers{layers}_finaldense{denso}_freezed{freezed}"
        nomefile = nomefile.replace(', ', '_')
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


def parallel_take_corr(epoca):
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
    graph_embeddings_array = graph_embedding_per_epoch[i]
    node_embeddings_array = node_embedding_per_epoch[i]
    embedding_class = Embedding(graph_embeddings_array, node_embeddings_array, dataset, exp_config)
    embedding_class.get_emb_per_graph()  # riempie node_emb_pergraph
    embedding_class.separate_embedding_by_classes()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    scatter_node_emb(embedding_class.node_emb_perclass, exp_config, ax=axes[0][0], show=False, close=True, sequential_colors=True)
    plot_graph_emb_1D(embedding_class.node_emb_perclass, exp_config, ax=axes[0][1], show=False, close=True, sequential_colors=True)
    axes[0][2].hist(np.array(output_per_epoch[i]).flatten(), bins=50);
    axes[1][0].plot(loss_list)
    axes[1][0].plot(i, loss_list[i], 'ro')
    axes[1][1].plot(accuracy_list)
    axes[1][1].plot(i, accuracy_list[i], 'ro')
    fig.suptitle(f"Network dynamics")
    return fig




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







def fill_df_with_results(df, i, avg_corr_classes, avg_tau_classes, test_loss_list, accuracy_list, node_embedding_class):
    df.loc[i, ('risultati', 'data')] = pd.Timestamp.now()
    df.at[i, ('risultati', 'test_loss')] = test_loss_list
    df.at[i, ('risultati', 'test_accuracy')] = accuracy_list
    df.at[i, ('risultati', 'correlation_allclasses')] = avg_corr_classes.tolist()
    df.at[i, ('risultati', 'tau_allclasses')] = avg_tau_classes.tolist()
    df.at[i, ('risultati', 'node_embedding_class')] = node_embedding_class

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
            plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

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
