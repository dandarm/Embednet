from models import GCN, GAEGCNEncoder, view_parameters, new_parameters, modify_parameters
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding, NodeEmbedding
from plot_model import plot_model
from config_valid import Config, TrainingMode
from graph_generation import GenerateGraph

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


def experiment_graph_embedding(config_file):
    config_c = Config(config_file)
    gg = GenerateGraph(config_c)
    dataset = gg.initialize_dataset()
    graph_embeddings_array, _, model, test_loss_list, trainer = train_take_embedding(config_c, dataset, type_embedding='graph')
    embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
    num_emb_neurons = model.convs[-1].out_channels
    get_metrics(embeddings, num_emb_neurons)

def experiment_node_embedding(config_file):
    config_c = Config(config_file)
    gg = GenerateGraph(config_c)
    dataset = gg.initialize_dataset()
    graph_embeddings_array, node_embeddings_array, model, test_loss_list, trainer = train_take_embedding(config_c, dataset, type_embedding='both')
    graph_embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
    node_embeddings = NodeEmbedding(node_embeddings_array, trainer.dataset, test_loss_list, config_c)
    num_emb_neurons = model.convs[-1].out_channels
    get_metrics(graph_embeddings, num_emb_neurons)

def experiment_node_emb_cm(config_file, methods, ripetiz=30):
    config_c = Config(config_file)
    gg = GenerateGraph(config_c)
    dataset = gg.initialize_dataset(parallel=True)
    model, trainer = init_model_dataset(config_c, dataset, parallel=False, verbose=False)
    for method in methods:
        for i in range(ripetiz):
            model = init_GCN(config_c, verbose=False)
            new_par = new_parameters(model, method=method)
            modify_parameters(model, new_par)
            graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, model, test_loss_list, trainer = train_take_embedding_alreadyinit(
                trainer, model)
            #graph_embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
            node_embeddings = NodeEmbedding(node_embeddings_array, node_embeddings_array_id, trainer.dataset,
                                            test_loss_list, config_c)
            node_emb_pergraphclass = node_embeddings.get_emb_per_graph_class_cm(graph_embeddings_array)
            emb_perclass0 = [n for n in node_emb_pergraphclass if n.graph_label == 0]
            emb_perclass1 = [n for n in node_emb_pergraphclass if n.graph_label == 1]

            str_filename = f"node_embeddings_{method}_{i}.png"
            plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"scatter_embeddings_degree_{method}_{i}.png"
            scatter_node_emb(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"graph_embedding_{method}_{i}.png"
            plot_graph_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)


def plot_graph_emb_1D(emb_perclass0, emb_perclass1, last_accuracy, str_filename=None):
    plt.figure(figsize=(12, 6))
    # plt.scatter(graph_embeddings.embeddings_array[:,0], graph_embeddings.embeddings_array[:,1], s=0.1, marker='.')
    bluehist = []
    redhist = []
    for emb_pergraph in emb_perclass0:
        bluehist.append(emb_pergraph.graph_embeddings_array)
    for emb_pergraph in emb_perclass1:
        redhist.append(emb_pergraph.graph_embeddings_array)
    bluehist = np.array(bluehist).flatten()
    redhist = np.array(redhist).flatten()
    plt.hist(bluehist, bins=30, color='blue');
    plt.hist(redhist, bins=30, color='red');
    plt.title(f'Final Test Acc. {last_accuracy}')
    if str_filename:
        plt.savefig(str_filename)


def scatter_node_emb(emb_perclass0, emb_perclass1, last_accuracy, str_filename=None):
    plt.figure(figsize=(12, 6))
    for emb_pergraph in emb_perclass0:
        plt.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color='blue', alpha=0.1)
    for emb_pergraph in emb_perclass1:
        plt.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color='red', alpha=0.01)
    plt.ylabel('Node Emb. values', fontsize=16);
    plt.xlabel('Degree sequence', fontsize=16);
    plt.title(f'Final Test Acc. {last_accuracy}')
    if str_filename:
        plt.savefig(str_filename)


def plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, last_accuracy, str_filename=None):
    plt.figure(figsize=(12, 6))
    for emb_pergraph in emb_perclass0:
        nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
        plt.plot(nodeemb_sorted, '.', c='blue', alpha=0.005)
        # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)
    for emb_pergraph in emb_perclass1:
        nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
        plt.plot(nodeemb_sorted, '.', c='red', alpha=0.005)
        # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)
    plt.title(f'Final Test Acc. {last_accuracy}')
    plt.xlabel('idx Node Emb.', fontsize=16);
    plt.ylabel('Value Node Emb.', fontsize=16);
    # plt.yscale('log')
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    if str_filename:
        plt.savefig(str_filename)


def train_take_embedding(config_class, dataset, type_embedding='both', verbose=False):
    model, trainer = init_model_dataset(config_class, dataset, False, verbose)
    return train_take_embedding_alreadyinit(trainer, type_embedding=type_embedding, verbose=verbose)

def train_take_embedding_alreadyinit(trainer, model=None, config_class=None, type_embedding='both', verbose=False):
    if config_class:
        trainer.reinit_conf(config_class)
    if model:
        trainer.reinit_model(model)

    train_loss_list, test_loss_list = trainer.launch_training(verbose)
    # get embedding of all graphs in dataset
    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False)
    graph_embeddings_array, node_embeddings_array, node_embeddings_array_id = trainer.take_embedding(all_data_loader, type_embedding)

    if type_embedding == 'graph':
        graph_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in graph_embeddings_array])
        return graph_embeddings_array, None, trainer.model, test_loss_list, trainer
    elif type_embedding == 'node':
        node_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in node_embeddings_array])
        return None, node_embeddings_array, trainer.model, test_loss_list, trainer
    elif type_embedding == 'both':
        graph_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in graph_embeddings_array])
        node_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in node_embeddings_array])
        return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, trainer.model, test_loss_list, trainer

def init_model_dataset(config_class, dataset, parallel, verbose):
    conf = config_class.conf
    print("Initialize model")
    model = init_GCN(config_class, verbose)
    trainer = Trainer(model, config_class)
    print("Loading Dataset...")
    trainer.load_dataset(dataset, percentage_train=conf['training']['percentage_train'], parallel=parallel)
    if verbose:
        batch = trainer.dataset.sample_dummy_data()
        plot = plot_model(trainer.model, batch)
        return model, trainer, plot
    return model, trainer

def init_GCN(config_class, verbose):
    conf = config_class.conf
    if conf['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"
    model = GCN(config_class)
    model.to(device)
    if verbose:
        print(model)
    return model

def get_metrics(embeddings, num_emb_neurons):
    if num_emb_neurons == 1:
        correlazioni, error, embeddings_per_cluster = embeddings.calc_correlation()
        # embeddings_per_cluster solo per distribuzione discreta
        plot_dim1(embeddings_per_cluster)
        plot_correlation_error(embeddings, correlazioni, error)
    else:
        embeddings.calc_distances()
        plot_dimN(embeddings)

def plot_dim1(embeddings_per_cluster, bins=10, want_kde=True):
    if embeddings_per_cluster:
        #plt.figure(figsize=(18, 6))  # , dpi=60)

        for emb in embeddings_per_cluster:
            h, e = np.histogram(emb, bins=bins, density=True)
            x = np.linspace(e.min(), e.max())
            plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')
            if want_kde:
                kde = stats.gaussian_kde(emb)
                plt.plot(x, kde.pdf(x), lw=5, label='KDE')

        plt.xlabel('p', fontsize=18)
        plt.xticks(fontsize=18)
        plt.savefig("dist_1D.png", dpi=72)
        plt.show()

def plot_correlation_error(embeddings, correlazioni, error):
    #plt.figure(figsize=(12, 6))  # , dpi=60)
    plt.scatter(embeddings.embedding_labels, embeddings.embeddings_array.flatten())  # , s=area, c=colors, alpha=0.5)
    # correlazione tra target e prediction
    #correlaz = np.corrcoef(embeddings.embeddings_array.flatten(), embeddings.embedding_labels)[0, 1]
    #error = np.sqrt(np.sum((embeddings.embeddings_array.flatten() - embeddings.embedding_labels) ** 2))
    if isinstance(correlazioni, list):
        cs = [round(c, 5) for c in correlazioni]
    else:
        cs = round(correlazioni, 5)
    plt.title(f"Corr = {cs}   -   Error  {round(error, 5)}")
    plt.savefig("correl_error.png", dpi=72)
    plt.show()

def plot_dimN(embeddings):
    #plt.figure(figsize=(14, 6))
    if len(embeddings.inter_dists) > 0 and embeddings.intra_dists is not None:
        plt.hist(embeddings.inter_dists, bins=80, alpha=0.8)
        plt.hist(embeddings.intra_dists, bins=80, alpha=0.8)
    else:
        all_distances = [e[0] for e in embeddings.distances]
        plt.hist(all_distances, bins=80)
    plt.savefig("dist_vectors.png", dpi=72)
    plt.show()

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
    trainer.load_dataset(dataset_grafi_nx, dataset_labels, percentage_train=conf['training']['percentage_train'])
    train_loss_list, test_loss_list = trainer.launch_training()

    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False)
    embeddings_array = trainer.take_embedding(all_data_loader)
    embeddings_array = np.array([emb.cpu().detach().numpy() for emb in embeddings_array])
    # embeddings_array = model(batch.x, batch.edge_index, batch.batch).cpu().detach().numpy()
    embeddings = Embedding(embeddings_array, trainer.dataset.labels, test_loss_list, conf)
    #embeddings.calc_distances()

    return embeddings, trainer
