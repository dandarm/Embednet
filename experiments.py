from models import GCN, GAEGCNEncoder
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding, NodeEmbedding
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



def train_take_embedding(config_class, dataset, type_embedding='both', verbose=False):
    model, trainer = init_model_dataset(config_class, dataset, verbose)
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
    graph_embeddings_array, node_embeddings_array = trainer.take_embedding(all_data_loader, type_embedding)

    if type_embedding == 'graph':
        graph_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in graph_embeddings_array])
        return graph_embeddings_array, None, trainer.model, test_loss_list, trainer
    elif type_embedding == 'node':
        node_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in node_embeddings_array])
        return None, node_embeddings_array, trainer.model, test_loss_list, trainer
    elif type_embedding == 'both':
        graph_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in graph_embeddings_array])
        node_embeddings_array = np.array([emb.cpu().detach().numpy() for emb in node_embeddings_array])
        return graph_embeddings_array, node_embeddings_array, trainer.model, test_loss_list, trainer

def init_model_dataset(config_class, dataset, verbose):
    conf = config_class.conf
    print("Initialize model")
    model = init_GCN(config_class, verbose)
    trainer = Trainer(model, config_class)
    print("Loading Dataset...")
    trainer.load_dataset(dataset, percentage_train=conf['training']['percentage_train'])
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
        plt.figure(figsize=(18, 6))  # , dpi=60)

        for emb in embeddings_per_cluster:
            h, e = np.histogram(emb, bins=bins, density=True)
            x = np.linspace(e.min(), e.max())
            plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')
            if want_kde:
                kde = stats.gaussian_kde(emb)
                plt.plot(x, kde.pdf(x), lw=5, label='KDE')

        plt.xlabel('p', fontsize=18)
        plt.xticks(fontsize=18)
        plt.show()

def plot_correlation_error(embeddings, correlazioni, error):
    plt.figure(figsize=(12, 6))  # , dpi=60)
    plt.scatter(embeddings.embedding_labels, embeddings.embeddings_array.flatten())  # , s=area, c=colors, alpha=0.5)
    # correlazione tra target e prediction
    #correlaz = np.corrcoef(embeddings.embeddings_array.flatten(), embeddings.embedding_labels)[0, 1]
    #error = np.sqrt(np.sum((embeddings.embeddings_array.flatten() - embeddings.embedding_labels) ** 2))
    if isinstance(correlazioni, list):
        cs = [round(c, 5) for c in correlazioni]
    else:
        cs = round(correlazioni, 5)
    plt.title(f"Corr = {cs}   -   Error  {round(error, 5)}")
    plt.show()

def plot_dimN(embeddings):
    plt.figure(figsize=(14, 6))
    if len(embeddings.inter_dists) > 0 and embeddings.intra_dists is not None:
        plt.hist(embeddings.inter_dists, bins=80, alpha=0.8)
        plt.hist(embeddings.intra_dists, bins=80, alpha=0.8)
    else:
        all_distances = [e[0] for e in embeddings.distances]
        plt.hist(all_distances, bins=80)
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
