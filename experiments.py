from models import GCN, GAEGCNEncoder, view_parameters, new_parameters, modify_parameters
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding, NodeEmbedding
from plot_model import plot_model
from config_valid import Config, TrainingMode
from graph_generation import GenerateGraph
from utils import array_wo_outliers
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

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
def init(config_file, parallel=True):
    config_c = Config(config_file)
    gg = GenerateGraph(config_c)
    print("Create dataset")
    dataset = gg.initialize_dataset(parallel=parallel)
    print("Initialize model")
    model = init_GCN(config_c, verbose=False)
    trainer = Trainer(model, config_c)
    return dataset, model, trainer, config_c

def init_model_dataset(config_file, verbose=False):
    dataset, model, trainer, config_class = init(config_file)
    conf = config_class.conf
    print("Loading Dataset...")
    trainer.load_dataset(dataset, percentage_train=conf['training']['percentage_train'], parallel=False)  # parallel false perché tanto con load_fromnetworkx non c'è nulla da fare...
    if verbose:
        batch = trainer.dataset.sample_dummy_data()
        plot = plot_model(trainer.model, batch)
        return model, trainer, plot
    return model, trainer

def train_take_embedding_alreadyinit(trainer, model=None, config_class=None, type_embedding='both', verbose=False):
    if config_class:
        trainer.reinit_conf(config_class)
    if model:
        trainer.reinit_model(model)

    train_loss_list, test_loss_list = trainer.launch_training(verbose)
    # get embedding of all graphs in dataset
    whole_data = trainer.dataset.dataset_pyg
    #print(whole_data[0].id)
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False)
    graph_embeddings_array, node_embeddings_array, node_embeddings_array_id = trainer.take_embedding(all_data_loader, type_embedding)

    return graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, trainer.model, test_loss_list, trainer

def train_take_embedding(config_file, parallel=True, type_embedding='both', verbose=False):
    model, trainer = init_model_dataset(config_file, verbose)
    return train_take_embedding_alreadyinit(trainer) #, type_embedding, verbose)

def experiment_graph_embedding(config_file):
    config_c = Config(config_file)
    graph_embeddings_array, _, _, model, test_loss_list, trainer = train_take_embedding(config_file, type_embedding='graph')
    embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
    num_emb_neurons = model.convs[-1].out_channels
    get_metrics(embeddings, num_emb_neurons)

def experiment_node_embedding(config_file):
    config_c = Config(config_file)
    graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, model, test_loss_list, trainer = train_take_embedding(config_file, type_embedding='both')
    emb_perclass0, emb_perclass1 = elaborate_embeddings(config_c, graph_embeddings_array, model, node_embeddings_array, node_embeddings_array_id, test_loss_list, trainer)

    for i in range(trainer.epochs):
        graph_embeddings_array = trainer.graph_embedding_per_epoch[i]
        node_embeddings_array = trainer.node_embedding_per_epoch[i]
        emb_perclass0, emb_perclass1 = elaborate_embeddings(config_c, graph_embeddings_array, model, node_embeddings_array, node_embeddings_array_id, test_loss_list, trainer)
        scatter_node_emb(emb_perclass0, emb_perclass1, trainer.last_accuracy)
        plot_graph_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy)


def elaborate_embeddings(config_c, graph_embeddings_array, model, node_embeddings_array, node_embeddings_array_id, test_loss_list, trainer):
    graph_embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
    node_embeddings = NodeEmbedding(node_embeddings_array, node_embeddings_array_id, trainer.dataset, test_loss_list, config_c)
    node_emb_pergraphclass = node_embeddings.get_emb_per_graph_class_cm(graph_embeddings_array)
    # graph embedding study
    #num_emb_neurons = model.convs[-1].out_channels
    #get_metrics(graph_embeddings, num_emb_neurons)
    # node embedding study

    # devo gestire il caso in cui i target no nsiano scalari
    distinct_labels = np.unique([n.graph_label for n in node_emb_pergraphclass], axis=0)
    emb_perclass0 = [n for n in node_emb_pergraphclass if (n.graph_label == distinct_labels[0]).all()]
    emb_perclass1 = [n for n in node_emb_pergraphclass if (n.graph_label == distinct_labels[1]).all()]
    return emb_perclass0, emb_perclass1


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
            graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, model, test_loss_list, trainer = train_take_embedding_alreadyinit(trainer, model)
            #graph_embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c)
            node_embeddings = NodeEmbedding(node_embeddings_array, node_embeddings_array_id, trainer.dataset, test_loss_list, config_c)
            node_emb_pergraphclass = node_embeddings.get_emb_per_graph_class_cm(graph_embeddings_array)
            emb_perclass0 = [n for n in node_emb_pergraphclass if n.graph_label == 0]
            emb_perclass1 = [n for n in node_emb_pergraphclass if n.graph_label == 1]

            str_filename = f"node_embeddings_{method}_{i}.png"
            plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"scatter_embeddings_degree_{method}_{i}.png"
            scatter_node_emb(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)

            str_filename = f"graph_embedding_{method}_{i}.png"
            plot_graph_emb_1D(emb_perclass0, emb_perclass1, trainer.last_accuracy, str_filename)


def run_with_weights(config_c, trainer, parameters):
    model = init_GCN(config_c, verbose=False)
    modify_parameters(model, parameters)
    trainer.reinit_conf(config_c)
    trainer.reinit_model(model)

    graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, model, test_loss_list, trainer = train_take_embedding_alreadyinit(trainer, model)
    return elaborate_embeddings(config_c, graph_embeddings_array, model, node_embeddings_array, node_embeddings_array_id, test_loss_list, trainer)








# region plots
def get_metrics(embeddings, num_emb_neurons):
    if num_emb_neurons == 1:
        correlazioni, error, embeddings_per_cluster = embeddings.calc_graph_emb_correlation()
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

def plot_graph_emb_1D(emb_perclass0, emb_perclass1, accuracy, str_filename=None, show=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    # plt.scatter(graph_embeddings.embeddings_array[:,0], graph_embeddings.embeddings_array[:,1], s=0.1, marker='.')
    bluehist = []
    redhist = []
    for emb_pergraph in emb_perclass0:
        bluehist.append(emb_pergraph.graph_embeddings_array)
    for emb_pergraph in emb_perclass1:
        redhist.append(emb_pergraph.graph_embeddings_array)
    bluehist = np.array(bluehist).flatten()
    redhist = np.array(redhist).flatten()
    ax.hist(bluehist, bins=30, color='blue');
    ax.hist(redhist, bins=30, color='red');
    ax.set_title(f'Final Test Acc. {round(accuracy,5)}')
    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    #else:
    #    plt.close()


def scatter_node_emb(emb_perclass0, emb_perclass1, accuracy, str_filename=None, show=True, epoch=None, ax=None, close=False):
    num_nodi_totali = len(emb_perclass0) * 2 * len(emb_perclass0[0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for emb_pergraph in emb_perclass0:
        ax.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color='blue', alpha=alpha_value)
    for emb_pergraph in emb_perclass1:
        ax.scatter(emb_pergraph.node_label, emb_pergraph.node_embedding_array, marker='.', color='red', alpha=alpha_value)
    ax.set_ylabel('Node Emb. values', fontsize=16);
    ax.set_xlabel('Degree sequence', fontsize=16);
    titolo = f'Final Test Acc. {round(accuracy,5)}'
    if epoch:
        titolo += f'\t epoch {epoch}'
    ax.set_title(titolo)
    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_node_emb_1D_perclass(emb_perclass0, emb_perclass1, accuracy, str_filename=None, show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    num_nodi_totali = len(emb_perclass0)*2 * len(emb_perclass0[0].node_embedding_array)
    alpha_value = min(1, 3000/num_nodi_totali)

    for emb_pergraph in emb_perclass0:
        nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
        degree_sorted = sorted(emb_pergraph.node_label, reverse=True)
        ax1.plot(nodeemb_sorted, '.', c='blue', alpha=alpha_value)
        ax2.plot(degree_sorted, '.', c='blue', alpha=alpha_value)
        # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)
    for emb_pergraph in emb_perclass1:
        nodeemb_sorted = sorted(emb_pergraph.node_embedding_array, reverse=True)
        degree_sorted = sorted(emb_pergraph.node_label, reverse=True)
        ax1.plot(nodeemb_sorted, '.', c='red', alpha=alpha_value)
        ax2.plot(degree_sorted, '.', c='red', alpha=alpha_value)
        # plt.ylim(ymax = max(nodeemb_sorted), ymin = 1)
    ax1.set_title(f'Final Test Acc. {round(accuracy, 5)}')

    ax1.set_xlabel('idx Node Emb.', fontsize=16);
    ax1.set_ylabel('Value Node Emb.', fontsize=16);
    ax2.set_xlabel('idx node', fontsize=16);
    ax2.set_ylabel('Node degree', fontsize=16);
    # plt.yscale('log')
    #ax1.xticks(fontsize=16);
    #plt.yticks(fontsize=16);

    if str_filename:
        plt.savefig(str_filename)
    if show:
        plt.show()
    else:
        plt.close()

def plot_data_degree_sequence(config_c, emb_perclass0, emb_perclass1):
    num_nodi_totali = len(emb_perclass0) * 2 * len(emb_perclass0[0].node_embedding_array)
    alpha_value = min(1, 3000 / num_nodi_totali)

    plt.figure(figsize=(12, 6))
    exps = config_c.conf['graph_dataset']['list_exponents']
    for emb_pergraph in emb_perclass0:
        counts = np.unique(emb_pergraph.node_label, return_counts=True)
        plt.loglog(*counts, c='blue', alpha=alpha_value, label=exps[emb_pergraph.graph_label])

    for emb_pergraph in emb_perclass1:
        counts = np.unique(emb_pergraph.node_label, return_counts=True)
        plt.loglog(*counts, c='red', alpha=alpha_value, label=exps[emb_pergraph.graph_label])

    # plt.legend(loc="upper left")
    plt.title(f'Esponenti: Blu: {exps[0]}, Rosso: {exps[1]}')
    plt.xlabel('Degrees', fontsize=16);
    plt.ylabel('Number of nodes', fontsize=16);
    # plt.gca().legend(('y0','y1'))
    plt.show()

# endregion

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
