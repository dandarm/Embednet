import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import time
from tqdm import tqdm

from graph_generation import create_ER, dataset_nclass_ER
from models import GCN, GCNEmbed, GAEGCNEncoder
from train import Trainer
from embedding import Embedding
from experiments import experiment_embedding#, autoencoder_embedding

import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda')

from scipy import stats

import yaml

def studio_embedding():
    config = yaml.safe_load(open("configs.yml"))
    dataset_grafi_nx, dataset_labels, list_p= dataset_nclass_ER(config)
    embeddings = experiment_embedding(config, dataset_grafi_nx, dataset_labels, list_p)

    #if len(config['graph_dataset']['list_p']) == 2:
    if config['model']['neurons_per_layer'][-1] == 1:
        #plt.hist(embeddings.embeddings_array, bins=80);
        plt.figure(figsize=(18, 6))  # , dpi=60)
        for p in list_p:
            mask_int = np.argwhere(embeddings.embedding_labels == p).flatten()
            emb = embeddings.embeddings_array[mask_int].flatten()

            h, e = np.histogram(emb, bins=30, density=True)
            x = np.linspace(e.min(), e.max())
            plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')

            kde = stats.gaussian_kde(emb)
            plt.plot(x, kde.pdf(x), lw=5, label='KDE')

        plt.xlabel('p', fontsize=18)
        plt.xticks(fontsize=18)
        plt.show()
    else:
        plt.hist(embeddings.inter_dists, bins=80, alpha=0.8)
        plt.hist(embeddings.intra_dists, bins=80, alpha=0.8)

    plt.show()

if __name__ == "__main__":
    #studio_embedding()
    config = yaml.safe_load(open("configs.yml"))
    dataset_grafi_nx, dataset_labels, list_p = dataset_nclass_ER(config)
    #embeddings, trainer = autoencoder_embedding(config, dataset_grafi_nx, dataset_labels, list_p)
    embeddings, trainer = experiment_embedding(config, dataset_grafi_nx, dataset_labels, list_p)

    plt.figure(figsize=(18, 6))  # , dpi=60)
    for p in list_p:
        mask_int = np.argwhere(embeddings.embedding_labels == p).flatten()
        emb = embeddings.embeddings_array[mask_int].flatten()

        h, e = np.histogram(emb, bins=10, density=True)
        x = np.linspace(e.min(), e.max())
        plt.bar(e[:-1], h, width=np.diff(e), ec='k', align='edge', label='histogram')

        kde = stats.gaussian_kde(emb)
        plt.plot(x, kde.pdf(x), lw=5, label='KDE')

    plt.xlabel('p', fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()