import itertools
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import time
from tqdm import tqdm

from models import Inits
from train import Trainer, GeneralDataset
from embedding import Embedding, NodeEmbedding
from experiments import Experiments, experiment_node_embedding, all_seeds
from config_valid import Config, TrainingMode
from graph_generation import GenerateGraph
#from experiments import (train_take_embedding, plot_dim1, plot_dimN, plot_correlation_error, get_metrics,
#                         init_GCN, init_model_dataset, train_take_embedding_alreadyinit)
from plot_funcs import (plot_dim1, plot_dimN, plot_correlation_error, get_metrics, plot_node_emb_1D_perclass, scatter_node_emb,
                        plot_graph_emb_1D, plot_data_degree_sequence)

import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda')

from scipy import stats

import yaml

# def debug1():
#     config_file = "configurations/classification_nodeemb.yml"
#     config_c_reg = Config(config_file)
#     config_c_reg.conf
#     gg_reg = GenerateGraph(config_c_reg)
#     dataset_reg = gg_reg.initialize_dataset(parallel=False)
#     model, trainer = init_model_dataset(config_c_reg, dataset_reg, parallel=False, verbose=False)
#     #trainer.dataset.dataset_pyg[0].id
#     graph_embeddings_array, node_embeddings_array, node_embeddings_array_id, model, test_loss_list, trainer = train_take_embedding_alreadyinit(
#         trainer, model)
#
#     graph_embeddings = Embedding(graph_embeddings_array, trainer.dataset, test_loss_list, config_c_reg)
#     node_embeddings = NodeEmbedding(node_embeddings_array, node_embeddings_array_id, trainer.dataset, test_loss_list, config_c_reg)
#
#     node_emb_pergraphclass = node_embeddings.get_emb_per_graph_class_cm(graph_embeddings_array)
#
#     emb_perclass0 = [n for n in node_emb_pergraphclass if n.graph_label == 0]
#     emb_perclass1 = [n for n in node_emb_pergraphclass if n.graph_label == 1]

def debug2():
    # config_file = "configurations/classification_cm.yml"
    # config_c = Config(config_file)
    # gg = GenerateGraph(config_c)  # xp.trainer.config_class)
    # gg.initialize_dataset()
    # print("fine")

    rootsave = Path("output_plots/")
    config_file = "configurations/classification_cm_manyclasses.yml"
    xp = Experiments(config_file, rootsave)
    xp.just_train()
    embs_by_class = xp.embedding()

    plot_data_degree_sequence(xp.trainer.config_class, embs_by_class)
    i, j = 15, 3
    plt.scatter(embs_by_class[i][j].node_label, embs_by_class[i][j].node_embedding_array)
    print(embs_by_class[i][j].graph_label)

if __name__ == "__main__":
    #debug1()
    debug2()



