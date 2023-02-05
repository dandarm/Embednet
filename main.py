import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import time
from tqdm import tqdm
from pathlib import Path
from config_valid import Inits
from train import Trainer, GeneralDataset
from embedding import Embedding
from experiments import experiment_node_embedding, experiment_node_emb_cm
from experiments import Experiments
from config_valid import Config, TrainingMode

import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda')

from scipy import stats

import yaml

rootsave = Path("output_plots/")

def studio_init(config_file):
    methods = [Inits.xavier_uniform, Inits.kaiming_uniform, Inits.uniform, 'esn']
    experiment_node_emb_cm(config_file, methods, ripetiz=30)

def studio_init_weights():
    rootsave = Path("output_plots/")
    config_file = "configurations/classification_cm-diversi_init_weights.yml"
    xp = Experiments(config_file, rootsave)
    xp.diversi_init_weights_stesso_dataset(ripetizioni=5)

def many_classes():
    rootsave = Path("output_plots/")
    config_file = "configurations/classification_cm_manyclasses.yml"
    diz_trials = {'graph_dataset.Num_nodes': [[30] * 3, [30] * 5], # [300] * 25, [300] * 30],
                  # 'model.neurons_last_linear': [[20,20], [25,25], [30,30]],
                  'graph_dataset.list_exponents': [list(np.linspace(-3.5, -6.5, 3)),
                                                   list(np.linspace(-1.5, -4.5, 5)),
                                                   #list(np.linspace(-1.5, -4.5, 25)),
                                                   #list(np.linspace(-1.5, -4.5, 30))
                                                   ],
                  'model.neurons_last_linear': [[10, 10, 3], [10, 10, 5]]}#, [10, 10, 25], [10, 10, 30]]}

    xp = Experiments(config_file, diz_trials, rootsave)
    xp.diverse_classi_stesso_dataset()

def diversi_init_weights_diversi_dataset():
    config_file = "configurations/classification_cm_manyclasses.yml"
    diz_trials = {'graph_dataset.Num_nodes': [[350] * 6],
                  'graph_dataset.list_exponents': [list(np.round(np.linspace(-3.5, -6.5, 6), 2)),
                                                   list(np.round(np.linspace(-1.5, -4.5, 6), 2)),
                                                   ],
                  'model.neurons_last_linear': [[10, 10, 6]],
                  'model.init_weights': ['uniform'] * 5 + ['kaiming_uniform'] * 5}
    xp = Experiments(config_file, diz_trials, rootsave)
    xp.GSdiversi_init_weights_diversi_dataset(dataset_key1='graph_dataset.list_exponents')
    outfile = "output_data/df_configs_unif_kaimunif_init_weights_diverse_classi_gcn_freezed.pkl"
    df = xp.gc.config_dataframe
    df.to_pickle(outfile)

def studioERp():
    config_file = "configurations/ER_embed_perturb.yml"
    diz_trials = {'graph_dataset.list_p': [[0.05, 0.3], [0.01, 0.3], [0.15, 0.3], [0.2, 0.3], [0.25, 0.3]]}
    # 'model.init_weights': ['orthogonal']*5 + ['eye']*5}

    xp = Experiments(config_file, diz_trials, rootsave)
    xp.GS_same_weight_inits_different_datasets()

def simple():
    config_file = "configurations/1layer_freezed.yml"
    xp = Experiments(config_file, diz_trials=None, rootsave=rootsave)
    xp.just_train(parallel=True)
    embedding_class = xp.embedding()

if __name__ == "__main__":
    simple()

    #studio_embedding()
    #config_file = "configurations/classification_cm.yml"
    #experiment_node_embedding(config_file)

    #studio_init_weights()
    #many_classes()
    #diversi_init_weights_diversi_dataset()
    #studioERp()




