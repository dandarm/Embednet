import itertools
from scipy import stats
import yaml
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, SimpleQueue
import time
from tqdm import tqdm
from pathlib import Path
from config_valid import Inits
from train import Trainer, GeneralDataset
from embedding import Embedding
from experiments import Experiments
from graph_generation import GenerateGraph
from config_valid import Config, TrainingMode

import torch
from torch_geometric.loader import DataLoader

device = torch.device('cuda')
from motif_count import init_worker, get_valid_p, build_permutation_complete_graph

import asyncio

import networkx.algorithms.isomorphism as iso

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 400)

rootsave = Path("output_plots/")

# region exps

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

def simple_grid_search():
    config_file = "configurations/Final1.yml"
    #c = Config(config_file)
    xp = Experiments(config_file, diz_trials=None, rootsave=rootsave)  # , config_class=c)
    xp.trainer.gg = GenerateGraph(xp.trainer.config_class)
    xp.trainer.gg.initialize_dataset(parallel=True)

# endregion


def shared_array(shape):
    """
    Form a shared memory numpy array.
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """
    shared_array_base = mp.Array(ctypes.c_double, shape[0] * shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array

n = 4
permutations = build_permutation_complete_graph(n)
valid_permutations = shared_array((1, len(permutations)))
def count_non_iso_motif_up_to_n(n):
    shared_queue = SimpleQueue()

    with Pool(processes=16, initializer=init_worker, initargs=(shared_queue,)) as pool:
        # issue tasks into the process pool
        _ = pool.map_async(get_valid_p, permutations)
        # read results from the queue as they become available
        for p in permutations:
            print(p)
            result = shared_queue.get()
            print(f'Got {result}', flush=True)

    #print("\nStampa delle permutazioni uniche")
    # for r in result:
    #    print(r)

    pool = Pool(16)
    results = []
    for perm in permutations:
        result = pool.apply_async(get_valid_p, args=(g, perm))
        results.append(result.get())

    pool.close()
    pool.join()

def autoencoder():
    config_file = "configurations/Final1.yml"
    num_nodi = 117
    c = Config(config_file)
    # c.conf['graph_dataset']['Num_nodes'] = [num_nodi]
    # c.conf['graph_dataset']['list_exponents'] = [-2.5]
    c.conf['model']['autoencoder'] = False
    c.conf['model']['autoencoder_confmodel'] = True
    c.conf['model']['autoencoder_graph_ae'] = False
    diz_trials = {'graph_dataset.ERmodel': [False],
                  'graph_dataset.confmodel': [True, False],
                  'graph_dataset.sbm': [False],
                  'graph_dataset.real_dataset': [False],
                  'graph_dataset.Num_nodes': [[num_nodi]*5, [num_nodi]*3, [[num_nodi, int(num_nodi/2)]]*3],  # per lo SBM: num nodi * num classi * num comunità
                  'model.GCNneurons_per_layer': [  # [1, 32, 16, len(c.conf['graph_dataset']['list_exponents'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['list_p'])],
                      # [1, 32, 16, len(c.conf['graph_dataset']['community_probs'])],
                      [1, 32, 32, 32, 1]
                  ],
                  # 'model.init_weights': ['xavier_normal'],# 'eye'],
                  'model.freezeGCNlayers': [False],
                  'model.last_layer_dense': [False],
                  }

    xp = Experiments(diz_trials=diz_trials, rootsave=rootsave, config_class=c, reset_all_seeds=False, verbose=False)
    xp.GS_simple_experiments()


if __name__ == "__main__":
    #simple()

    #studio_embedding()
    #config_file = "configurations/classification_cm.yml"
    #experiment_node_embedding(config_file)

    #studio_init_weights()
    #many_classes()
    #diversi_init_weights_diversi_dataset()
    #studioERp()

    #simple_grid_search()

    #count_non_iso_motif_up_to_n(4)

    autoencoder()



