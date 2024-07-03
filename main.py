import itertools
from scipy import stats
import yaml
import sys
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, SimpleQueue
import time
#from tqdm import tqdm
from pathlib import Path
from config_valid import Inits
from train import Trainer, GeneralDataset
from embedding import Embedding
import experiments
from experiments import Experiments
from graph_generation import GenerateGraph
from config_valid import Config, TrainingMode
from dictionary_of_trials import get_diz_trials, modify_some_trials, load_trials_edits

import torch
from torch_geometric.loader import DataLoader

from prove import parallel_coord1

device = torch.device('cuda')
from motif_count import init_worker, get_valid_p, build_permutation_complete_graph

import asyncio

import networkx.algorithms.isomorphism as iso

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 400)

sys.path.append("../")
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

def grid_same_init_W(argv):
    c, diz_trials, path_to_save = load_trials(argv)

    is_verbose = argv[2] == "True"
    print(f"verbose: {is_verbose}")
    xp = Experiments(diz_trials=diz_trials,
                     rootsave=rootsave / path_to_save, config_class=c,
                     reset_all_seeds=False, verbose=is_verbose)
    xp.GS_same_init_W(verbose=is_verbose)

def simple_grid_search(argv):
    c, diz_trials, path_to_save = load_trials(argv)

    is_verbose = argv[2] == "verbose=True"
    print(f"verbose: {is_verbose}")
    xp = Experiments(diz_trials=diz_trials,
                     rootsave=rootsave / path_to_save, config_class=c,
                     reset_all_seeds=False, verbose=is_verbose)
    xp.GS_simple_experiments(verbose=is_verbose)


def load_trials(argv):
    c, diz_trials = get_diz_trials("configurations/Final1.yml")
    try:
        modification_file = argv[1]
        edits = load_trials_edits(modification_file)
    except Exception as e:
        print(e)
        print("Nessuna modifica ai trials")
        edits = None
    path_to_save = Path(str(argv[0]))
    # print(f"Salvo plot in {rootsave / path_to_save} ")
    if edits is not None:
        diz_trials = modify_some_trials(diz_trials, **edits)
    return c, diz_trials, path_to_save


def run_many_same_training(argv):
    c, diz_trials, path_to_save = load_trials(argv)
    is_verbose = argv[2] == "True"

    xp = Experiments(diz_trials=diz_trials,
                     rootsave=rootsave / path_to_save, config_class=c,
                     reset_all_seeds=False, verbose=is_verbose)
    reps = c.conf['training'].get('repetitions')
    if reps > 0:
        xp.many_same_training(reps)

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


def count_non_iso_motif_up_to_n(n):
    n = 4
    permutations = build_permutation_complete_graph(n)
    valid_permutations = shared_array((1, len(permutations)))

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




# def run_grid_w_gif(xp):
#     nomifilesgif = []
#     k = 0
#     for c in xp.gc.configs:
#         print(f'Run {k + 1}\t\t exp name: {c.unique_train_name}')
#         # all_seeds()
#         xp.trainer.reinit_conf(c)
#         xp.just_train()
#         #embedding_class = xp.embedding()
#         #num_emb_neurons = xp.trainer.embedding_dimension
#         #trainmode = xp.trainer.config_class.modo
#         #embedding_class.get_metrics(num_emb_neurons)  #, trainmode)
#
#         nomefile = xp.make_video(skip=1, fromfiles=True, seq_colors=False)
#         nomifilesgif.append(nomefile)
#         k += 1
#     return nomifilesgif


if __name__ == "__main__":
    #simple()

    #array = np.array([np.random.normal(mu, 0.5, 16) for mu in np.arange(-3, 3, 0.01)])
    #print(array.shape)
    #parallel_coord1(array, "Graph embedding")

    #studio_embedding()
    #config_file = "configurations/classification_cm.yml"
    #experiment_node_embedding(config_file)

    #studio_init_weights()
    #many_classes()
    #diversi_init_weights_diversi_dataset()
    #studioERp()

    #count_non_iso_motif_up_to_n(4)

    print(sys.argv)
    if 'init_w=True' in sys.argv:
        grid_same_init_W(sys.argv[1:])

    else:
        simple_grid_search(sys.argv[1:])

    #run_many_same_training(sys.argv[1:])

    # from NEMtropy import UndirectedGraph, network_functions
    # from graph_generation import rndm
    #
    # Num_nodes = 350
    # exponent = -2.1
    # s = rndm(3, Num_nodes, exponent, Num_nodes)
    # x = np.array(s, int)
    # ps = x * x[:, np.newaxis]
    # p_ij = ps / (1 + ps)
    # graph = UndirectedGraph(p_ij)
    # graph.solve_tool(model="cm_exp",
    #                  method="newton",
    #                  initial_guess="random")
    # res = graph.ensemble_sampler(10, cpu_n=None, output_dir="None")
    # #network_functions.build_graph_from_edgelist(res[0], is_directed=False)
    # network_functions.build_adjacency_from_edgelist(res[0], is_directed=False)





