import itertools
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
import time
from tqdm import tqdm

from models import GCN, GAEGCNEncoder
from train import Trainer, GeneralDataset
from embedding import Embedding
from experiments import experiment_graph_embedding
from config_valid import Config, TrainingMode

import torch
from torch_geometric.loader import DataLoader
device = torch.device('cuda')

from scipy import stats

import yaml

def studio_embedding():
    config_file = "configs.yml"
    experiment_graph_embedding(config_file)



if __name__ == "__main__":
    studio_embedding()



