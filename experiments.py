from models import GCN, GAEGCNEncoder
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


def experiment_embedding(config_class, dataset_grafi_nx, dataset_labels, list_p, continuous_p):
    conf = config_class.conf
    layer = conf['model']['neurons_per_layer']
    num_last_neurons = layer[-1]
    if conf['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"

    model = GCN(config_class)
    model.to(device)
    print(model)

    trainer = Trainer(model, config_class)
    print("Loading Dataset...")
    trainer.load_dataset(dataset_grafi_nx, dataset_labels, percentage_train=conf['training']['percentage_train'])
    train_loss_list, test_loss_list = trainer.launch_training()

    # get embedding of all graphs in dataset
    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False) #batch_size=trainer.dataset.len_data
    embeddings_array = trainer.take_embedding(all_data_loader)
    embeddings_array = np.array([emb.cpu().detach().numpy() for emb in embeddings_array])

    # calcola le metriche di interesse
    embeddings = Embedding(embeddings_array, trainer.dataset.dataset, trainer.dataset.labels, list_p, test_loss_list, conf, continuous_p)
    embeddings.calc_distances()

    return embeddings, trainer, test_loss_list


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
    embeddings = Embedding(embeddings_array, trainer.dataset.labels, list_p, test_loss_list, conf)
    #embeddings.calc_distances()

    return embeddings, trainer
