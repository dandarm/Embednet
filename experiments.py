from models import GCN, GAEGCNEncoder
from train import Trainer
from train_autoencoder import Trainer_Autoencoder
from embedding import Embedding

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


def experiment_embedding(config, dataset_grafi_nx, dataset_labels, list_p):
    num_last_neurons = config['model']['neurons_per_layer'][-1]
    if config['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"

    model = GCN(neurons_per_layer=config['model']['neurons_per_layer'],
                num_classes=len(config['graph_dataset']['list_p']))
    model.to(device)
    print(model)

    trainer = Trainer(model, config)
    print("Loading Dataset...")
    trainer.load_dataset(dataset_grafi_nx, dataset_labels, percentage_train=config['training']['percentage_train'])
    train_loss_list, test_loss_list = trainer.launch_training()

    # get embedding of all graphs in dataset
    # embed_model = GCNEmbed(model, config['model']['neurons_per_layer'])
    # embed_model = embed_model.to(device)
    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False) #batch_size=trainer.dataset.len_data
    embeddings_array = trainer.take_embedding(all_data_loader)
    embeddings_array = np.array([emb.cpu().detach().numpy() for emb in embeddings_array])
    #embeddings_array = model(batch.x, batch.edge_index, batch.batch).cpu().detach().numpy()
    embeddings = Embedding(embeddings_array, trainer.dataset.dataset, trainer.dataset.labels, list_p, test_loss_list, config)
    embeddings.calc_distances()

    return embeddings, trainer, test_loss_list


def autoencoder_embedding(config, dataset_grafi_nx, dataset_labels, list_p):
    num_last_neurons = config['model']['neurons_per_layer'][-1]
    if config['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"

    model = GAEGCNEncoder(neurons_per_layer=config['model']['neurons_per_layer'], put_batchnorm=config['model']['put_batchnorm'])
    model.to(device)
    print(model)

    #variational = False
    trainer = Trainer_Autoencoder(model, config)
    print("Loading Dataset...")
    trainer.load_dataset(dataset_grafi_nx, dataset_labels, percentage_train=config['training']['percentage_train'])
    train_loss_list, test_loss_list = trainer.launch_training()

    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.bs, shuffle=False)
    embeddings_array = trainer.take_embedding(all_data_loader)
    embeddings_array = np.array([emb.cpu().detach().numpy() for emb in embeddings_array])
    # embeddings_array = model(batch.x, batch.edge_index, batch.batch).cpu().detach().numpy()
    embeddings = Embedding(embeddings_array, trainer.dataset.labels, list_p, test_loss_list, config)
    #embeddings.calc_distances()

    return embeddings, trainer
