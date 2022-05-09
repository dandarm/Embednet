from models import GCN, GCNEmbed, GCN1n
from train import Trainer
from embedding import Embedding

import torch
from torch_geometric.loader import DataLoader

def experiment_embedding_1D_ER(config, dataset_grafi_nx, dataset_labels, list_p):
    #config = yaml.safe_load(open("configs.yml"))
    if config['device'] == 'gpu':
        device = torch.device('cuda')
    else:
        device = "cpu"

    model = GCN1n(hidden_channels=config['model']['num_neurons'])
    model.to(device)
    #print(model)
    
    trainer = Trainer(model, config)
    print("Loading Dataset...")
    trainer.load_dataset(dataset_grafi_nx, dataset_labels, percentage_train=config['training']['percentage_train'])
    train_loss_list, test_loss_list = trainer.launch_training()
    
    # get embedding of all graphs in dataset
    embed_model = GCNEmbed(model)
    embed_model = embed_model.to(device)
    whole_data = trainer.dataset.dataset_pyg
    all_data_loader = DataLoader(whole_data, batch_size=trainer.dataset.len_data, shuffle=False)
    batch = next(iter(all_data_loader))
    embeddings_array = embed_model(batch.x, batch.edge_index, batch.batch).cpu().detach().numpy()
    
    
    embeddings = Embedding(embeddings_array, trainer.dataset.labels, list_p, test_loss_list)
    embeddings.calc_distances()
    
    return embeddings