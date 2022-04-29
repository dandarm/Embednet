import random
import datetime
import os

import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.utils.convert import from_networkx

from tqdm import tqdm

   
class Trainer():
    
    def __init__(self, model, learning_rate, epochs, batch_size, device=None):
        self.lr = learning_rate
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda')
    
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataset = None

    def train(self, epoch):
        self.model.train()

        for data in self.dataset.train_loader:  # Iterate in batches over the training dataset.            
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.  
            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
        return loss

    def test(self, epoch):
        self.model.eval()
        for data in self.dataset.test_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(out, data.y)  # Compute the loss.
        return loss

    def accuracy(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    
    
    def load_dataset(self, dataset_list, labels, percentage_train):
        self.dataset = Dataset(dataset_list, labels, percentage_train, self.batch_size, self.device)
        self.dataset.prepare()


    def launch_training(self):
        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        expstr = f"lr-{self.lr}_epochs{self.epochs}_bs{self.batch_size}_layers{3}"
        LOG_DIR = f"runs/{expstr}/{nowstr}"
        print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)

        for epoch in range(1, self.epochs):
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            train_acc = self.accuracy(self.dataset.train_loader)
            test_acc = self.accuracy(self.dataset.test_loader)
            #writer.add_scalar("Loss/train", loss, epoch)
            #writer.add_scalar("Train accuracy", train_acc, epoch)
            #writer.add_scalar("Test accuracy", test_acc, epoch)
            writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)
            writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
            #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        writer.flush()
    
    
    
    
class Dataset():
    
    def __init__(self, dataset_list, labels, percentage_train, batch_size, device):
        self.dataset = dataset_list # rename in dataset_list
        self.dataset_pyg = None
        self.labels = labels
        self.len_data = len(self.dataset)
        self.tt_split = int(self.len_data*percentage_train)
        self.train_dataset = None
        self.test_dataset = None
        self.bs = batch_size
        self.device = device
        self.train_loader = None
        self.test_loader = None
        
    def nx2pyg(self, graph_list_nx, graph_labels):
        dataset_pyg = []
        i = 0
        for g in tqdm(graph_list_nx, total = len(graph_list_nx)):

            # aggiungo i metadati x e y per l'oggetto Data di PYG
            nodi = list(g.nodes)
            for n in nodi:
                g.nodes[n]["x"] = [1.0]

            pyg_graph = from_networkx(g)
            #type_graph = 0 if i < Num_grafi_per_tipo else 1
            type_graph = graph_labels[i]
            
            pyg_graph.y = torch.tensor([type_graph],dtype = torch.long)

            pyg_graph = pyg_graph.to(self.device)
            dataset_pyg.append(pyg_graph)
            i+=1
        return dataset_pyg
        
    def prepare(self):        
        self.dataset_pyg = self.nx2pyg(self.dataset, self.labels)
        
        # shuffle before train test split
        x = list(enumerate(self.dataset_pyg))
        random.shuffle(x)
        indices, self.dataset_pyg = zip(*x)
        self.labels = self.labels[list(indices)]
        
        self.train_dataset = self.dataset_pyg[:self.tt_split]
        self.test_dataset = self.dataset_pyg[self.tt_split:]
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False)

        """
        for step, data in enumerate(self.train_loader):
            print(f'Step {step + 1}:')
            print('=======')
            print(f'Number of graphs in the current batch: {data.num_graphs}')
            print(data)
            print()
        """