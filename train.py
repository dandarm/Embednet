import random
import datetime
import os
from time import time

from multiprocessing import Pool

import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils.convert import from_networkx

from pytorchtools import EarlyStopping

from tqdm import tqdm

from models import GCN1n

   
class Trainer():
    
    def __init__(self, model, config): #learning_rate, epochs, batch_size, layers, neurons, last_layer_neurons, criterion, device=None):
        self.model = model
        self.config = config

        self.lr = config['training']['learning_rate']
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        
        self.layers= config['model']['layers']
        self.neurons = config['model']['num_neurons']
        self.last_layer_neurons = config['model']['num_last_neurons']
        
        if config['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = "cpu"
    
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr , )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        decayRate = 0.96
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
        
        criterion = config['training']['criterion']
        if criterion == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif criterion == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss()
            
        self.dataset = None

    def correct_shape(self, y):
        if isinstance(self.model, GCN1n):
            target = y.unsqueeze(1).float()
        else:
            target = y
        return target
        
    def train(self, epoch):
        self.model.train()

        for data in self.dataset.train_loader:  # Iterate in batches over the training dataset.            
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.  
            target = self.correct_shape(data.y)
            #print(target)
            loss = self.criterion(out, target)  # Compute the loss.
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            #self.scheduler.step()
        return loss.item()

    def test(self, epoch):
        self.model.eval()
        for data in self.dataset.test_loader:  # Iterate in batches over the training dataset.
            out = self.model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            target = self.correct_shape(data.y)
            loss = self.criterion(out, target)  # Compute the loss.
        return loss.item()

    def accuracy(self, loader):
        self.model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = self.model(data.x, data.edge_index, data.batch) 
            target = self.correct_shape(data.y)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == target).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.
    
    
    def load_dataset(self, dataset_list, labels, percentage_train):
        self.dataset = Dataset(dataset_list, labels, percentage_train, self.batch_size, self.device)
        self.dataset.prepare()


    def launch_training(self):
        nowstr = datetime.datetime.now().strftime("%d%b_%H-%M-%S")
        expstr = f"lr-{self.lr}_epochs{self.epochs}_bs{self.batch_size}_layers{self.layers}_neurons{self.neurons}_lln{self.last_layer_neurons}"
        LOG_DIR = f"runs/{expstr}/{nowstr}"
        print(LOG_DIR)
        writer = SummaryWriter(LOG_DIR)

        #best_loss = 0 # for early stopping
        early_stopping = EarlyStopping(patience=200, delta = 0.02, initial_delta = 0.2)
        
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        for epoch in range(1, self.epochs):
            train_loss = self.train(epoch)
            test_loss = self.test(epoch)
            train_acc = self.accuracy(self.dataset.train_loader)
            test_acc = self.accuracy(self.dataset.test_loader)
            #writer.add_scalar("Loss/train", loss, epoch)
            #writer.add_scalar("Train accuracy", train_acc, epoch)
            writer.add_scalar("Test Loss", test_loss, epoch)
            #writer.add_scalars(f'Loss', {'Train': train_loss, 'Test': test_loss}, epoch)            
            #writer.add_scalars(f'Accuracy', {'Train': train_acc, 'Test': test_acc}, epoch)
            #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            #train_acc_list.append(train_acc)
            #test_acc_list.append(test_acc)
            
            #if test_loss > best_loss:  # check for early stopping
            #    best_loss = test_loss
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Early stopping!!!")
                break
            
        writer.flush()
        return train_loss_list, test_loss_list#, train_acc_list, test_acc_list
    
    
    
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
        
    def convert_G(self, g_i):
        g, i = g_i
        # aggiungo i metadati x e y per l'oggetto Data di PYG
        nodi = list(g.nodes)
        for n in nodi:
            g.nodes[n]["x"] = [1.0]

        pyg_graph = from_networkx(g)
        type_graph = self.labels[i]
        # print(f'type_graph {type_graph}')
        pyg_graph.y = torch.tensor([type_graph],dtype = torch.float)
        
        return pyg_graph
        
    def nx2pyg(self, graph_list_nx):
        dataset_pyg = []
        total = len(graph_list_nx)
        """
        with Pool(processes=12) as p:
            total = len(graph_list_nx)
            with tqdm(total=total) as pbar:
                for pyg_graph in p.imap_unordered(self.convert_G, zip(graph_list_nx, range(total)) ):
                    pbar.update()
                    dataset_pyg.append(pyg_graph)
        
        # process the test list elements in parallel
        pool = Pool(processes=32)
        dataset_pyg = pool.map(self.convert_G, zip(graph_list_nx, range(total)))

        """
        i = 0
        for g in tqdm(graph_list_nx, total = len(graph_list_nx)):
            pyg_graph = self.convert_G((g,i))
            dataset_pyg.append(pyg_graph)            
            i+=1
        
        """
        from joblib import Parallel, delayed
        def process(i):
            return i * i

        results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
        print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        """
        
        for pyg_graph in dataset_pyg:
            pyg_graph = pyg_graph.to(self.device)
            
        return dataset_pyg
        
    def prepare(self):        
        starttime = time()
        self.dataset_pyg = self.nx2pyg(self.dataset)
        durata = time() - starttime
        print(f"Tempo impiegato: {durata}")
        
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