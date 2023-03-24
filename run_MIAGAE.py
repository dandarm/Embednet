import sys
#sys.path.append('/workspace/Embednet/')
sys.path.append('/workspace/Graph_AE/')

from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph
from utils.train_utils import train_cp
import torch
import argparse

from classification.Graph_AE import Net


args = {'d':'REDDIT-BINARY','m':'MIAGAE', 'device':'cuda', 'batch':512, 'e':100, 'lr':0.001, 'model_dir':'data/model/', 'n_train':1500, 'n_test':1000, 'k':2, 'depth':3, 'c_rate':0.8, 'shapes':'64,64,64'}

device = torch.device(args['device'])

num_epoch = args['e']
batch_size = args['batch']

SelectGraph.data_name = args['d']
data_set = SelectGraph('data/' + SelectGraph.data_name)

#check sulle feature
dataset_mod = []
for d in data_set:
    if d.x is None:
        d.x = torch.ones([d.num_nodes], dtype=torch.float).unsqueeze(1) 
    dataset_mod.append(d)

    
train_set = DataLoader(dataset_mod[:args['n_train']], batch_size=batch_size, shuffle=True)
test_set = DataLoader(dataset_mod[args['n_train']:args['n_train'] + args['n_test']], batch_size=batch_size, shuffle=False)

input_size = 1 # data_set.num_features
shapes = list(map(int, args['shapes'].split(",")))
model = Net(input_size, args['k'], args['depth'], [args['c_rate']] * args['depth'], shapes, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
train_cp(model, optimizer, device, train_set, test_set, num_epoch, args['model_dir'], args['m'])