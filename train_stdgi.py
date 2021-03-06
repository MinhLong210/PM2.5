import torch
import torch.nn as nn
from utils.dgi import process
import scipy.sparse as sp
import pandas as pd
import numpy as np

from models.dgi.stdgi import *
from config.config import *

#hyper params
time_steps =  TIME_STEPS

pm_dataset = pd.read_csv('./data/pm.csv')
pm_dataset = pm_dataset.replace("**", 0)
pm_dataset = pm_dataset.to_numpy()
pm_data = pm_dataset[:, 4:]
pm_data = pm_data.astype(np.float) 

adj = pd.read_csv('./data/locations.csv').to_numpy()
adj = adj[:28, :28] #adj matrix of 28 nodes
adj = process.build_graph(adj)
adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
adj = (adj + sp.eye(adj.shape[0])).todense()
adj = torch.FloatTensor(adj[np.newaxis])
# print(adj.shape)

pm_data = pm_data[:, :40]

idx = int(pm_data.shape[1]*0.7)


train_data = pm_data[: , :idx]
val_data = pm_data[:, idx:int(pm_data.shape[1]*0.8)]
test_data = pm_data[:, int(pm_data.shape[1]*0.8) : ]

# print(train_data.shape, val_data.shape, test_data.shape)

def getData(dataset, sequence_length, horizon, output_dim=1):
    
    T = dataset.shape[0]-sequence_length-horizon
    ips = np.empty(shape=(T, sequence_length, dataset.shape[1]-output_dim, 1))
    # ops = np.empty(shape=(T, horizon, output_dim))
    for i in range(T):
        dt = dataset[i:i+sequence_length, 0:(dataset.shape[1]-output_dim)]
        dt = dt.reshape(sequence_length,dataset.shape[1]-output_dim,1)
        ips[i, :, :,0] = dt[:,:,0]
        # ops[i, :, :] = dataset[i+sequence_length:i+sequence_length+horizon, -output_dim]
    
    return torch.Tensor(ips)


train_data = getData(train_data, TIME_STEPS, TIME_STEPS, 0)

adj = adj.expand((train_data.shape[1], adj.shape[1], adj.shape[2]))

model = STDGI(in_ft=train_data.shape[3], out_ft=1)
x = torch.squeeze(train_data[0], dim=0)

res = model(x, adj)
print(res.shape)

