import torch
import torch.nn as nn
import numpy as np

from models.dgi.gcn2 import *
# from gcn2 import *


class STDGI(nn.Module):
    def __init__(self, in_ft,out_ft):
        super(STDGI, self).__init__()
        self.encoder = Encoder(in_ft,out_ft)
        self.disc = Discriminator(h_ft=out_ft, x_ft=in_ft)

    def forward(self, x, adj):
        h = self.encoder(x, adj)
        x_c = self.corrupt(x)
        ret = self.disc(h, x, x_c)

        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        print('/////////', len(idx))
        shuf_fts = X[:, idx, :]
        return shuf_fts


class Encoder(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(in_ft, in_ft)
        self.gcn = GCN(in_ft, out_ft, hidden_unit=16)
    
    def forward(self, x, adj):
        x = self.fc(x)
        x = self.gcn(x, adj)
        return x

class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=1)
    
    def forward(self, h, x, x_c):
        ret1 = self.fc(h ,x)
        ret2 = self.fc(h, x_c)

        return torch.cat((ret1, ret2), 2)


# x = torch.randn((1,64, 28))
# G = torch.randn((28, 64, 28))
# model = STDGI(in_ft=8, out_ft=1)
# res = model(x, G)
# print(res.shape)

        