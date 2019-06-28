import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
#from pygcn.conv_layers import BlockConvolution
import torch
from torch.nn.parameter import Parameter
import math

class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        nhid = 1024     
        
        # self.weight = Parameter(torch.FloatTensor(adj_size, adj_size))
        # self.weight = Parameter(torch.FloatTensor(adj_size, 1))
        self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        # self.gc2 = GraphConvolution(nhid, nhid ,adj_size)
        self.gc3 = GraphConvolution(nhid, 512,adj_size)
        self.fc = nn.Linear(512,nclass)
        self.dropout = 0.5
        # self.reset_parameters()
           
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))########???
        self.weight.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.weight_.size(1))########???
        # self.weight_.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x = F.dropout(x, 0.5, training=self.training)      
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.5, training=self.training)    
        #x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        # x =  F.relu(x * self.weight)
        x = torch.mean(x,1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        
        return x
