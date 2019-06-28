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
        nfeat= 2048   

        self.gc1 = GraphConvolution(2048, nhid ,8)
        self.gc3 = GraphConvolution(nhid, 512,8)
        
        self.gc1_pose = GraphConvolution(90, 512 ,34)
        self.gc3_pose = GraphConvolution(512, 256,34)
        
        self.fc = nn.Linear(512+256,nclass)
        self.dropout = dropout
        # self.reset_parameters()
           
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))########???
        self.weight.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.weight_.size(1))########???
        # self.weight_.data.uniform_(-stdv, stdv)

    def forward(self, x, adj,pose,pose_adj):

        x = F.dropout(x, 0.5, training=self.training)      
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.5, training=self.training)    
        x = F.relu(self.gc3(x, adj))
        x = torch.mean(x,1)
        
        #  pose = F.dropout(pose, 0.5, training=self.training)      
        pose = F.relu(self.gc1_pose(pose, pose_adj))
        # pose = F.dropout(pose, 0.5, training=self.training)    
        pose = F.relu(self.gc3_pose(pose, pose_adj))
        pose = torch.mean(pose,1)
        
        x = torch.cat((x,pose),1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        
        return x
