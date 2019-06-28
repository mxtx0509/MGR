import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random

class SRDataset(data.Dataset):
    def __init__(self, adj_dir, feature_dir, file_list,adj_size=8,graph_mode='pose',is_train = True):
        super(SRDataset, self).__init__()

        self.adj_dir = adj_dir
        self.obj_dir = '/export/home/zm/test/icme2019/objects/PISC_obj_embedding/'
        self.visual_dir = feature_dir# '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/'   
        print (self.visual_dir)
        self.location_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_location_90/'     
        self.graph_dir = '/export/home/zm/test/icme2019/pygcn/graph/'
        self.file_list = file_list
        self.adj_size=adj_size
        self.obj_dic = {}
        self.is_train=is_train
        
        obj_list = os.listdir(self.obj_dir)
        obj_list.sort()
        for file in obj_list :
            id = file.split('_')[0]
            if id not in self.obj_dic:
                self.obj_dic[id] = []
            self.obj_dic[id].append(file)
        
        # print (self.obj_dic['00001'])
        self.names = []
        self.box1s = []
        self.box2s = []
        self.labels = []

        list_file = open(file_list)
        lines = list_file.readlines()
        label_dic = {}
        if is_train==True:
            random.shuffle (lines)
        print ('line 0 is', lines[0])
        for line in lines:
            box1 = [] # x1, y1, x2, y2
            box2 = []
            line = line.strip().split()
            assert (len(line) == 10), 'The len of the row in list file is {%d}, does not equal to 10'.format(len(line))
            if line[9] not in label_dic:
                label_dic[line[9]] = 0
            label_dic[line[9]]=label_dic[line[9]] +1
            if label_dic[line[9]] > 1000 and is_train == True:
                continue
            self.names.append(line[0])

            box1.append(int(line[1]))
            box1.append(int(line[2]))
            box1.append(int(line[3]))
            box1.append(int(line[4]))
            self.box1s.append(box1)

            box2.append(int(line[5]))
            box2.append(int(line[6]))
            box2.append(int(line[7]))
            box2.append(int(line[8]))
            self.box2s.append(box2)
            

            self.labels.append(int(line[9]))
        print (len(self.names))
        list_file.close()

    def __getitem__(self, index):
        # For normalize

        label = self.labels[index]
        file_name = self.names[index]
        box1 = self.box1s[index]
        box2 = self.box2s[index]
        
        file_id = file_name.split('.')[0]

        graph_feature = np.zeros((8,2048))
        
        
        feature_name1 =  file_id + '_%d_%d.npy'%(box1[0],box1[1])
        feature_name2 =  file_id + '_%d_%d.npy'%(box2[0],box2[1])
        union_name =  file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1])

        bbox1_visual = np.load(self.visual_dir + feature_name1)

        bbox2_visual = np.load(self.visual_dir + feature_name2)

        union_visual = np.load(self.visual_dir + union_name)

        

        graph_feature[0] = union_visual
        graph_feature[1] = bbox1_visual
        graph_feature[2] = bbox2_visual
        
        
        if file_id in self.obj_dic:
            obj_list = self.obj_dic[file_id]
            # if self.is_train == True:
                # random.shuffle(obj_list)
            for idx in range(len(obj_list)):
                obj_fn = obj_list[idx]
                id = obj_fn.split('.')[0].split('_')[2]
                graph_feature[3 + int(id)] = np.load(self.obj_dir + obj_fn)



        # adj_npy = np.loadtxt(self.graph_dir + file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1]))
        adj_npy = np.ones((8,8))
        adj_npy = self.L_Matrix(adj_npy,8) 

        adj_tensor = torch.FloatTensor(adj_npy)
        graph_feature_tensor = torch.FloatTensor(graph_feature)


        return adj_tensor, graph_feature_tensor,label

    def __len__(self):
        return len(self.names)



    def L_Matrix(self,adj_npy,adj_size):

        D =np.zeros((adj_size,adj_size))
        for i in range(adj_size):
            tmp = adj_npy[i,:]
            count = np.sum(tmp==1)
            if count>0:
                number = count ** (-1/2)
                D[i,i] = number

        x = np.matmul(D,adj_npy)
        L = np.matmul(x,D)
        return L






