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
        self.obj_dir = '/export/home/zm/test/icme2019/objects/PIPA_obj_embedding/'
        self.visual_dir = feature_dir# '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/'   
        print (self.visual_dir)
        self.location_dir = '/export/home/zm/test/icme2019/pose_embedding/PIPA_location/'      
        self.graph_perobj_dir = '/export/home/zm/test/icme2019/pygcn_pipa/graph/'
        self.graph_pose_dir = '/export/home/zm/test/icme2019/pygcn_pipa/graph_pose/'
        self.file_list = file_list
        self.adj_size=adj_size
        self.obj_dic = {}
        self.is_train=is_train
        self.graph_adj = self.graph_build()
        np.savetxt('a.txt',self.graph_adj,fmt = '%d')
        
        obj_list = os.listdir(self.obj_dir)
        obj_list.sort()
        for file in obj_list :
            id_list = file.split('_')
            id = '_'.join(id_list[0:2])
            if id not in self.obj_dic:
                self.obj_dic[id] = []
            self.obj_dic[id].append(file)
        
        print (self.obj_dic['72157632944861067_8539657864'])
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
            # if line[9] not in label_dic:
                # label_dic[line[9]] = 0
            # label_dic[line[9]]=label_dic[line[9]] +1
            # if label_dic[line[9]] > 3000 and is_train == True:
                # continue
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
        pose_feature = np.zeros((34,90))
        
        
        feature_name1 =  file_id + '_%d_%d.npy'%(box1[0],box1[1])
        feature_name2 =  file_id + '_%d_%d.npy'%(box2[0],box2[1])
        union_name =  file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1])
        
        
        bbox1_location = np.load(self.location_dir + feature_name1)
        bbox1_visual = np.load(self.visual_dir + feature_name1)

        bbox2_location = np.load(self.location_dir + feature_name2)
        bbox2_visual = np.load(self.visual_dir + feature_name2)

        
        pose_feature[0:17] = bbox1_location
        pose_feature[17:34] = bbox2_location
        
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
                id = obj_fn.split('.')[0].split('_')[-1]
                graph_feature[3 + int(id)] = np.load(self.obj_dir + obj_fn)


        graph_perobj = np.loadtxt(self.graph_perobj_dir + file_id + '_%d_%d_%d_%d.txt'%(box1[0],box1[1],box2[0],box2[1]))
        graph_perobj = self.L_Matrix(graph_perobj,8) 
        graph_pose = np.loadtxt(self.graph_pose_dir + file_id + '_%d_%d_%d_%d.txt'%(box1[0],box1[1],box2[0],box2[1]))
        graph_pose = self.L_Matrix(graph_pose,34) 

        graph_perobj = torch.FloatTensor(graph_perobj)
        graph_feature_tensor = torch.FloatTensor(graph_feature)
        graph_pose = torch.FloatTensor(graph_pose)
        pose_feature_tensor = torch.FloatTensor(pose_feature)

        return graph_perobj, graph_feature_tensor,graph_pose,pose_feature_tensor,label

    def __len__(self):
        return len(self.names)

    def graph_build(self):
        skele_graph = np.zeros((42,42))
        soft_key = [0,9,10,15,16]

        skele_connect =  [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

        skele_graph[1,8:25] = 1
        skele_graph[2,25:42] = 1
        
        skele_graph[8,8:25] = 1
        skele_graph[9,25:42] = 1

        
        skele_graph1 = np.zeros((17,17))
        for i in range(len(skele_connect)):
            skele_a = skele_connect[i][0] - 1
            skele_b = skele_connect[i][1] - 1
            if skele_graph1[skele_a,skele_b] ==0. :
                skele_graph1[skele_a,skele_b]=1
                skele_graph1[skele_b,skele_a]=1
        for i in range(17):
            skele_graph1[i,i]=1

        skele_graph[8:25,8:25] = skele_graph1
        skele_graph[25:42,25:42] = skele_graph1

        #def build_graph(skele_graph,heatmap1,heatmap2,box1,box2,file_name):
        pose_graph = np.zeros((17,17))   

        for i in range(17):
            for j in range(17):
                if i in soft_key:
                    pose_graph[i,j] = 1

        skele_graph[8:25,25:42] = pose_graph
        skele_graph[25:42,8:25] = pose_graph    

        for i in range(42):
            for j in range(42):
                if skele_graph[i,j] == 1:
                    skele_graph[j,i] = 1
                elif skele_graph[j,i] == 1:
                    skele_graph[i,j] = 1
        return skele_graph



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






