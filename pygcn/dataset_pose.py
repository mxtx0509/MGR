import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random

class SRDataset(data.Dataset):
    def __init__(self,args, file_list,is_train=False):
        super(SRDataset, self).__init__()

        self.obj_dir = args.fea_obj_dir #'/export/home/zm/test/icme2019/objects/PISC_obj_embedding/'
        self.visual_dir = args.fea_person_dir # '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/'   
        print (self.visual_dir)
        self.location_dir = args.fea_pose_dir #'/export/home/zm/test/icme2019/pose_embedding/PISC_location_90/'    
        self.graph_perobj_dir = args.graph_perobj_dir #'/export/home/zm/test/icme2019/pygcn/graph/'
        self.graph_pose_dir = args.graph_pose_dir #'/export/home/zm/test/icme2019/pygcn/graph_pose/'
        self.file_list = file_list
        self.obj_dic = {}
        self.is_train=is_train

        
        obj_list = os.listdir(self.obj_dir)
        obj_list.sort()
        for file in obj_list :
            id = file.split('_')[0]
            if id not in self.obj_dic:
                self.obj_dic[id] = []
            self.obj_dic[id].append(file)
        
        print (self.obj_dic['00001'])
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

        perobj_feature = np.zeros((8,2048))
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
        perobj_feature[0] = union_visual
        perobj_feature[1] = bbox1_visual
        perobj_feature[2] = bbox2_visual
        
        
        if file_id in self.obj_dic:
            obj_list = self.obj_dic[file_id]
            for idx in range(len(obj_list)):
                obj_fn = obj_list[idx]
                id = obj_fn.split('.')[0].split('_')[-1]
                perobj_feature[3 + int(id)] = np.load(self.obj_dir + obj_fn)


        graph_perobj = np.loadtxt(self.graph_perobj_dir + file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1]))
        graph_perobj = self.L_Matrix(graph_perobj,8) 
        graph_pose = np.loadtxt(self.graph_pose_dir + file_id + '_%d_%d_%d_%d.txt'%(box1[0],box1[1],box2[0],box2[1]))
        graph_pose = self.L_Matrix(graph_pose,34) 


        graph_perobj = torch.FloatTensor(graph_perobj)
        perobj_feature_tensor = torch.FloatTensor(perobj_feature)
        graph_pose = torch.FloatTensor(graph_pose)
        pose_feature_tensor = torch.FloatTensor(pose_feature)

        return graph_perobj, perobj_feature_tensor,graph_pose,pose_feature_tensor,label

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






