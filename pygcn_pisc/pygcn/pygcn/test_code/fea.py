import numpy
import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random
def get_visual(heatmap,feature):
    result = np.zeros((17,656))
    for i in range(17):
        sorted_value = np.sort(heatmap[i], axis=None)
        x,y = np.where(heatmap[i] == sorted_value[-1])
        x = x[0]; y = y[0]
        x_start = x -10; x_end = x + 10
        y_start = y -10; y_end = y + 10
        if x-10 < 0:
            x_start = 0; x_end =20
        if x+10 > heatmap.shape[1]:
            x_start =heatmap.shape[1]-20; x_end =heatmap.shape[1]
        if y-10 < 0:
            y_start = 0; y_end =20
        if y+10 > heatmap.shape[2]:
            y_start =heatmap.shape[2]-20; y_end =heatmap.shape[2]

        
        temp_heat = heatmap[i,x_start:x_end,y_start:y_end]
        temp_fea = feature[:,x_start:x_end,y_start:y_end]
        temp = temp_heat * temp_fea
        channel_fea = np.mean(temp,axis=0)
        channel_fea = channel_fea.reshape(channel_fea.shape[0]*channel_fea.shape[1])
        spatial_fea = np.mean(temp,axis=1)
        spatial_fea = np.mean(spatial_fea,axis=1)
        temp = np.concatenate((channel_fea,spatial_fea),axis=0) 
        result[i] = temp
        #print (i,'what wrong')
    return result
feature_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_256/'
heatmap_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_17/'
combine_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_656/'
reader = open('/export/home/zm/test/icme2019/SR_v1_pose/list/PISC_fine_all.txt')
lines = reader.readlines()

box1s = []
box2s = []
count = 0
for line in lines:
    count = count +1
    if count%1000==0:
        print (count)
    box1 = [] # x1, y1, x2, y2
    box2 = []
    line = line.strip().split()
    assert (len(line) == 10), 'The len of the row in list file is {%d}, does not equal to 10'.format(len(line))
    file_name = line[0]

    box1.append(int(line[1]))
    box1.append(int(line[2]))
    box1.append(int(line[3]))
    box1.append(int(line[4]))

    box2.append(int(line[5]))
    box2.append(int(line[6]))
    box2.append(int(line[7]))
    box2.append(int(line[8]))
    
    file_id = file_name.split('.')[0]
    
    bbox1_name = file_id + '_%d_%d.npy'%(box1[0],box1[1])
    bbox1_17 = np.load(heatmap_dir + bbox1_name)
    bbox1_256 = np.load(feature_dir + bbox1_name)
    bbox1_visual = get_visual(bbox1_17,bbox1_256)
    
    bbox2_name = file_id + '_%d_%d.npy'%(box2[0],box2[1])
    bbox2_17 = np.load(heatmap_dir + bbox2_name)
    bbox2_256 = np.load(feature_dir + bbox2_name)
    bbox2_visual = get_visual(bbox2_17,bbox2_256)
    
    #pose_feature = np.concatenate((bbox1_visual,bbox2_visual),axis=0)
    np.save(combine_dir + bbox1_name,bbox1_visual )
    np.save(combine_dir + bbox2_name,bbox2_visual )







