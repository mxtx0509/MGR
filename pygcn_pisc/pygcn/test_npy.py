import numpy as np
import os


file_list = '/export/home/zm/test/icme2019/SR_v1_pose/list/PIPC_fine_train.txt'
feature_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_pose/'
list_file = open(file_list)
lines = list_file.readlines()
for line in lines:
    box1 = [] # x1, y1, x2, y2
    box2 = []
    line = line.strip().split()
    assert (len(line) == 10), 'The len of the row in list file is {%d}, does not equal to 10'.format(len(line))
    file_id = line[0].split('.')[0]

    box1.append(int(line[1]))
    box1.append(int(line[2]))
    box1.append(int(line[3]))
    box1.append(int(line[4]))


    box2.append(int(line[5]))
    box2.append(int(line[6]))
    box2.append(int(line[7]))
    box2.append(int(line[8]))
    
    feature_name1 = feature_dir + file_id + '_%d_%d.npy'%(box1[0],box1[1])
    feature_name2 = feature_dir + file_id + '_%d_%d.npy'%(box2[0],box2[1])
    if not os.path.exists(feature_name1):
        print (feature_name1)

    if not os.path.exists(feature_name2):
        print (feature_name2)


















