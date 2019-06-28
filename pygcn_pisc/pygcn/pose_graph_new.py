import os
import math
import numpy as np

def comupte_dist (p1,p2):
    dis = math.sqrt( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) )
    return dis




def build_graph(location1,location2):

    skele_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]   

    soft_key = [0,9,10,15,16]

    skele_connect =  [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

    skele_graph = np.zeros((34,34))

    skele_graph1 = np.zeros((17,17))
    for i in range(len(skele_connect)):
        skele_a = skele_connect[i][0] - 1
        skele_b = skele_connect[i][1] - 1
        if skele_graph1[skele_a,skele_b] ==0. :
            skele_graph1[skele_a,skele_b]=1
            skele_graph1[skele_b,skele_a]=1
    for i in range(17):
        skele_graph1[i,i]=1

    skele_graph[0:17,0:17] = skele_graph1
    skele_graph[17:34,17:34] = skele_graph1

    #def build_graph(skele_graph,heatmap1,heatmap2,box1,box2,file_name):
    pose_graph = np.zeros((17,17))   
    
    distance = np.zeros((17,17))
    ones = np.ones((17,17))

    for i in range(17):
        for j in range(17):
            distance[i,j] = comupte_dist(location1[i,0:2],location2[j,0:2])
            if i in soft_key:
                pose_graph[i,j] = 1
    
    # print (distance.max())
    max_dis = distance.max()
    weight = ones*2 - distance/max_dis
    pose_graph = weight * pose_graph
    # print (weight)
    
    skele_graph[0:17,17:34] = pose_graph
    skele_graph[17:34,0:17] = pose_graph    



    for i in range(34):
        for j in range(34):
            if skele_graph[i,j] != 0:
                skele_graph[j,i] = skele_graph[i,j]
            elif skele_graph[j,i]  != 0:
                skele_graph[i,j] = skele_graph[j,i]
    return skele_graph
    # np.savetxt('a.txt',skele_graph)
    # np.save('graph_only_pose.npy',skele_graph )


location_dir = '/export/home/zm/test/icme2019/pose_embedding/PIPA_location/'    
graph_dir = '/export/home/zm/test/icme2019/pygcn_pipa/graph_pose/'
txt = '/export/home/zm/test/icme2019/SR_graph/list/PIPA_relation_test_zm.txt'  
reader = open(txt)
print (location_dir)
print (graph_dir )
print (txt )


lines = reader.readlines()
print (len(lines))

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
    location1 = np.load(location_dir + bbox1_name)


    bbox2_name = file_id + '_%d_%d.npy'%(box2[0],box2[1])
    location2 = np.load(location_dir + bbox2_name)
    
    graph = build_graph(location1, location2)
    graph_name =file_id + '_%d_%d_%d_%d.txt'%(box1[0],box1[1],box2[0],box2[1])
    np.savetxt(graph_dir + graph_name,graph ,fmt='%.6f')


















