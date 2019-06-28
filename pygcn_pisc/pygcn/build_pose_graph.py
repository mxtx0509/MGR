import os
import numpy as np


# def build_graph():
    
# elif
skele_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

skele_connect =  [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

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

# for i in range(17):
    # skele_graph[i,17:]=1

for i in range(34):
    for j in range(0,i):
        skele_graph[i][j] = skele_graph[j][i]
np.savetxt('a.txt',skele_graph,fmt='%d')
np.save('pose_graph_1.npy',skele_graph)














