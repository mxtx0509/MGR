import os
import numpy as np
import math
def comupte_dist (p1,p2):
    p3=p2-p1
    p4=math.hypot(p3[0],p3[1])
    return p4

def get_center(heatmap):
    sorted_value = np.sort(heatmap, axis=None)
    x,y = np.where(heatmap == sorted_value[-1])
    x = x[0]; y = y[0]
    return np.array([x,y])

def find(heatmap,box1,file_name):

    """
    top-10's location  in the original pic & value of them in the feature map

    :param bbox_info: The information of the bbox, containing the npy file name and the size of the bbox
    :return: dict = {bbox_name:{channel_no:[(x1,y1,c10),....(x10,y10,c10)]}}
    """
    result = np.zeros((17,2))
    imagedir = '/export/home/zm/dataset/PISC/image/'
    image = cv2.imread(imagedir + file_name)
    h_image, w_image = image.shape[0] , image.shape[1]
    print (h_image, w_image)

    x_min = box1[0]
    y_min = box1[1]

    h_heat1 = heatmap.shape[1]
    w_heat1 = heatmap.shape[2]
    
    h_box1 = box1[3] - box1[1]
    w_box1 = box1[2] - box1[0]

    x_scale = w_box1 / w_heat1
    y_scale = h_box1 / h_heat1

    for i in range(17):
        sorted_value = np.sort(heatmap[i], axis=None)
        # print(sorted_value)
        value = sorted_value[-1]
        
        # print(len(top10))
        index_list = []
        temp_index = np.where(heatmap[i] == value) 
        y = int(temp_index[0][0] * y_scale)
        x = int(temp_index[1][0] * x_scale)

        y = (y + y_min)#/h_image
        x = (x + x_min)#/w_image
        if x >w_image or y>h_image:
            print (file_id,'  erro')
            y = h_image
            x = w_image
        result[i,0] = x
        result[i,1] = y
        # image = cv2.rectangle(image,(x-10,y-10),(x+10,y+10),(255,0,0),2)
    # cv2.imwrite(str(tag)+file_name,image)

    return result

    
skele_name = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

soft_key = [0,9,10,15,16]

skele_connect =  [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]

skele_graph = np.zeros((44,44))
skele_graph[0:10,0:10] = 1
skele_graph[1,10:27] = 1
skele_graph[2,27:44] = 1
skele_graph[8,10:27] = 1
skele_graph[9,27:44] = 1

skele_graph1 = np.zeros((17,17))
for i in range(len(skele_connect)):
    skele_a = skele_connect[i][0] - 1
    skele_b = skele_connect[i][1] - 1
    if skele_graph1[skele_a,skele_b] ==0. :
        skele_graph1[skele_a,skele_b]=1
        skele_graph1[skele_b,skele_a]=1
for i in range(17):
    skele_graph1[i,i]=1

skele_graph[10:27,10:27] = skele_graph1
skele_graph[27:44,27:44] = skele_graph1

#def build_graph(skele_graph,heatmap1,heatmap2,box1,box2,file_name):
pose_graph = np.zeros((17,17))   

for i in range(17):
    for j in range(17):
        if i in soft_key:
            pose_graph[i,j] = 1

skele_graph[10:27,27:44] = pose_graph
skele_graph[27:44,10:27] = pose_graph    

np.savetxt('b.txt',skele_graph,fmt='%d')

for i in range(44):
    for j in range(44):
        if skele_graph[i,j] == 1:
            skele_graph[j,i] = 1
        elif skele_graph[j,i] == 1:
            skele_graph[i,j] = 1

np.savetxt('a.txt',skele_graph,fmt='%d')
np.save('graph_pose.npy',skele_graph )

# heatmap_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_17/'
# graph_dir = '/export/home/zm/test/icme2019/pygcn/graph/'
# reader = open('/export/home/zm/test/icme2019/SR_v1_pose/list/PISC_fine_test.txt')
# lines = reader.readlines()

# box1s = []
# box2s = []
# count = 0
# for line in lines:
    # count = count +1
    # if count%1000==0:
        # print (count)
    # box1 = [] # x1, y1, x2, y2
    # box2 = []
    # line = line.strip().split()
    # assert (len(line) == 10), 'The len of the row in list file is {%d}, does not equal to 10'.format(len(line))
    # file_name = line[0]

    # box1.append(int(line[1]))
    # box1.append(int(line[2]))
    # box1.append(int(line[3]))
    # box1.append(int(line[4]))

    # box2.append(int(line[5]))
    # box2.append(int(line[6]))
    # box2.append(int(line[7]))
    # box2.append(int(line[8]))
    
    # file_id = file_name.split('.')[0]
    
    # bbox1_name = file_id + '_%d_%d.npy'%(box1[0],box1[1])
    # heatmap1 = np.load(heatmap_dir + bbox1_name)
    
    # bbox2_name = file_id + '_%d_%d.npy'%(box2[0],box2[1])
    # heatmap2 = np.load(heatmap_dir + bbox2_name)
    
    # graph = build_graph(skele_graph,heatmap1,heatmap2,box1,box2,file_name)
    
    # # np.savetxt('a.txt',graph,fmt='%d')
    # graph_name =file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1])
    # np.save(graph_dir + graph_name,graph )













