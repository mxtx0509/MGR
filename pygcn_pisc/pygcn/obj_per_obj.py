import os
import numpy as np
import math
import json
import cv2
def comupte_dist (p1,p2):
    p3=p2-p1
    p4=math.hypot(p3[0],p3[1])
    return p4
def combine(a,b,c):
    CombineList = list();
    for index in range(len(a)):
        CombineList.append((a[index],b[index],c[index]));
    return CombineList


def compute_iou(box1,box2):
    area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
    area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)

    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    h = max(0, yy2-yy1+1)
    w = max(0, xx2-xx1+1)

    intersection = w * h

    iou = intersection / (area1 + area2 - intersection)
    return iou

def get_obj_bbox(file_id,json_name):
    obj_list = []
    obj_score = []
    score_list = []
    
    with open (json_name) as f:
        num_sum = 0
        box_list = []
        score_list = []
        key_list = []
        data = json.load(f)
        for key in data:
            if key == 'person':
                continue
            dic = data[key]
            box = dic['bbox']
            num = dic['num']
            score = dic['scores']
            num_sum = num_sum + num
            for i in range(num):
                box_list.append(box[i])
                score_list.append(score[i])
                key_list.append(key)
        if num_sum == 0:
            return []
        all_list = combine(key_list,score_list,box_list)
        all_list.sort(key=lambda x:x[1],reverse=True)
        if num_sum > 5:
            all_list = all_list[0:5]
        for ele in range(len(all_list)):
            box_change = []
            key_,score_,box1 = all_list[ele]
            box_change.append(box1[1])
            box_change.append(box1[0])
            box_change.append(box1[3])
            box_change.append(box1[2])
            obj_list.append(box_change)
            obj_score.append(score_)
    return obj_list

def find(heatmap,box1,file_name):
    result = np.zeros((17,2))
    imagedir = '/export/home/zm/dataset/PIPA/image/'
    image = cv2.imread(imagedir + file_name)
    h_image, w_image = image.shape[0] , image.shape[1]


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
        value = sorted_value[-1]

        index_list = []
        temp_index = np.where(heatmap[i] == value) 
        
        y = int(temp_index[0][0] * y_scale)
        x = int(temp_index[1][0] * x_scale)

        y = (y + y_min)#/h_image
        x = (x + x_min)#/w_image
        if x >w_image or y>h_image:
            y = h_image
            x = w_image
        result[i,0] = x
        result[i,1] = y
    return result,h_image, w_image 

def build_graph(count,heatmap,result,file_name,obj_bbox_list,h_image, w_image):
    imagedir = '/export/home/zm/dataset/PIPA/image/'
    image = cv2.imread(imagedir + file_name)
    #result = find(heatmap,box,file_name)
    iou_graph = np.zeros((5))
    box1 = [0,0,0,0]

    for i in range(17):
        box1[0] = int(max(result[i,0]-25, 0))
        box1[1] = int(max(result[i,1]-25, 0))
        box1[2] = int(min(result[i,0]+25, w_image))
        box1[3] = int(min(result[i,1]+25, h_image))
        image = cv2.rectangle(image,(box1[0],box1[1]),(box1[2],box1[3]),(255,0,0),2)
        
        for j in range(len(obj_bbox_list)):
            iou_temp = compute_iou(box1, obj_bbox_list[j])
            if iou_temp > 0:
                iou_graph[j] = 1
            image = cv2.rectangle(image,(obj_bbox_list[j][0],obj_bbox_list[j][1]),(obj_bbox_list[j][2],obj_bbox_list[j][3]),(0,255,0),2)
    cv2.imwrite(str(count)+'_'+file_name,image)
    return iou_graph
    
    
heatmap_dir = '/export/home/zm/test/icme2019/pose_embedding/PIPA_17/'
graph_dir = '/export/home/zm/test/icme2019/pygcn_pipa/graph/'
json_dir = '/export/home/zm/test/icme2019/pytorch-mask-rcnn/result_json/PIPA_json/test/'
reader = open('/export/home/zm/test/icme2019/SR_graph/list/PIPA_relation_test_zm.txt')
print (heatmap_dir)
print (graph_dir,json_dir )


lines = reader.readlines()

count = 0
for line in lines[900:]:
    graph = np.zeros((8,8))
    graph[0:3,0:3] = 1
    
    count = count +1
    if count%1000==0:
        print (count)
    if count ==10:   
        break
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
    
    json_name = json_dir + file_id + '.json'
    obj_bbox_list  = get_obj_bbox(file_id,json_name)
    
    
    bbox1_name = file_id + '_%d_%d.npy'%(box1[0],box1[1])
    heatmap1 = np.load(heatmap_dir + bbox1_name)
    bbox1_17 ,h_image, w_image= find(heatmap1,box1,file_name)
    
    bbox2_name = file_id + '_%d_%d.npy'%(box2[0],box2[1])
    heatmap2 = np.load(heatmap_dir + bbox2_name)
    bbox2_17 ,h_image, w_image= find(heatmap2,box2,file_name)
    
    graph1 = build_graph(count,heatmap1,bbox1_17,file_name,obj_bbox_list,h_image, w_image)
    graph2 = build_graph(count,heatmap2,bbox2_17,file_name,obj_bbox_list,h_image, w_image)
    # graph[1,3:8] = graph1
    # graph[2,3:8] = graph2
    # temp = np.ones((5))
    # union = graph1 + graph2
    # union = np.where(union<=1, union, temp)
    # graph[0,3:8] = union
    # graph[3:8,3:8] = 1
    # for i in range(8):
        # for j in range(8):
            # if graph[i,j] == 1:
                # graph[j,i] = 1
            # elif graph[j,i] == 1:
                # graph[i,j] = 1
    # graph_name =file_id + '_%d_%d_%d_%d.txt'%(box1[0],box1[1],box2[0],box2[1])
    # np.savetxt(graph_dir + graph_name,graph,fmt='%d')    


    # # graph2 = build_graph(heatmap2,box2,file_name,obj_bbox_list,h_image, w_image)
    
    
    
    # np.savetxt('a.txt',graph,fmt='%d')
    # graph_name =file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1])
    # np.save(graph_dir + graph_name,graph )













