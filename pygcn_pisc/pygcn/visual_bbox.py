import cv2
import numpy as np
def find(heatmap1,box1,file_name,tag):

    """
    top-10's location  in the original pic & value of them in the feature map

    :param bbox_info: The information of the bbox, containing the npy file name and the size of the bbox
    :return: dict = {bbox_name:{channel_no:[(x1,y1,c10),....(x10,y10,c10)]}}
    """
    result = np.zeros((17,30))
    imagedir = '/export/home/zm/dataset/PIPA/image/'
    image = cv2.imread(imagedir + file_name)
    h_image, w_image = image.shape[0] , image.shape[1]
    print (h_image, w_image)

    x_min = box1[0]
    y_min = box1[1]

    h_heat1 = heatmap1.shape[1]
    w_heat1 = heatmap1.shape[2]
    
    h_box1 = box1[3] - box1[1]
    w_box1 = box1[2] - box1[0]
    #print ('heatmap1',heatmap1.shape)

    # height_feature = bbox_npy[0].shape[0]
    # weight_feature = bbox_npy[0].shape[1]
    x_scale = w_box1 / w_heat1
    y_scale = h_box1 / h_heat1
    print ('y_scale,x_scale',y_scale,x_scale)
    #print('The x scale is {}, the y scale is {}'.format(x_scale, y_scale))
    for i in range(17):
        sorted_value = np.sort(heatmap1[i], axis=None)
        # print(sorted_value)
        value = sorted_value[-1]
        if value <0.3:
            continue
        
        # print(len(top10))
        index_list = []
        temp_index = np.where(heatmap1[i] == value) 
        y = int(temp_index[0][0] * y_scale)
        x = int(temp_index[1][0] * x_scale)
        print (value,x,y,y_scale,temp_index)
        # if y > h_image:
            # print ('???')
            # y = int(h_box1)
        # if x > w_image:
            # x = int(w_box1)
        y = (y + y_min)#/h_image
        x = (x + x_min)#/w_image
        if x >w_image or y>h_image:
            print (file_id,'  erro')
        image = cv2.rectangle(image,(x-25,y-25),(x+25,y+25),(255,0,0),2)
    cv2.imwrite(str(tag)+file_name,image)

    return x,y




heatmap_dir = '/export/home/zm/test/icme2019/pose_embedding/PIPA_17/'
graph_dir = '/export/home/zm/test/icme2019/pygcn/graph/'
reader = open('/export/home/zm/test/icme2019/SR_graph/list/PIPA_relation_test_zm.txt')
lines = reader.readlines()

box1s = []
box2s = []
count = 0
for line in lines[900:]:
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
    heatmap1 = np.load(heatmap_dir + bbox1_name)
    
    bbox2_name = file_id + '_%d_%d.npy'%(box2[0],box2[1])
    heatmap2 = np.load(heatmap_dir + bbox2_name)
    
    graph = find(heatmap1,box1,file_name,count)
    graph = find(heatmap2,box2,file_name,count+10)
    if count ==10:
        break
    
    # np.savetxt('a.txt',graph,fmt='%d')
    # graph_name =file_id + '_%d_%d_%d_%d.npy'%(box1[0],box1[1],box2[0],box2[1])
    # np.save(graph_dir + graph_name,graph )














