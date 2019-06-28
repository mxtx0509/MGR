import os
import numpy as np
np.set_printoptions(threshold=np.nan, linewidth=220)

text_loaction = './PISC_fine_all.txt'


class FindLocation(object):
    def __init__(self, npy_loaction):
        self.npy_loaction = npy_loaction

    def find(self, bbox_info):

        """
        top-10's location  in the original pic & value of them in the feature map

        :param bbox_info: The information of the bbox, containing the npy file name and the size of the bbox
        :return: dict = {bbox_name:{channel_no:[(x1,y1,c10),....(x10,y10,c10)]}}
        """
        result = {}
        feature_path = self.npy_loaction + bbox_info[0]
        bbox_npy = np.load(feature_path)
        # print(bbox_info[0])
        # print(bbox_info[0].split('_')[1])
        x_min = int(bbox_info[0].split('_')[1])
        y_min = int(bbox_info[0].split('_')[2].split('.')[0])
        # print(x_min)
        # print(y_min)
        # print(bbox_info[1])
        # print(bbox_npy[0].shape)
        height_original = bbox_info[1][1]
        weight_original = bbox_info[1][0]

        height_feature = bbox_npy[0].shape[0]
        weight_feature = bbox_npy[0].shape[1]
        x_scale = weight_original / weight_feature
        y_scale = height_original / height_feature
        print('The x scale is {}, the y scale is {}'.format(x_scale, y_scale))
        for i in range(17):
            sorted_value = np.sort(bbox_npy[i], axis=None)
            # print(sorted_value)
            top10 = sorted_value[-10:]
            # print(len(top10))
            index_list = []
            for value in top10:
                temp_index = np.where(bbox_npy[i] == value)  # the index is (y,x)
                y = int(temp_index[0] * y_scale)
                x = int(temp_index[1] * x_scale)

                if y > height_original:
                    y = int(height_original)
                if x > weight_original:
                    x = int(weight_original)
                y = y + y_min
                x = x + x_min
                # print('The o is {} and {}'.format(x, y))
                index_list.append((y,x,value))
            # print('The channel is {} and the index is {}'.format(i, index_list))
            result[i] = index_list
        # print('The result is {}'.format(result))

        return result


if __name__ == '__main__':
    fl = FindLocation(npy_loaction='./')

    npy_name_format = '{}_{}_{}.npy'
    with open(text_loaction, 'r') as f:
        bbox_names = f.readlines()

    bbox_list = []
    for bbox in bbox_names:
        elements = bbox.split()
        image_id = elements[0].split('.')[0]
        bbox1_npy = npy_name_format.format(image_id, elements[1], elements[2])
        bbox1_size_w = int(elements[3]) - int(elements[1])
        bbox1_size_h = int(elements[4]) - int(elements[2])
        bbox1_size = (bbox1_size_w, bbox1_size_h)
        bbox2_npy = npy_name_format.format(image_id, elements[5], elements[6])
        bbox2_size_w = int(elements[7]) - int(elements[5])
        bbox2_size_h = int(elements[8]) - int(elements[6])
        bbox2_size = (bbox2_size_w, bbox2_size_h)
        # print('The bbox is {}, {}'.format(bbox1_npy,  bbox2_npy))
        bbox_list.append((bbox1_npy, bbox1_size))
        bbox_list.append((bbox2_npy, bbox2_size))
        # print('The bbox_list is {}'.format(bbox_list))

    result = fl.find(bbox_list[0])
    print(result)








