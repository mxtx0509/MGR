# import os

# import glob 
# list_path = '/export/home/cjw/zm/crop/'

# writer = open('list_obj.txt','w')
# path_list = glob.glob(list_path+'*/*.jpg')
# path_list.sort()
# for path in path_list:
    # path = path.strip(list_path)
    # writer.write(path+'\n')



import os
import random
import glob 
w_test = open('test_list_lable.txt','w')
w_train = open('train_list_lable.txt','w')

reader_test = open('test_list.txt','r')
reader_train = open('train_list.txt','r')
test_list = reader_test.readlines()
train_list = reader_train.readlines()
total_list=[]
train_id=[]
test_id=[]
for path in test_list:
    test_id.append(path.strip())
for path in train_list:
    train_id.append(path.strip())

lable_reader = open('label2-final.txt')
lable_list = lable_reader.readlines()
for path in lable_list:
    id_path=path.split(' ')[0].strip('.mp4').strip('.mkv').strip('.avi')
    if id_path in test_id:
        w_test.write(path)
    if id_path in train_id:
        w_train.write(path)











