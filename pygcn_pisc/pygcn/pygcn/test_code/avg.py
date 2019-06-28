import numpy as np
import os
image_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_256/'
new_dir = '/export/home/zm/test/icme2019/pose_embedding/PISC_256_1/'
fea_list = os.listdir(image_dir)
count = 0
for file in fea_list:
    count = count +1 
    if count%1000==0:
        print (count)
    fea = np.load(image_dir + file)
    fea = np.mean(fea,axis=1)
    fea = np.mean(fea,axis=1)
    np.save(new_dir+file,fea)







