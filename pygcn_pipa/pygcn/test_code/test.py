import numpy as np
adj = np.load('/export/home/cjw/zm/test/cvpr2019/adj/adj_same_1/0000.npy')
D =np.zeros((40,40))
D_1 =np.zeros((40,40))
for i in range(40):
    tmp = adj[i,:]
    count = np.sum(tmp==1)
    number = count ** (-1/2)
    D[i,i] = count
    D_1[i,i] = number
print (D[0:20])
print (D_1[0:20])
# D = np.power(D, -(1/2))
# print (D[0:20])