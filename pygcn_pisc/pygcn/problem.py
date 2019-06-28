import numpy as np
list1 = [0,0,0,0]
list2 = [1,2,3,4]
list3 = [5,6,7,8]
list1[0] = list2[1]
list1[1] = list2[0]
list1[2] = list2[3]
list1[3] = list2[2]

print list1

list1[0] = list3[1]
list1[1] = list3[0]
list1[2] = list3[3]
list1[3] = list3[2]

print list1
