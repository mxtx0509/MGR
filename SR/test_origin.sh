#!/bin/bash

# Path to Images
ImagePath="/export/home/zm/dataset/PIPA/train/"
# Path to object boxes
ObjectsPath="./objects/PISC_objects/faster_bboxes_categories_t3/"  
# Path to test list
TestList="./list/PIPA_relation_train_zm.txt"  
# Path to adjacency matrix
AdjMatrix="./adjacencyMatrix/PISC_fine_level_matrix.npy"
# Number of classes
num=16
# Path to save scores
ResultPath="./result/"
# Path to model
ModelPath="./models/layer4/model_best.pth.tar"
echo $ModelPath

CUDA_VISIBLE_DEVICES=2 python -u extract_fea.py $ImagePath $ObjectsPath $TestList --weight $ModelPath --adjacency-matrix $AdjMatrix -n $num -b 1 --print-freq 100 --write-out --result-path $ResultPath --num_classes 16 --feature True

