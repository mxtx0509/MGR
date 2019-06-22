#!/bin/bash

# Path to Images
ImagePath="/export/home/zm/dataset/PIPA/"
# Path to object boxes
ObjectsPath="./objects/PISC_objects/faster_bboxes_categories_t3/"
# Path to test list
TrainList="./list/PIPA_relation_train_zm.txt"
TestList="./list/PIPA_relation_test_zm.txt"

# Path to adjacency matrix
AdjMatrix="./adjacencyMatrix/PISC_fine_level_matrix.npy"
Weights_zm=""

# Number of classes
num=16
# Path to save scores
ResultPath="./result/"
# Path to model
ModelPath=""


CUDA_VISIBLE_DEVICES=0,1 python -u main.py $ImagePath $ObjectsPath $TrainList $TestList --adjacency_matrix $AdjMatrix -n $num --batch_size 256 --val_step 1 --write-out --start_epoch 0  --save_dir './models/layer4/' --weights '/export/home/zm/test/icme2019/SR_PIPA/models/fc/12_60000_checkpoint_ep12.pth.tar'
