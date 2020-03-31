CUDA_VISIBLE_DEVICES=1 python -u main_pose.py --save_dir './model_final/1000_pose_32wd_1/' \
                                              --val_step 4 \
                                              --batch_size 64 \
                                              --print_freq 16 \
                                              --num_class 6 \
                                              --train_list '/export/home/zm/test/icme2019/SR_graph/list/PISC_fine_train.txt' \
                                              --test_list '/export/home/zm/test/icme2019/SR_graph/list/PISC_fine_test.txt' \
                                              --fea_obj_dir '/export/home/zm/test/icme2019/objects/PISC_obj_embedding/' \
                                              --fea_person_dir '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/' \
                                              --fea_pose_dir '/export/home/zm/test/icme2019/pose_embedding/PISC_location_90/' \
                                              --graph_perobj_dir '/export/home/zm/test/icme2019/pygcn/graph/' \
                                              --graph_pose_dir '/export/home/zm/test/icme2019/pygcn/graph_pose/'