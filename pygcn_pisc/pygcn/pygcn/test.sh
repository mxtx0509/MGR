CUDA_VISIBLE_DEVICES=2 python -u test.py   --feature_dir '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/' --weights './models_new/per_obj_graph/28_checkpoint_ep28.pth.tar'  --result_path './result/per_obj_graph/28/'

# CUDA_VISIBLE_DEVICES=2 python -u test.py   --feature_dir '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/' --weights './models_new/fc_per_obj/28_checkpoint_ep28.pth.tar'  --result_path './result/fc_per_obj/28/'

# CUDA_VISIBLE_DEVICES=2 python -u test.py   --feature_dir '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/' --weights './models_new/fc_per_obj/100_checkpoint_ep100.pth.tar'  --result_path './result/fc_per_obj/100/'

# CUDA_VISIBLE_DEVICES=2 python -u test.py   --feature_dir '/export/home/zm/test/icme2019/SR_graph/feature/PISC_fine_32/' --weights './models_new/fc_per_obj/160_checkpoint_ep160.pth.tar'  --result_path './result/fc_per_obj/160/'