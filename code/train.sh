
## Training on MOS
# stage 1, only optimize the mean part
CUDA_VISIBLE_DEVICES=0   python trainer/train_panopli_tensorf.py template.dataset_root=data/old_room_25 
## stage 2, optimize the mean + covariance part
CUDA_VISIBLE_DEVICES=0   python trainer/train_panopli_tensorf_learned_covariance_v2_smaller_clamp.py template.dataset_root=data/old_room_25 
## stage 3 (optional), if you want to adopt the cross-view constraint.
CUDA_VISIBLE_DEVICES=0   python trainer/train_panopli_tensorf_learned_covariance_v2_smaller_clamp_cross_constraint.py template.dataset_root=data/old_room_25 




