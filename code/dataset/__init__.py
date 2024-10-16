# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from pathlib import Path
from dataset.base import BaseDataset, create_segmentation_data_sem, InconsistentBaseDataset, InconsistentSingleBaseDataset
from dataset.many_object_scenes import MOSDataset, InconsistentMOSSingleDataset, SegmentMOSDataset, InconsistentMOSSingleDataset_Multiframe

def get_dataset(config, load_only_val=False, use_gt_inssem=False):
    print(config.dataset_class)
    if config.dataset_class == "mos" or config.dataset_class == "multi_frame_mos" or config.dataset_class == "multi_frame_mos_debug": # Many-Object-Scenes
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        train_set = None
        if not load_only_val:
            train_set = MOSDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                   load_feat=False, feature_type=None, # features not used for MOS
                                   instance_dir=instance_dir, instance_to_semantic_key=None,
                                   create_seg_data_func=None, subsample_frames=config.subsample_frames)
        val_set = MOSDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                             instance_dir=instance_dir, instance_to_semantic_key=None, create_seg_data_func=None,
                             subsample_frames=config.subsample_frames)
        return train_set, val_set
    elif config.dataset_class[:9]=="mos_noise":
        noise_ratio = config.dataset_class[10:]
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = f'detic_instance_windows_noise_{noise_ratio}', 'detic_semantic'
        train_set = None
        if not load_only_val:
            train_set = MOSDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                   load_feat=False, feature_type=None, # features not used for MOS
                                   instance_dir=instance_dir, instance_to_semantic_key=None,
                                   create_seg_data_func=None, subsample_frames=config.subsample_frames)
        val_set = MOSDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                             instance_dir=instance_dir, instance_to_semantic_key=None, create_seg_data_func=None,
                             subsample_frames=config.subsample_frames)
        return train_set, val_set
    elif config.dataset_class[:8] == "mos_open": # Many-Object-Scenes
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = f'detic_instance_{config.dataset_class[4:]}', f'detic_semantic_{config.dataset_class[4:]}'
        train_set = None
        if not load_only_val:
            train_set = MOSDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                                   load_feat=False, feature_type=None, # features not used for MOS
                                   instance_dir=instance_dir, instance_to_semantic_key=None,
                                   create_seg_data_func=None, subsample_frames=config.subsample_frames)
        val_set = MOSDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir=semantics_dir,
                             instance_dir=instance_dir, instance_to_semantic_key=None, create_seg_data_func=None,
                             subsample_frames=config.subsample_frames)
        return train_set, val_set
    raise NotImplementedError


def get_inconsistent_single_dataset(config, use_gt_inssem=False):
    if config.dataset_class == "mos":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return InconsistentMOSSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    elif config.dataset_class[:9]=="mos_noise":
        noise_ratio = config.dataset_class[10:]
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = f'detic_instance_windows_noise_{noise_ratio}', 'detic_semantic'
        return InconsistentMOSSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    elif config.dataset_class[:8] == "mos_open": # Many-Object-Scenes
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = f'detic_instance_{config.dataset_class[4:]}', f'detic_semantic_{config.dataset_class[4:]}'
        return InconsistentMOSSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    raise NotImplementedError

def get_inconsistent_single_dataset_two_frame(config, use_gt_inssem=False):
    if config.dataset_class == "multi_frame_mos":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return InconsistentMOSSingleDataset_Multiframe(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    elif config.dataset_class == "mos":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return InconsistentMOSSingleDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                            max_rays=config.max_rays_instances, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                            create_seg_data_func=None, subsample_frames=config.subsample_frames)
    raise NotImplementedError


def get_segment_dataset(config, use_gt_inssem=False):

    if config.dataset_class == "mos" or config.dataset_class == "multi_frame_mos" or config.dataset_class[:9]=="mos_noise" or config.dataset_class == "multi_frame_mos_debug":
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            instance_dir, semantics_dir = 'detic_instance', 'detic_semantic'
        return SegmentMOSDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                 max_rays=config.max_rays_segments, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                 create_seg_data_func=None, subsample_frames=config.subsample_frames)
    elif config.dataset_class[:8] == "mos_open": # Many-Object-Scenes
        if use_gt_inssem:
            instance_dir, semantics_dir = 'instance', 'semantic'
        else:
            # instance_dir, semantics_dir = 'detic_instance_open_LIVS_BSC', 'detic_semantic_open_LIVS_BSC'
            instance_dir, semantics_dir = f'detic_instance_{config.dataset_class[4:]}', f'detic_semantic_{config.dataset_class[4:]}'
        return SegmentMOSDataset(Path(config.dataset_root), "train", (128, 128), config.max_depth, overfit=config.overfit,
                                 max_rays=config.max_rays_segments, semantics_dir=semantics_dir, instance_dir=instance_dir, instance_to_semantic_key=None,
                                 create_seg_data_func=None, subsample_frames=config.subsample_frames)
    raise NotImplementedError
