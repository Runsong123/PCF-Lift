# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import sys
import time
from pathlib import Path
import pickle
from typing import Any
import torch
import omegaconf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MeanShift
from scipy.stats import gaussian_kde
from hdbscan import HDBSCAN

sys.path.append("/research/d1/gds/rszhu22/Contrastive-Lift/")
from dataset import PanopLiDataset, create_segmentation_data_panopli
from dataset.many_object_scenes import MOSDataset
from model.radiance_field.tensoRF import TensorVMSplit, MLPRenderInstanceFeature
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict
from util.points_util import savePlyFromPtsRGB
from datetime import datetime
import glob
from scipy.spatial import distance
import os
import torch.nn.functional as F

def render_panopli_checkpoint(
        config: Any,
        trajectory_name: Any,
        test_only: bool = False,
        output_dir: Any = None,
        feature_dimension: int = 4,
        min_covariance: float =0.1,
        debug: bool= False,
):
    
    output_dir = Path(output_dir)
    print(output_dir)
    (output_dir).mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    if config.dataset_class == "panopli":
        test_set = PanopLiDataset(
            Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics',
            instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
            create_seg_data_func=create_segmentation_data_panopli,
            subsample_frames=config.subsample_frames
        )
    elif config.dataset_class == "mos" or config.dataset_class == "multi_frame_mos":
        test_set = MOSDataset(
            Path(config.dataset_root), "test_on_training", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='detic_semantic',
            instance_dir='detic_instance', instance_to_semantic_key=None,
            create_seg_data_func=None, subsample_frames=config.subsample_frames
        )

    H, W, alpha = config.image_dim[0], config.image_dim[1], 0.65
    # whether to render the test set or a predefined trajectory through the scene
    if test_only:
        trajectory_set = test_set
    else:
        trajectory_set = test_set.get_trajectory_set(trajectory_name, True)
    trajectory_loader = DataLoader(trajectory_set, shuffle=False, num_workers=0, batch_size=1)
    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    total_classes = len(test_set.segmentation_data.bg_classes) + len(test_set.segmentation_data.fg_classes)
    output_mlp_semantics = torch.nn.Identity() if config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
    model = TensorVMSplit(
        [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
        num_semantics_comps=(32, 32, 32), num_instance_comps=(32, 32, 32), num_semantic_classes=total_classes,
        dim_feature_instance=2*config.max_instances if config.instance_loss_mode=="slow_fast" else config.max_instances,
        output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=config.use_mlp_for_semantics,  
        use_instance_mlp=config.use_mlp_for_instances,
        use_distilled_features_semantic=config.use_distilled_features_semantic, use_distilled_features_instance=config.use_distilled_features_instance,
        pe_sem=config.pe_sem, pe_ins=config.pe_ins,
        slow_fast_mode=config.instance_loss_mode=="slow_fast", use_proj=config.use_proj,
    )
    renderer = TensoRFRenderer(
        test_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
        semantic_weight_mode=config.semantic_weight_mode
    )
    renderer.load_state_dict(get_parameters_from_state_dict(state_dict, "renderer"))
    for epoch in config.grid_upscale_epochs[::-1]:
        if checkpoint['epoch'] >= epoch:
            model.upsample_volume_grid(renderer.grid_dim)
            renderer.update_step_size(renderer.grid_dim)
            break

    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))

    model = model.to(device)
    renderer = renderer.to(device)

    # disable this for fast rendering (just add more steps along the ray)
    renderer.update_step_ratio(renderer.step_ratio * 0.5)

    all_points_rgb, all_points_semantics, all_points_instances, all_points_depth = [], [], [], []
    all_instance_features, all_thing_features, all_slow_features = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(trajectory_loader)):
            if debug and batch_idx>=10:
                break
            # if batch_idx>=10:
            #     break
            batch['rays'] = batch['rays'].squeeze(0).to(device)
            concated_outputs = []
            outputs = []
            # infer semantics and surrogate ids
            for i in range(0, batch['rays'].shape[0], config.chunk):
                out_rgb_, out_semantics_, out_instances_, out_depth_, _, _ = renderer(
                    model, batch['rays'][i: i + config.chunk], config.perturb, test_set.white_bg, False)
                outputs.append([out_rgb_, out_semantics_, out_instances_, out_depth_])
            for i in range(len(outputs[0])):
                concated_outputs.append(torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0))
            p_rgb, p_semantics, p_instances, p_dist = concated_outputs
            p_depth = distance_to_depth(test_set.intrinsics[0], p_dist.view(H, W))

            if False: ## we don't need position information
                points_xyz = batch['rays'][...,0:3] + p_dist[...,None] * batch['rays'][...,3:6] # B x 3
                # p_instances = p_instances + points_xyz

            if model.slow_fast_mode:
                slow_features = p_instances[...,config.max_instances:] 
                all_slow_features.append(slow_features)
                p_instances = p_instances[...,0:config.max_instances] # keep fast features only

            # p_instances = torch.cat([p_instances,points_xyz], axis=-1) we don't need position information
            all_instance_features.append(p_instances)

            all_points_rgb.append(p_rgb)
            all_points_semantics.append(p_semantics)
            all_points_depth.append(p_depth)

            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, test_set.segmentation_data.fg_classes)
            p_instances = p_instances[:,:feature_dimension]
            all_thing_features.append(p_instances)

    all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
    # np.save(output_dir / "instance_features.npy", all_instance_features)

    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
    np.save(output_dir / "thing_features.npy", all_thing_features)

    if model.slow_fast_mode:
        all_slow_features = torch.cat(all_slow_features, dim=0).cpu().numpy()
        # np.save(output_dir / "slow_features.npy", all_slow_features)
    

    ############### NMS part
    all_thing_features = np.load(output_dir / "thing_features.npy")
    os.remove(output_dir / "thing_features.npy")
    feature = all_thing_features.reshape(-1, config.image_dim[0]*config.image_dim[1], feature_dimension) ## N * H * W * feature_dimension
    print(feature.shape)

    pseudo_label_path = sorted(glob.glob(f"{config.dataset_root}/detic_instance/*.npy"))
    len_ratio = len(pseudo_label_path) * 0.8

    sample_indices = list(range(len(pseudo_label_path)))
    # use self.random_train_val_ratio to select last 20% as test set
    # this works because the frames were generated at random to begin with
    # also, always using last 20% means the test set is deterministic and fixed for all experiments
    indices = sample_indices[:int(len(pseudo_label_path) * 0.8)]
    pseudo_label_path = [pseudo_label_path[i] for i in indices]
    print(len(pseudo_label_path))
    # print()
    # exit()
    pseudo_label = []
    for pseudo_label_item in pseudo_label_path:
        pseudo_item  = read_and_resize_labels_npy(pseudo_label_item, config.image_dim)
        # print(pseudo_item.shape)
        # exit()
        pseudo_label.append(pseudo_item)


    data_association(feature, pseudo_label,output_dir,min_covariance)



def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances

def calculate_cross_distribution_distance_large_split(distribution_1, distribution_2, min_covariance):
    ##### convert numpy to pytorch
    # distribution_1 = torch.from_numpy(distribution_1).cuda()
    # distribution_2 = torch.from_numpy(distribution_2).cuda()

    print(min_covariance)
    
    distribution_1 = distribution_1.unsqueeze(1)
    mean_1, covariance_1 = distribution_1[...,:3], distribution_1[...,3:]
    distribution_2 = distribution_2.unsqueeze(0)
    mean_2, covariance_2 = distribution_2[...,:3], distribution_2[...,3:]

    
    covariance_1 =  F.elu(covariance_1) + 1.0 + min_covariance
    covariance_2 =  F.elu(covariance_2) + 1.0 + min_covariance

    alpha_distribution = 4 * (covariance_1 + covariance_2)
    alpha_distribution = torch.clamp(alpha_distribution,1e-6,1e+6)
    distribution_dist = torch.sum(((mean_1- mean_2)**2)/(alpha_distribution), dim=-1)
    distribution_dist = torch.exp(-1 * distribution_dist)
    
    stand_covarance_1 = torch.sqrt(covariance_1)
    stand_covarance_2 = torch.sqrt(covariance_2)

    beta_distribtion = stand_covarance_2/stand_covarance_1 +  stand_covarance_1/stand_covarance_2
    beta_distribtion = beta_distribtion/2
    # beta_distribtion = torch.cumprod(beta_distribtion/2, dim=-1).clamp(1e-6, 1e+6)
    beta_distribtion = torch.clamp(torch.clamp(beta_distribtion[...,0] * beta_distribtion[...,1],1e-6,1e+6) * beta_distribtion[...,2],1e-6,1e+6)
    beta_distribtion = 1.0/torch.sqrt(beta_distribtion)

    return (distribution_dist * beta_distribtion)

def calculate_cross_distribution_distance_large(distribution_1, distribution_2, min_covariance):
    
    ##### convert numpy to pytorch
    distribution_1 = torch.from_numpy(distribution_1)
    distribution_2 = torch.from_numpy(distribution_2)

    
    chunksize = 1000
    distances = torch.zeros((distribution_1.shape[0], distribution_1.shape[0])).cuda()
    for i in range(0, distribution_1.shape[0], chunksize):
        distances[i:i+chunksize] = calculate_cross_distribution_distance_large_split(
            torch.FloatTensor(distribution_1[i:i+chunksize]).cuda(),
            torch.FloatTensor(distribution_2).cuda(),
            min_covariance
        )
    #
    return distances.cpu().numpy()

# def calculate_cross_distribution_distance_chunk(distribution_1, distribution_2):

def calculate_cross_distribution_distance(distribution_1, distribution_2,min_covariance):
    
    ##### convert numpy to pytorch
    distribution_1 = torch.from_numpy(distribution_1)
    distribution_2 = torch.from_numpy(distribution_2)

    
    #
    
    # distances = torch.zeros((distribution_1.shape[0], distribution_1.shape[0])).cuda()
    distribution_1 = distribution_1.unsqueeze(1)
    mean_1, covariance_1 = distribution_1[...,:3], distribution_1[...,3:]

    # chunksize = 10**7
    
        # distribution_2 = distribution_2.reshape(-1, thing_cls_features.shape[-1])
    # for i in range(0, distribution_1.shape[0], chunksize):
    #     distances[i:i+chunksize] = torch.cdist(
    #         torch.FloatTensor(thing_cls_features_reshaped[i:i+chunksize]).to(device),
    #         torch.FloatTensor(centroids).to(device)
    #     )
        # thing_cls_all_labels = torch.argmin(distances, dim=-1).cpu().numpy()

    distribution_2 = distribution_2.unsqueeze(0)
    mean_2, covariance_2 = distribution_2[...,:3], distribution_2[...,3:]

    covariance_1 =  F.elu(covariance_1) + 1.0 + min_covariance
    covariance_2 =  F.elu(covariance_2) + 1.0 + min_covariance
    
    ########## fixed as 0.1
    # covariance_1  = covariance_1 * 0.0 + 0.1
    # covariance_2  = covariance_2 * 0.0 + 0.1

    alpha_distribution = 4 * (covariance_1 + covariance_2)
    alpha_distribution = torch.clamp(alpha_distribution,1e-6,1e+6)
    distribution_dist = torch.sum(((mean_1- mean_2)**2)/(alpha_distribution), dim=-1)
    distribution_dist = torch.exp(-1 * distribution_dist)
    
    stand_covarance_1 = torch.sqrt(covariance_1)
    stand_covarance_2 = torch.sqrt(covariance_2)

    beta_distribtion = stand_covarance_2/stand_covarance_1 +  stand_covarance_1/stand_covarance_2
    beta_distribtion = beta_distribtion/2
    # beta_distribtion = torch.cumprod(beta_distribtion/2, dim=-1).clamp(1e-6, 1e+6)
    beta_distribtion = torch.clamp(torch.clamp(beta_distribtion[...,0] * beta_distribtion[...,1],1e-6,1e+6) * beta_distribtion[...,2],1e-6,1e+6)
    beta_distribtion = 1.0/torch.sqrt(beta_distribtion)

    return (distribution_dist * beta_distribtion).numpy()

def calculate_cross_distribution_distance_half(distribution_1, distribution_2, min_covariance):
    
    ##### convert numpy to pytorch
    distribution_1 = torch.from_numpy(distribution_1)
    distribution_2 = torch.from_numpy(distribution_2)

    
    #
    
    # distances = torch.zeros((distribution_1.shape[0], distribution_1.shape[0])).cuda()
    distribution_1 = distribution_1.unsqueeze(1)
    mean_1, covariance_1 = distribution_1[...,:3], distribution_1[...,3:]

    # chunksize = 10**7
    
        # distribution_2 = distribution_2.reshape(-1, thing_cls_features.shape[-1])
    # for i in range(0, distribution_1.shape[0], chunksize):
    #     distances[i:i+chunksize] = torch.cdist(
    #         torch.FloatTensor(thing_cls_features_reshaped[i:i+chunksize]).to(device),
    #         torch.FloatTensor(centroids).to(device)
    #     )
        # thing_cls_all_labels = torch.argmin(distances, dim=-1).cpu().numpy()

    distribution_2 = distribution_2.unsqueeze(0)
    mean_2, covariance_2 = distribution_2[...,:3], distribution_2[...,3:]


    covariance_1 =  F.elu(covariance_1) + 1.0 + min_covariance
    covariance_2 =  F.elu(covariance_2) + 1.0 + min_covariance
    
    ########## fixed as 0.1
    # covariance_1  = covariance_1 * 0.0 + 0.1
    # covariance_2  = covariance_2 * 0.0 + 0.1

    alpha_distribution = 4 * (covariance_1 + covariance_2)
    alpha_distribution = torch.clamp(alpha_distribution,1e-6,1e+6)
    distribution_dist = torch.sum((((mean_1- mean_2)/2)**2)/(alpha_distribution), dim=-1)
    distribution_dist = torch.exp(-1 * distribution_dist)
    
    stand_covarance_1 = torch.sqrt(covariance_1)
    stand_covarance_2 = torch.sqrt(covariance_2)

    beta_distribtion = stand_covarance_2/stand_covarance_1 +  stand_covarance_1/stand_covarance_2
    beta_distribtion = beta_distribtion/2
    # beta_distribtion = torch.cumprod(beta_distribtion/2, dim=-1).clamp(1e-6, 1e+6)
    beta_distribtion = torch.clamp(torch.clamp(beta_distribtion[...,0] * beta_distribtion[...,1],1e-6,1e+6) * beta_distribtion[...,2],1e-6,1e+6)
    beta_distribtion = 1.0/torch.sqrt(beta_distribtion)

    return (distribution_dist * beta_distribtion).numpy()

# def calculate_distribtion_concentrate(class_feature,class_covariance):
def calculate_distribtion_concentrate(class_feature, class_covariance,min_covariance):
    
    # distribution_1 =
    mean_feature = np.mean(class_feature, axis=0, keepdims=True)
    mean_covariance = np.mean(class_covariance, axis=0, keepdims=True)



    ##### convert numpy to pytorch
    class_feature = torch.from_numpy(class_feature)
    class_covariance = torch.from_numpy(class_covariance)
    mean_feature = torch.from_numpy(mean_feature)
    mean_covariance = torch.from_numpy(mean_covariance)


    class_covariance =  F.elu(class_covariance) + 1.0 + min_covariance
    mean_covariance =  F.elu(mean_covariance) + 1.0 + min_covariance

    alpha_distribution = 4 * (mean_covariance + class_covariance)
    alpha_distribution = torch.clamp(alpha_distribution,1e-6,1e+6)
    distribution_dist = torch.sum(((class_feature - mean_feature)**2)/(alpha_distribution), dim=-1)
    distribution_dist = torch.exp(-1 * distribution_dist)
    
    stand_covarance_1 = torch.sqrt(mean_covariance)
    stand_covarance_2 = torch.sqrt(class_covariance)

    beta_distribtion = stand_covarance_2/stand_covarance_1 +  stand_covarance_1/stand_covarance_2
    beta_distribtion = beta_distribtion/2
    # beta_distribtion = torch.cumprod(beta_distribtion/2, dim=-1).clamp(1e-6, 1e+6)
    beta_distribtion = torch.clamp(torch.clamp(beta_distribtion[:,0] * beta_distribtion[:,1],1e-6,1e+6) * beta_distribtion[:,2],1e-6,1e+6)
    beta_distribtion = 1.0/torch.sqrt(beta_distribtion)

    return (distribution_dist * beta_distribtion).numpy()

    

def data_association(feature, pseudo_label, NMS_output_dir, min_covariance):
    print("min_covariance", min_covariance)
    # exit()
    start_time = time.time()

    all_prototype_feature = []
    all_similarity_per_image = []
    score = []
    for i in tqdm(range(feature.shape[0])):
        
        # 
        inst = pseudo_label[i]   # uint16 -> int32
        inst = inst.reshape(-1)

        all_subfeature = feature[i]
        
        
        semantic = all_subfeature[:,0]
        sub_feature = all_subfeature[:,1:4]
        sub_covariance = all_subfeature[:,4:]
        # sub_pts = all_subfeature[:,4:]

        batch_masks = []
        calculated_id = np.zeros_like(inst)
        
        
        prototype_feature_in_single_image = []
        for inst_id in np.unique(inst):
            inst_mask = (inst == inst_id).astype(np.int32) # instance mask
            inst_mask = inst_mask.reshape(-1)
            semantic_mask  =  (semantic == -float('inf')).astype(np.int32)
            inst_mask = (inst_mask * semantic_mask).astype(np.bool_)
            # instance_mask = 
            if (inst_mask.sum() < 10):
                continue
            class_feature = sub_feature[inst_mask,:]
            class_covariance = sub_covariance[inst_mask,:]
            
            # std = np.sqrt(((class_feature- np.mean(class_feature, axis=0, keepdims=True)) **2).mean())
            # cencentrate_dis = np.sum((class_feature- np.mean(class_feature, axis=0, keepdims=True)) **2, axis=1)
            distribution_similarity =  calculate_distribtion_concentrate(class_feature,class_covariance,min_covariance)
            # print(cencentrate_dis.shape)
            # concentrate_score = np.exp(-1 * cencentrate_dis).mean()
            # original 0.01 -> 0.99
            # concentrate_score = np.sum(distribution_similarity>0.99)/distribution_similarity.shape[0]
            concentrate_score = distribution_similarity.mean()

            ### here we assume that the semantic is the same: thing
            prototype_feature = np.mean(class_feature, axis=0)
            prototype_covariance = np.mean(class_covariance, axis=0)
            prototype_distribution = np.concatenate([prototype_feature,prototype_covariance], axis=-1)
            all_prototype_feature.append(prototype_distribution)
            score.append(concentrate_score)
            # print(prototype_feature.shape)

            prototype_feature_in_single_image.append(prototype_distribution)

        if len(prototype_feature_in_single_image)>0:
            prototype_feature_in_single_image = np.stack(prototype_feature_in_single_image)
            similarity_single_image = calculate_cross_distribution_distance_half(prototype_feature_in_single_image, prototype_feature_in_single_image, min_covariance)
            # distance_single_image = distance_single_image[~np.eye(distance_single_image.shape[0],dtype=np.bool_)]
            similarity_single_image -= np.eye(similarity_single_image.shape[0])
            max_similarity = np.max(similarity_single_image,axis=-1).mean()
            all_similarity_per_image.append(max_similarity)

    
    score = np.array(score)

    similarity_threshold = np.array(all_similarity_per_image).mean()
    print("similarity_threshold: ", similarity_threshold)


    all_prototype_feature = np.stack(all_prototype_feature)
    print(all_prototype_feature.shape)

    # overlapping
    similarity = calculate_cross_distribution_distance_large(all_prototype_feature, all_prototype_feature, min_covariance)
    find_centorid = []
    left = all_prototype_feature.shape[0]
    #########
    # NMS algorithm
    ##########
    while left!=0:

        max_index = np.argmax(score)
        if score[max_index]==0:
            break
        
        ## suppression by increase the cdist
        local_mask = similarity[max_index]>(similarity_threshold) # find their local

        save_item = np.zeros(7)
        save_item[:6] = all_prototype_feature[max_index]
        save_item[6:] = score[max_index]
        find_centorid.append(save_item)


        ###### suppression the local
        similarity[local_mask,:] = 0
        similarity[:,local_mask] = 0
        score[local_mask] = 0
        left -= sum(local_mask)
        print(left)

    find_centorid = np.stack(find_centorid)
    print("find_centorid.shape: ", find_centorid.shape)
    os.makedirs(f"{NMS_output_dir}", exist_ok=True)
    np.savetxt(f"{NMS_output_dir}/NMS_centroid.txt",find_centorid)
    # exit()

def read_and_resize_labels_npy(path, size):
    image = np.load(path)
    image = Image.fromarray(image.astype(np.int16))
    return np.array(image.resize((size[1],size[0]), Image.Resampling.NEAREST))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--render_trajectory", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_dimension", type=int, required=True)
    parser.add_argument("--subsample", type=int, default=1, required=False)
    parser.add_argument("--minimal_value", type=float, required=True)
    parser.add_argument('--debug', action='store_true',
                        help='visualized images will be saved when debug is True') # default is false
    args = parser.parse_args()

    # needs a predefined trajectory named trajectory_blender in case test_only = False
    cfg = omegaconf.OmegaConf.load(Path(args.ckpt_path).parents[1] / "config.yaml")
    cfg.resume = args.ckpt_path
    output_dir = args.output_dir
    TEST_MODE = not args.render_trajectory
    cfg.subsample_frames = args.subsample
    min_covariance = args.minimal_value
    debug = args.debug

    cfg.image_dim = [256, 384]    
    if isinstance(cfg.image_dim, int):
        cfg.image_dim = [cfg.image_dim, cfg.image_dim]

    render_panopli_checkpoint(
        cfg, "trajectory_blender", test_only=TEST_MODE,output_dir=output_dir,feature_dimension=args.feature_dimension,min_covariance=min_covariance,debug=debug)
