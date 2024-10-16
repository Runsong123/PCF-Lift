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

sys.path.append("/research/d1/gds/rszhu22/Contrastive-Lift")
from dataset import PanopLiDataset, create_segmentation_data_panopli
from dataset.many_object_scenes import MOSDataset
from model.radiance_field.tensoRF import TensorVMSplit, MLPRenderInstanceFeature
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict
from util.points_util import savePlyFromPtsRGB
from datetime import datetime
import torch.nn.functional as F

def render_panopli_checkpoint(
        config: Any,
        trajectory_name: Any,
        test_only: bool = False,
        cached_centroids_path: Any = None,
        output_dir : Any = None,
        find_score: Any = None,
        min_covariance: float = 0.0,
        
):

    # if subpath is not None:
        # output_dir = output_dir / subpath
    output_dir = Path(output_dir)
    print(output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda:0")

    if config.dataset_class == "panopli":
        test_set = PanopLiDataset(
            Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]),
            config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics',
            instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
            create_seg_data_func=create_segmentation_data_panopli,
            subsample_frames=config.subsample_frames
        )
    elif config.dataset_class == "mos" or config.dataset_class == "multi_frame_mos" or config.dataset_class[:9]=="mos_noise" or config.dataset_class[:3] == "mos":
        test_set = MOSDataset(
            Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]),
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
        if checkpoint['epoch'] >= epoch or True:
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

            if config.use_delta:
                points_xyz = batch['rays'][...,0:3] + p_dist[...,None] * batch['rays'][...,3:6] # B x 3
                p_instances = p_instances + points_xyz

            if model.slow_fast_mode:
                slow_features = p_instances[...,config.max_instances:] 
                all_slow_features.append(slow_features)
                p_instances = p_instances[...,0:config.max_instances] # keep fast features only

            # p_instances = p_instances[:,:3]
            all_instance_features.append(p_instances)

            all_points_rgb.append(p_rgb)
            all_points_semantics.append(p_semantics)
            all_points_depth.append(p_depth)

            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, test_set.segmentation_data.fg_classes)
            all_thing_features.append(p_instances)

    all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
    # np.save(output_dir / "instance_features.npy", all_instance_features)

    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
    # np.save(output_dir / "thing_features.npy", all_thing_features)

    if model.slow_fast_mode:
        all_slow_features = torch.cat(all_slow_features, dim=0).cpu().numpy()
        # np.save(output_dir / "slow_features.npy", all_slow_features)

    use_cached_centroids = cached_centroids_path is not None
    if use_cached_centroids:
        # with open(cached_centroids_path, 'rb') as f:
        #     all_centroids = pickle.load(f)
        all_centroids = np.loadtxt(cached_centroids_path)
        # find_score
        mask = all_centroids[:,-1]>=find_score
        select_centroid = all_centroids[mask,:6]
        all_centroids = torch.from_numpy(select_centroid).float()
        all_points_instances,_ = cluster_segmentwise(H,W,test_set,
                all_points_rgb, all_points_depth,output_dir,test_only,all_centroids,
                all_thing_features, all_points_semantics, device, num_images=len(all_points_rgb),min_covariance=min_covariance
            )


def calculate_cross_distribution_distance(distribution_1, distribution_2,min_covariance):
    
    ##### convert numpy to pytorch
    # distribution_1 = torch.from_numpy(distribution_1)
    # distribution_2 = torch.from_numpy(distribution_2)

    
    #
    distribution_1 = distribution_1.unsqueeze(1)
    mean_1, covariance_1 = distribution_1[...,:3], distribution_1[...,3:]
    distribution_2 = distribution_2.unsqueeze(0)
    mean_2, covariance_2 = distribution_2[...,:3], distribution_2[...,3:]

    # min_covariance = 0.1
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

    return (distribution_dist * beta_distribtion)


def cluster_segmentwise(H, W,test_set,all_points_rgb, all_points_depth,output_dir,test_only,all_centroids,
 all_thing_features, all_points_semantics_origin, device, num_images=None,min_covariance=-0.0):
    print("min_covariance", min_covariance)
    all_points_semantics = torch.cat(all_points_semantics_origin, dim=0).argmax(dim=-1).cpu().numpy()

    thing_mask = all_thing_features[...,0] == -float('inf')
    features = all_thing_features[thing_mask]
    features = features[:,1:]
    all_thing_features = all_thing_features[:,1:]

    thing_semantics = all_points_semantics[thing_mask]
    thing_classes = np.unique(thing_semantics)



    all_labels = np.zeros(all_thing_features.shape[0], dtype=np.int32)
    all_thing_labels = np.zeros(features.shape[0], dtype=np.int32)
    max_label = 0
    # all_centroids = []
    for thing_cls in thing_classes:
        thing_cls_mask = thing_semantics == thing_cls
        thing_cls_features = features[thing_cls_mask] # features of this thing class

    
        centroids = all_centroids
        similarity = torch.zeros((thing_cls_features.shape[0], centroids.shape[0]), device=device)
        chunksize = 10**5
        thing_cls_features_reshaped = thing_cls_features.reshape(-1, thing_cls_features.shape[-1])
        for i in range(0, thing_cls_features.shape[0], chunksize):
            # distances[i:i+chunksize] = torch.cdist(
            #     torch.FloatTensor(thing_cls_features_reshaped[i:i+chunksize]).to(device),
            #     torch.FloatTensor(centroids).to(device)
            # )
            similarity[i:i+chunksize] = calculate_cross_distribution_distance(
                torch.FloatTensor(thing_cls_features_reshaped[i:i+chunksize]).to(device),
                torch.FloatTensor(centroids).to(device),
                min_covariance
            )
        thing_cls_all_labels = torch.argmax(similarity, dim=-1).cpu().numpy()
        # assign labels
        # if thing_cls_all_labels=-1, keep it as -1
        # else add max_label and assign it to thing_cls_all_labels
        thing_cls_all_labels[thing_cls_all_labels != -1] += max_label
        if np.any(thing_cls_all_labels != -1): # i.e. if there are clusters
            max_label = thing_cls_all_labels.max() + 1
        all_thing_labels[thing_cls_mask] = thing_cls_all_labels

    all_labels[thing_mask] = all_thing_labels
    all_labels[~thing_mask] = -1 # assign -1 to stuff points
    all_labels = all_labels + 1 # -1,0,...,K-1 -> 0,1,...,K
    # num_unique_labels = np.unique(all_labels).shape[0] 
    # NOTE: the above line has a problem when there is no stuff class (i.e. all_labels > 0)
    num_unique_labels = all_labels.max() + 1 # 0,1,...,K
    print("Num unique labels: ", num_unique_labels)


    all_labels = all_labels.reshape(num_images,-1)
        # save outputs
    (output_dir / "vis_semantics_and_surrogate").mkdir(exist_ok=True)
    (output_dir / "pred_semantics").mkdir(exist_ok=True)
    (output_dir / "pred_surrogateid").mkdir(exist_ok=True)
    for i, _ in enumerate(all_points_rgb):

        label_per_img = all_labels[i]
        all_labels_onehot = np.zeros((label_per_img.shape[0], num_unique_labels))
        all_labels_onehot[np.arange(label_per_img.shape[0]), label_per_img] = 1
        # all_labels_onehot = all_labels_onehot.reshape(num_images, -1, num_unique_labels)
        all_points_instances = torch.from_numpy(all_labels_onehot).to(device)

        # 
        name = f"{test_set.all_frame_names[test_set.val_indices[i]]}.png" if test_only else f"{i:04d}.png"
        p_rgb, p_semantics, p_instances, p_depth = all_points_rgb[i], all_points_semantics_origin[i], all_points_instances, all_points_depth[i]
        stack = visualize_panoptic_outputs(
            p_rgb, p_semantics, p_instances, p_depth, None, None, None,
            H, W, thing_classes=test_set.segmentation_data.fg_classes, visualize_entropy=False
        )
        output_semantics_with_invalid = p_semantics.detach().argmax(dim=1)
        grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=5).permute((1, 2, 0)).contiguous()
        grid = (grid * 255).cpu().numpy().astype(np.uint8)

        Image.fromarray(grid).save(output_dir / "vis_semantics_and_surrogate" / name)
        Image.fromarray(output_semantics_with_invalid.reshape(H, W).cpu().numpy().astype(np.uint8)).save(output_dir / "pred_semantics" / name)
        Image.fromarray(p_instances.argmax(dim=1).reshape(H, W).cpu().numpy().astype(np.uint16)).save(output_dir / "pred_surrogateid" / name)

    
    return all_labels_onehot, all_centroids






def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--render_trajectory", action="store_true")
    parser.add_argument("--subsample", type=int, default=1, required=False)
    parser.add_argument("--cached_centroids_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--find_score", type=float, required=True)
    parser.add_argument("--minimal_value", type=float, required=True)
    args = parser.parse_args()

    # needs a predefined trajectory named trajectory_blender in case test_only = False
    cfg = omegaconf.OmegaConf.load(Path(args.ckpt_path).parents[1] / "config.yaml")
    cfg.resume = args.ckpt_path
    TEST_MODE = not args.render_trajectory
    cfg.subsample_frames = args.subsample
    min_covariance = args.minimal_value

    
    print(args.find_score)
    cfg.image_dim = [256, 384]    
    if isinstance(cfg.image_dim, int):
        cfg.image_dim = [cfg.image_dim, cfg.image_dim]

    render_panopli_checkpoint(
        cfg, "trajectory_blender", test_only=TEST_MODE,
        cached_centroids_path=args.cached_centroids_path,
        output_dir=args.output_dir,
        find_score=args.find_score,
        min_covariance=min_covariance)
