import sys
import argparse
from pathlib import Path

sys.path.append(".")
from dataset.preprocessing.preprocess_scannet import (
    calculate_iou_folders, 
    calculate_panoptic_quality_folders,
    calculate_iou_folders_MOS, 
    calculate_panoptic_quality_folders_MOS_with_Instance_Number
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metrics')
    parser.add_argument('--root_path', required=False)
    parser.add_argument('--exp_path', required=False)
    # add flag
    parser.add_argument('--MOS', action='store_true') # evaluating MOS dataset
    parser.add_argument('--scene', required=False)
    args = parser.parse_args()
    
    scene = args.scene
    # print('calculating metrics for ours')
    image_dim = (512, 512)
    if not args.MOS:
        iou = calculate_iou_folders(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "rs_semantics", image_dim)
        instance_number_GT, instance_number_pred = -1,-1
        pq, rq, sq, instance_number_GT,instance_number_pred = calculate_panoptic_quality_folders(
            Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"),
            Path(args.root_path) / "rs_semantics", Path(args.root_path) / "rs_instance", image_dim)
    else:
        iou = calculate_iou_folders_MOS(Path(args.exp_path, "pred_semantics"), Path(args.root_path) / "semantic", image_dim)
        pq, rq, sq, instance_number_GT,instance_number_pred= calculate_panoptic_quality_folders_MOS_with_Instance_Number(
            Path(args.exp_path, "pred_semantics"), Path(args.exp_path, "pred_surrogateid"),
            Path(args.root_path) / "semantic", Path(args.root_path) / "instance", image_dim)
    # print(f'[dataset] iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')
    # write metrics to file
    # with open(Path(args.exp_path, "metrics.txt"), "w") as f:
        # f.write(f'iou, pq, sq, rq: {iou:.3f}, {pq:.3f}, {sq:.3f}, {rq:.3f}')
    # print(f'{scene}, {iou:.3f}, {pq:.3f}, {sq*100:.3f}, {rq:.3f}, {instance_number_GT}, {instance_number_pred}')
    # print(f'{scene},  {pq * 100:.1f}')
    print(f'{scene}, {iou*100:.1f}, {pq*100:.1f},{sq*100:.1f}, {rq*100:.1f}, {instance_number_GT}, {instance_number_pred}')

    