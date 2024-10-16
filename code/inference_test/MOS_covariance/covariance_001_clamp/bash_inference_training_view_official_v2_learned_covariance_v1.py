
import glob
import os
import subprocess
import time
import numpy as np
import argparse

def get_all_dataset():
    all_dataset_1 = ["large_corridor"] *4
    number =[25, 50, 100, 500]
    all_dataset_1 = [data_name+"_"+str(num) for data_name, num in zip(all_dataset_1, number)]

    all_dataset_2 = ["old_room"] *4
    number =[25, 50, 100, 500]
    all_dataset_2 = [data_name+"_"+str(num) for data_name, num in zip(all_dataset_2, number)]
    all_data = all_dataset_1 + all_dataset_2
    return all_data

def getlist_number(filename, column=1):
    import pandas as pd
    data = pd.read_csv(filename)
    return data.iloc[:,column].values.tolist() ## scene, best_bd, select the 1st column

def get_last_value(filename, column=1):
    import pandas as pd
    data = pd.read_csv(filename)
    return data.iloc[:,column].values.tolist()[-1] ## scene, best_bd, select the 1st column

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--export_table_name", type=str, required=True)
    parser.add_argument("--feature_dimension", type=int, required=True)
    
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    feature_dimension = args.feature_dimension
    export_table_name = args.export_table_name


    minimal_value_list = [0.01, 0.01, 0.1, 0.01, 0.01, 0.1, 0.01, 0.1]
    # find_score_list = [0.999, 0.999, 0.999, 0.994, 0.994,0.994, 0.994, 0.994]
    find_score_list = [0.999, 0.999, 0.999, 0.994, 0.994,0.994, 0.994, 0.9646402934831231]
    
    all_scene = get_all_dataset()
    print(all_scene)

    
    ####### data root
    data_root = "/research/d1/gds/rszhu22/Contrastive-Lift/data"

    #############
    table_results_dir = f"/research/d1/gds/rszhu22/PCF-Lift/code/results/{export_table_name}"
    print(table_results_dir)
    os.makedirs(table_results_dir, exist_ok=True)
    evaluation_txt_file = f"{table_results_dir}/results.csv"
    evaluation_txt_file = open(evaluation_txt_file, 'w')
    # head_str = "dataset, iou, pq, sq, rq\n"
    head_str = "dataset, pq, instance_GT, instance_pred\n"
    evaluation_txt_file.write(f"{head_str}")
    evaluation_txt_file.flush()
    ############

    find_score_txt_csv_path = f"{table_results_dir}/optimal_score_NMS.csv"
    # find_score_txt = open(find_score_txt_csv_path, 'w')
    with open(find_score_txt_csv_path, 'w') as find_score_txt:
        head_str = "dataset, best_bd\n"
        find_score_txt.write(f"{head_str}")
        find_score_txt.flush()

    

    basedir="/research/d1/gds/rszhu22/PCF-Lift/code"
    # 
    ################
    # output_training_view_cache
    output_dir_parent = f"{basedir}/runs/{output_dir}"
    os.makedirs(output_dir_parent, exist_ok=True)
    output_training_view_cache = f"{basedir}/runs/{output_dir}/cache_training_view_feature"
    os.makedirs(output_training_view_cache, exist_ok=True)

    # output_training_view_cache sweep 20%
    output_training_view_cache_sweep = f"{basedir}/runs/{output_dir}/cache_training_view_feature_sweep"
    os.makedirs(output_training_view_cache_sweep, exist_ok=True)

    # inference feature
    output_inference = f"{basedir}/runs/{output_dir}/inference_on_test"
    os.makedirs(output_inference, exist_ok=True)
    ################
    epoch_list = [7]
    GPU_ID = 1
    for epoch in epoch_list:    
        for scene_idx, scene_name in enumerate(all_scene):

                
            
            minimal_value = minimal_value_list[scene_idx]
            
            
            ## todo add your checkpoint path
            ckpt = sorted(glob.glob(f"/research/d1/gds/rszhu22/PCF-Lift/checkpoint/MOS_covariance*/simplify_checkpoint/{scene_name}_res/checkpoints/epoch={epoch}-step=*"))[-1]
            print("epoch: ", epoch)
            
            # 
            scene_training_feature_cache = f"{output_training_view_cache}/{scene_name}"
            os.makedirs(scene_training_feature_cache, exist_ok=True)

            ##### save and find the centorid in the training view feature using the NMS algorithm
            # 
            cmd = f"CUDA_VISIBLE_DEVICES={GPU_ID} python3 inference/MOS_covariance/covariance_001_clamp/render_panopli_training_view_official_v2_learned_covariance_v1.py --ckpt_path {ckpt}  --output_dir {scene_training_feature_cache} --feature_dimension {feature_dimension} --minimal_value {minimal_value}"
            if not os.path.exists(f"{scene_training_feature_cache}/NMS_centroid.txt"):
                os.system(cmd)

            # continue
            cache_centroid = f"{scene_training_feature_cache}/NMS_centroid.txt"
            
            
            ########### sweep the best score threshold to find the suitable score
            
            # minimal_value_list = [0.01, 0.01, 0.1, 0.01,0.01, 0.1,0.01, 0.1,]
            

            # find_score = 0.999 # or 0.994
            find_score = find_score_list[scene_idx]
            print("find_score", find_score)
            

            
            scene_test_resulst = f"{output_inference}/{scene_name}"
            cmd = f"CUDA_VISIBLE_DEVICES={GPU_ID} python3 inference/MOS_covariance/covariance_001_clamp/render_panopli_training_view_inference_on_test_view_official_1_learned_covariance.py --ckpt_path {ckpt} --cached_centroids_path {cache_centroid}   --find_score {find_score} --output_dir {scene_test_resulst} --minimal_value {minimal_value}"
            if len(glob.glob(f"{scene_test_resulst}/*"))==0 or True:
                os.system(cmd)


            ######## evaluation
            evluatate_cmd =  f"CUDA_VISIBLE_DEVICES={GPU_ID}  python3 inference/evaluate_number.py --root_path /research/d1/gds/rszhu22/PCF-Lift/code/data/{scene_name} --exp_path {scene_test_resulst} --scene {scene_name} --MOS "
            output = subprocess.check_output(evluatate_cmd, shell=True).decode("utf-8")

            evaluation_txt_file.write(f"{output}")
            evaluation_txt_file.flush()
            # exit()
            # break
    
    pq_number = getlist_number(f"{table_results_dir}/results.csv",column=1)
    # scene_names_report = getlist_number(evaluation_txt_file,column=0)
    evaluation_txt_file = f"{table_results_dir}/results_report.csv"
    evaluation_txt_file = open(evaluation_txt_file, 'w')
    # head_str = "dataset, iou, pq, sq, rq\n"
    head_str = "dataset," + ",".join(all_scene) + "\n"
    evaluation_txt_file.write(f"{head_str}")
    evaluation_txt_file.flush()

    number_str = [str(num) for num in pq_number]
    number_str = "ours," + ",".join(number_str) + "\n"
    evaluation_txt_file.write(f"{number_str}")
    evaluation_txt_file.flush()

        

            

            


