# PCF-Lift (ECCV 2024)
## [[Paper](https://arxiv.org/abs/2410.10659)] [[Poster](https://github.com/Runsong123/PCF-Lift/blob/main/assets/Poster_PCF_Lift.pdf)]

> **Runsong Zhu**, Shi Qiu, Qianyi Wu, Ka-Hei Hui, Pheng-Ann Heng, Chi-Wing Fu
> 

>**TL;DR**: Our paper presents a novel "probabilistic" fusion method to lift 2D predictions to 3D for effective and robust instance segmentation, achieving SOTA performances.

> 

![image](https://github.com/Runsong123/PCF-Lift/blob/main/assets/Overview.png)




## Data and Pretrained checkpoints
You can download the Messy Rooms (MOS) dataset from [here](https://figshare.com/s/b195ce8bd8eafe79762b). For all other datasets, refer to
the instructions provided in [Panoptic-Lifting](https://github.com/nihalsid/panoptic-lifting)



we provide **pretrained checkpoints** for MOS dataset and you can download them from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155183723_link_cuhk_edu_hk/EYVeG_sGhRVEtEoZoh9nj_kB6VwTBa22kqWCcEeD5BqsyA?e=9MlcYd).

## Inference and Evaluation.
Download the pretrained checkpoints and place them to ```./code```. Then, run the following commands to evaluate the pretrained models:
```
cd code & python inference_test/MOS_covariance/covariance_001_clamp/bash_inference_training_view_official_v2_learned_covariance_v1.py --output_dir PCF_res --feature_dimension 7 --export_table_name PCF_res 
```

## Training
```
cd code & bash train.sh
```


# Citation
If you find this work useful in your research, please cite our paper:
```
@inproceedings{zhu2025pcf,
  title={PCF-Lift: Panoptic Lifting by Probabilistic Contrastive Fusion},
  author={Zhu, Runsong and Qiu, Shi and Wu, Qianyi and Hui, Ka-Hei and Heng, Pheng-Ann and Fu, Chi-Wing},
  booktitle={European Conference on Computer Vision},
  pages={92--108},
  year={2025},
  organization={Springer}
}
```


## Thanks
This code is based on [Contrastive Lift](https://github.com/yashbhalgat/Contrastive-Lift), [Panoptic-Lifting](https://github.com/nihalsid/panoptic-lifting) and [TensoRF](https://github.com/apchenstu/TensoRF) codebases. We thank the authors for releasing their code. 
