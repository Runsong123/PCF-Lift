# PCF-Lift (ECCV 2024)


> **Runsong Zhu**, Shi Qiu, Qianyi Wu, Ka-Hei Hui, Pheng-Ann Heng, Chi-Wing Fu
> 
> Panoptic lifting is an effective technique to address the 3D panoptic segmentation task by unprojecting 2D panoptic segmentations from multi-views to 3D scene. However, the quality of its results largely depends on the 2D segmentations, which could be noisy and error-prone, so its performance often drops significantly for complex scenes. In this work, we design a new pipeline coined PCF-Lift based on our Probabilis-tic Contrastive Fusion (PCF) to learn and embed probabilistic features throughout our pipeline to actively consider inaccurate segmentations and inconsistent instance IDs. Technical-wise, we first model the probabilistic feature embeddings through multivariate Gaussian distributions. To fuse the probabilistic features, we incorporate the probability product kernel into the contrastive loss formulation and design a cross-view constraint to enhance the feature consistency across different views. For the inference, we introduce a new probabilistic clustering method to effectively associate prototype features with the underlying 3D object instances for the generation of consistent panoptic segmentation results. Further, we provide a theoretical analysis to justify the superiority of the proposed probabilistic solution. By conducting extensive experiments, our PCF-lift not only significantly outperforms the state-of-the-art methods on widely used benchmarks including the ScanNet dataset and the challenging Messy Room dataset (4.4% improvement of scene-level PQ), but also demonstrates strong robustness when incorporating various 2D segmentation models or different levels of hand-crafted noise.

> Codes will be released soon.

![image](https://github.com/Runsong123/PCF-Lift/blob/main/assets/Overview.png)



