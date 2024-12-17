# SeSame: Simple, Easy 3D Object Detection with Point-Wise Semantics

> ***SeSame: Simple, Easy 3D Object Detection with Point-Wise Semantics*** \
> [Hayeon O](https://scholar.google.com/citations?user=KDQukv0AAAAJ&hl=ko), [Chanuk Yang](https://ieeexplore.ieee.org/author/37089004272), [Kunsoo Huh](https://scholar.google.com/citations?user=iRQAwt8AAAAJ&hl=ko&oi=ao) \
> Hanyang University

<a src="https://img.shields.io/badge/cs.CV-2404.12389-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2403.06501">  
<img src="https://img.shields.io/badge/cs.CV-2404.12389-b31b1b?logo=arxiv&logoColor=red"></a>

<div align="center">
    <img src="./figure/whole_flow.jpg" alt="overview" width="60%">
</div>

## News
**[24.09.20]** 🎉 Congratulations! The paper has been accepted to ACCV 2024 ! 🎉

**[24.07.31]** Update existing KITTI entry due to the expiration of submission 
- [SeSame-point](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=246df06571519a2fd61045424524b724fb8fffa3)
- [SeSame-voxel](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=7275a9efa5344149beb0a4051392a6e9039c1f52)
- [SeSame-pillar](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=4cdef7fed94a42226613f3273332791ce17acf55)

**[24.07.08]** Fix bugs 

**[24.03.08]** All result and model zoo are uploaded.

**[24.02.28]** The result is submitted to KITTI 3D/BEV object detection benchmark with name [SeSame-point](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=3b3e791de572beb66e177976a3fae9e1f82c45a5), [SeSame-voxel](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=5ab0f2fcd328fe476234bedbff7398b3dd7f2546), [SeSame-pillar](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=4b2461007e0a9c58237e20b4f2ec7541a0eeaf03)

## To Do 
- [x] Preprint of our work will be available after review process 
- [x] Upload whole project including training, validation logs and result on test split
- [x] Evaluation on KITTI ***val split*** and ***test split***
- [x] Code conversion ```spconv1.x``` to ```spconv2.x```

## Model Zoo
### 3D detection (car)
|model|AP_easy|AP_mod|AP_hard|config|pretrained weight|result|
|------|---|---|---|---|---|---|
|SeSame-point|85.25|76.83|71.60|[pointrcnn_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointrcnn_sem_painted.yaml)|[pointrcnn_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/log_train_20240212-184951.txt)|
|SeSame-voxel|81.51|75.05|70.53|[second_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/second_sem_painted.yaml)|[second_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/log_train_20240212-200457.txt)|
|SeSame-pillar|83.88|73.85|68.65|[pointpillar_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointpillar_sem_painted.yaml)|[pointpillar_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/log_train_20240212-185040.txt)|

### BEV detection (car)
|model|AP_easy|AP_mod|AP_hard|config|pretrained weight|result|
|------|---|---|---|---|---|---|
|SeSame-point|90.84|87.49|83.77|[pointrcnn_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointrcnn_sem_painted.yaml)|[pointrcnn_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/log_train_20240212-184951.txt)|
|SeSame-voxel|89.86|85.62|80.95|[second_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/second_sem_painted.yaml)|[second_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/log_train_20240212-200457.txt)|
|SeSame-pillar|90.61|86.88|81.93|[pointpillar_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointpillar_sem_painted.yaml)|[pointpillar_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|[log](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/log_train_20240212-185040.txt)|




## Contents
- [Requirements](#requirements)
- [Setup](#setup)
- [Datasets](#datasets)
- [Segment point clouds](#segment-point-clouds)
- [Generate GT database](#generate-gt-database)
- [Train](#train)
- [Test](#test)
- [Acknowledgments](#acknowledgments)

## Requirements
- CUDA 10.2
- NVIDIA TITAN RTX
- pcdet : 0.3.0+0
- spconv : 2.3.6
- torch : 1.10.1
- torchvision : 0.11.2
- torch-scatter : 2.1.2

If your CUDA version is not 10.2, it might be better to install those packages on your own.

The `environment.yaml` is suitable for CUDA 10.2 users. 

## Setup
```bash
git clone https://github.com/HAMA-DL-dev/SeSame.git
cd SeSame
conda env create -f environment.yaml
```


## Datasets
**KITTI 3D object detection** [(link)](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
```bash
/path/to/your/kitti
    ├── ImageSets
    ├── training
        ├── labels_cylinder3d        # !<--- segmented point clouds from 3D sem.seg.
        ├── segmented_lidar          # !<--- feature concatenated point clouds 
        ├── velodyne                 # !<--- point clouds 
        ├── planes
        ├── image_2
        ├── image_3
        ├── label_2
        └── calib
    ├── kitti_infos_train.pkl
    └── kitti_infos_val.pkl
```

|dataset|numbers of datset|index infos|dataset infos|
|---|---|---|---|
|train|3712 / 7481|[train.txt](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/data/kitti/ImageSets/train.txt)|kitti_infos_train.pkl|
|val|3769 / 7481|[val.txt](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/data/kitti/ImageSets/val.txt)|kitti_infos_val.pkl|
|test|7518|[test.txt](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/data/kitti/ImageSets/test.txt)|N/A|

For more information of `*.pkl` files, reference this documentation : [mmdetection3d-create-kitti-datset](https://mmdetection3d.readthedocs.io/en/v0.17.1/datasets/kitti_det.html#create-kitti-dataset)

## Segment point clouds
#### [Step1] Load pretrained weights at this [link](https://github.com/xinge008/Cylinder3D?tab=readme-ov-file#pretrained-models)

#### [Step2] Modify related paths like below

`semantickitti.yaml` [(link)](https://github.com/HAMA-DL-dev/SeSame/blob/main/segment/config/semantickitti.yaml#L64) : path to the downloaded weight

`painting_cylinder3d.py` [(link)](https://github.com/HAMA-DL-dev/SeSame/blob/main/segment/painting_cylinder3d.py) : path to your KITTI and semantic-kitti configs

```python
# point clouds from KITTI 3D object detection dataset
TRAINING_PATH = "/path/to/your/SeSame/detector/data/kitti/training/velodyne/"

# semantic map of Semantic KITTI dataset
SEMANTIC_KITTI_PATH = "/path/to/your/SeSame/detector/tools/cfgs/dataset_configs/semantic-kitti.yaml" 
```
    
#### [Step3] Segment raw point clouds from KITTI object detection dataset 
```
cd /path/to/your/kitti/training
mkdir segmented_lidar
mkdir labels_cylinder3d
cd /path/to/your/SeSame/segment/

python demo_folder.py --demo-folder /path/to/your/kitti/training/velodyne/ --save-folder /path/to/your/kitti/training/labels_cylinder3d/

python pointpainting_cylinder3d.py
```


## Generate GT database
```bash
cd detector/tools
python -m pcdet.datasets.kitti.sem_painted_kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/semantic_painted_kitti.yaml
```


## Train
```bash
cd ~/SeSame/detector/tools
python train.py --cfg_file cfgs/kitti_models/${model.yaml} --batch_size 16 --epochs 80 --workers 16 --ckpt_save_interval 5
```

#### example
```bash
python train.py --cfg_file cfgs/kitti_models/pointpillar_sem_painted.yaml --batch_size 16 --epochs 80 --workers 16 --ckpt_save_interval 5
```

If you stop the training process for mistake, don't worry. 

You can resume training with option `--start_epoch ${numbers of epoch}`


## Test
```bash
python test.py --cfg_file ${configuration file of each model with *.yaml} --batch_size ${4,8,16} --workers 4 --ckpt ${path to *.pth file} --save_to_file
```

#### example
```bash
python test.py --cfg_file ../output/kitti_models/pointpillar_sem_painted/default/pointpillar_sem_painted.yaml --batch_size 16 --workers 4 --ckpt ../output/kitti_models/pointpillar_sem_painted/default/ckpt/checkpoint_epoch_70.pth --save_to_file
```

# Acknowledgments
Thanks for the opensource codes from [Cylinder3D](https://github.com/xinge008/Cylinder3D), [PointPainting](https://github.com/Song-Jingyu/PointPainting) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

# Citation
```
@InProceedings{O_2024_ACCV,
    author    = {O, Hayeon and Yang, Chanuk and Huh, Kunsoo},
    title     = {SeSame: Simple, Easy 3D Object Detection with Point-Wise Semantics},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {2889-2905}
}
```
