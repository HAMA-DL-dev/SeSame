# SeSame : 3D Semantic Segmentation-Driven Representations for Precise 3D Object Detection

![qualitative result](./figure/3dbbd_merged.png)

## News
**[24.03.08]** All result and model zoo are uploaded.

**[24.02.28]** The result is submitted to KITTI 3D/BEV object detection benchmark with name [SeSame-point](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=3b3e791de572beb66e177976a3fae9e1f82c45a5), [SeSame-voxel](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=5ab0f2fcd328fe476234bedbff7398b3dd7f2546), [SeSame-pillar](https://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=4b2461007e0a9c58237e20b4f2ec7541a0eeaf03)

## To Do 
- [ ] Preprint of our work will be available after review process 
- [x] Upload whole project including training, validation logs and result on test split
- [x] Evaluation on KITTI ***val split*** and ***test split***
- [x] Code conversion ```spconv1.x``` to ```spconv2.x```

## Model Zoo
### 3D detection (car)
|model|AP_easy|AP_mod|AP_hard|config|pretrained weight|
|------|---|---|---|---|---|
|SeSame-point|85.25|76.83|71.60|[pointrcnn_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointrcnn_sem_painted.yaml)|[pointrcnn_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|
|SeSame-voxel|81.51|75.05|70.53|[second_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/second_sem_painted.yaml)|[second_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|
|SeSame-pillar|83.88|73.85|68.65|[pointpillar_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointpillar_sem_painted.yaml)|[pointpillar_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|

### BEV detection (car)
|model|AP_easy|AP_mod|AP_hard|config|pretrained weight|
|------|---|---|---|---|---|
|SeSame-point|90.84|87.49|83.77|[pointrcnn_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointrcnn_sem_painted.yaml)|[pointrcnn_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointrcnn_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|
|SeSame-voxel|89.86|85.62|80.95|[second_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/second_sem_painted.yaml)|[second_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/second_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|
|SeSame-pillar|90.61|86.88|81.93|[pointpillar_sem_painted.yaml](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/tools/cfgs/kitti_models/pointpillar_sem_painted.yaml)|[pointpillar_epoch80.pth](https://github.com/HAMA-DL-dev/SeSame/blob/main/detector/output/kitti_models/pointpillar_sem_painted_no_softmax/default/ckpt/checkpoint_epoch_80.pth)|



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
- CUDA 10.1
- NVIDIA TITAN RTX
- pcdet : 0.3.0+0
- spconv : 2.3.6
- torch : 1.10.1
- torchvision : 0.11.2
- torch-scatter : 2.1.2

## Setup
```bash
git clone https://github.com/HAMA-DL-dev/SeSame.git
cd SeSame

conda create -n sesame python=3.8
conda activate sesame
pip install -r requirements.txt
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

Before this step, please modify a location value named `TRAINING_PATH` and `SEMANTIC_KITTI_PATH` in `painting_cylinder3d.py`
```python
TRAINING_PATH = "/path/to/your/SeSame/detector/data/kitti/training/velodyne/"                     # <!--- point clouds from KITTI 3D object detection dataset
SEMANTIC_KITTI_PATH = "/path/to/your/SeSame/detector/tools/cfgs/dataset_configs/semantic-kitti.yaml" # <!--- semantic map of Semantic KITTI dataset
```

```
cd segment/
python demo_folder.py \
--demo-folder /path/to/your/kitti/training/velodyne/ \
--save-folder /path/to/your/labels_cylinder3d/

cd /path/to/your/kitti/training
mkdir segmented_lidar
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

CUDA_VISIBLE_DEVICES=2 python train.py \
--cfg_file cfgs/kitti_models/pointpillar_sem_painted.yaml \
--batch_size 16 --epochs 80 --workers 16 \
--ckpt_save_interval 5
```

If you stop the training process for mistake, don't worry. 

You can resume training with option `--start_epoch ${numbers of epoch}`


## Test
```bash
CUDA_VISIBLE_DEVICES=3 python test.py \
--cfg_file ../output/kitti_models/pointpillar_sem_painted/default/pointpillar_sem_painted.yaml \
--batch_size 16 \
--workers 4 \
--ckpt ../output/kitti_models/pointpillar_sem_painted/default/ckpt/checkpoint_epoch_70.pth \
--save_to_file
```

# Acknowledgments
Thanks for the opensource codes from [Cylinder3D](https://github.com/xinge008/Cylinder3D), [PointPainting](https://github.com/Song-Jingyu/PointPainting) and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
