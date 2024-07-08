import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

import pcdet_utils.calibration_kitti as calibration_kitti

import time 

# TRAINING_PATH = "/home/mmc-server3/Server/Datasets/Kitti/training/"
# TRAINING_PATH = "/home/mmc-server3/Server/Datasets/Kitti/testing/"

DATASET_PATH = "/home/mmc-server3/Server/Datasets/nuscenes/" 
CLASS_MAP_NUSC = "/home/mmc-server3/Server/Users/Hayeon/Cylinder3D-updated-CUDA/config/label_mapping/nuscenes.yaml"

CLASSES = ["noise", "car","truck","bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier" ]

CLASSES_DICT = {
  'noise': 0,
  'barrier': 1,
  'bicycle': 2,
  'bus' : 3,
  'car' : 4,
  'construction_vehicle': 5,
  'motorcycle' : 6,
  'pedestrian' : 7,
  'traffic_cone' : 8,
  'trailer' : 9,
  'truck' : 10
}



class Painter:
    def __init__(self):
        self.root_path = DATASET_PATH
        self.pcd_path = DATASET_PATH + 'sweeps/LIDAR_TOP/'
        self.absolute_files = sorted([f for f in os.listdir(self.pcd_path) if f.endswith('.pcd.bin')])
        
        self.save_path = DATASET_PATH + "painted_lidar/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            
        self.model = None
        self.sem_nusc_label = CLASS_MAP_NUSC
        self.class_map = None
        
    def get_lidar(self, idx):
        lidar_file = self.pcd_path + self.absolute_files[idx]
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self,idx):
        label_file = self.root_path + 'labels_cylinder3d/' + ('%06d.label' % idx)
        label_file = np.fromfile(label_file, dtype=np.uint32)
        sem_label = label_file & 0xFFFF  # semantic label in lower half
        inst_label = label_file >> 16    # instance id in upper half
        
        import yaml
        with open(self.sem_nusc_label) as f:
            self.class_map = yaml.load(f, Loader=yaml.FullLoader)
        
        label_mapped = []
        for label in sem_label:
            # label_mapped.append(clss_label['labels_16'][label]) # 'labels(32)' ? 'labels_16(17)' ?
            label_sem_to_det = self.class_map['learning_map'][label]
            label_mapped.append(self.class_map['labels_16'][label_sem_to_det])
        
        from collections import Counter
        counts = Counter(label_mapped)
        for word, count in counts.items():print(f"{word}: {count}")
        print("\n")
        return label_mapped
    
    def augment_lidar_score(self, points, labels):
        score = np.zeros((points.shape[0], 1+len(CLASSES_DICT)), dtype=np.float32) # plus one is for 'timestamp'
        for idx, point in enumerate(points):
            cls = labels[idx]
            if cls in CLASSES_DICT:
                score[idx][CLASSES_DICT[cls]+1] = 1.0
        score = torch.from_numpy(score).float()
        points_with_score = np.concatenate((points,score),axis=1)
        
        return points_with_score

    def run(self):
        SEGMENTED_POINTS_PATH = os.listdir(DATASET_PATH + '/labels_cylinder3d')
        total_num = len(SEGMENTED_POINTS_PATH)
        for idx, value in enumerate(sorted(SEGMENTED_POINTS_PATH)):
            points = self.get_lidar(idx)
            labels = self.get_label(idx)
            assert len(labels) == points.shape[0],f"Numbers of points({points.shape[0]}) and labels({len(labels)}) are unmatched"
            
            points_sem_scored = self.augment_lidar_score(points,labels)
            np.save("/home/mmc-server3/Server/Datasets2/nuscenes/painted_lidar/" + ("%06d.npy" % idx), points_sem_scored)
#             np.save(self.save_path + ("%06d.npy" % idx), points_sem_scored)
            
if __name__ == '__main__':
    painter = Painter()
    painter.run()