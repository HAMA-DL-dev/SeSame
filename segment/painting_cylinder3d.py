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

TRAINING_PATH = "/path/to/your/kitti/training/"
SEMANTIC_KITTI_PATH = "/path/to/your/semantic-kitti.yaml"
CLASSES = ["car","ped","cyclist"]
CLSS_DICT = {
    10 : "car",
    6 : "person",
    7 : "cyclist",
    8 : "cyclist"
}

class Painter:
    def __init__(self):
        self.root_split_path = TRAINING_PATH
        self.save_path = TRAINING_PATH + "segmented_lidar/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model = None
        self.sem_kitti_label = SEMANTIC_KITTI_PATH
        
    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_label(self,idx):
        label_file = self.root_split_path + 'labels_cylinder3d/' + ('%s.label' % idx)
        label_file = np.fromfile(label_file, dtype=np.uint32)
        sem_label = label_file & 0xFFFF  # semantic label in lower half
        inst_label = label_file >> 16    # instance id in upper half

        import yaml
        with open(self.sem_kitti_label) as f:
            clss_label = yaml.load(f, Loader=yaml.FullLoader)
        
        label_mapped = []
        for label in sem_label:
            label_mapped.append(clss_label['learning_map'][label])
        
        return label_mapped
    
    def augment_lidar_score(self,points,labels):
        score = np.zeros((points.shape[0],len(CLASSES)+1),dtype=np.float32)
        
        # TODO : one-hot encoding 
        for idx,point in enumerate(points):
            try:
                for i, (key,cls) in enumerate(CLSS_DICT.items()):
                    print(i,cls,CLSS_DICT[labels[idx]])
                    score[idx][i+1] = 1.0 if CLSS_DICT[labels[idx]]==cls else False
            except:pass
        
        score = torch.from_numpy(score).float()
        
        ## NOT TODO : Softmax function 
        # sf = torch.nn.Softmax(dim=-2)
        # score = sf(score).cpu().numpy()
        points_with_score = np.concatenate((points,score),axis=1)

        return points_with_score
        
    def run(self):
        total_num = len(os.listdir(TRAINING_PATH + '/labels_cylinder3d'))
        for idx in tqdm(range(total_num)):
            sample_idx = "%06d" % idx
            points = self.get_lidar(sample_idx)
            labels = self.get_label(sample_idx)
            assert len(labels) == points.shape[0],"Numbers of points and labels are unmatched"

            points_sem_scored = self.augment_lidar_score(points,labels)
            np.save(self.save_path + ("%06d.npy" % idx), points_sem_scored)

if __name__ == '__main__':
    painter = Painter()
    painter.run()
