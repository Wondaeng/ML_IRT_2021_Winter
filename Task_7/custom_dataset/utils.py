from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

output_dir = "./output/object_detection"
num_classes = 1

device = "cuda"

train_dataset_name = "LP_train"    # License plate (LP) detection model
train_images_path = "train"
train_json_annot_path = "train.json"
