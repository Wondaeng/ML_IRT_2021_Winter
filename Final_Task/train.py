import os
import pickle
import gc
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo



def get_init_train_cfg(conf):

    config_file_path = conf['network_config']
    checkpoint_url = conf['pretrained_weights']
    train_dataset_name = conf['train_set_name']
    test_dataset_name = conf['test_set_name']

    num_classes = len(conf['classes'])
    num_workers = conf['num_workers']
    num_iteration = conf['max_iteration']

    img_per_batch = conf['train_batch_size']
    learning_rate = conf['base_learning_rate']
    device = conf['devices']
    output_dir = conf['output_dir']

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.SOLVER.IMS_PER_BATCH = img_per_batch
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = num_iteration
    cfg.SOLVER.STEPS = []    # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg


def get_loop_train_cfg(conf):

    train_dataset_name = conf['train_set_name']
    test_dataset_name = conf['test_set_name']

    num_classes = len(conf['classes'])
    num_workers = conf['num_workers']
    num_iteration = conf['max_iteration']

    img_per_batch = conf['train_batch_size']
    learning_rate = conf['base_learning_rate']
    device = conf['devices']
    output_dir = conf['output_dir']
    cfg_save_path = conf['cfg_save_path']

    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)


    cfg.DATALOADER.NUM_WORKERS = num_workers

    cfg.SOLVER.IMS_PER_BATCH = img_per_batch
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = num_iteration
    cfg.SOLVER.STEPS = []    # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg