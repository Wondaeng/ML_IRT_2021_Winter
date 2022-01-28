import yaml
import os
import operator
import cv2
import numpy as np
import torch
from active_learning.uncertainty_pooling import uncertainty_pooling
from data_prep.load import register_datasets
from detectron2.engine import DefaultTrainer
from train import get_init_train_cfg, get_loop_train_cfg
from Task_8.data_prep.pooling import random_pooling
from data_prep.filter import check_annotated, get_not_annotated
from data_prep.transfer_json import annot2train, train2annot


if __name__ == '__main__':

    # Get the config file name
    config_name = str(input('[Main]: Enter the config file name, or press enter to use default...'))

    if config_name == '':
        config_name = 'config.yaml'    # If just press enter, use 'config.yaml' as a default
        print('[Main]: Config file is set as default (config.yaml)')
    else:
        print(f'[Main]: Config file is set as {config_name}')

    with open(config_name, 'rb') as file:    # Open the config.yaml
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Initial dataset set up (according to config)
    # When original train images are not annotated at all
    if config['initial_random_training']:
        random_pooling(config, n=config['initial_pool_size'])    # Random Pooling
        input('[Main]: Please annotate ALL initial (random) data points (Press enter to continue...)')
        while not check_annotated(config):
            input('[Main]: Not all data points are annotated! (Press enter to continue...)')
        annot2train(config)    # Transfer annotation to original train folder

    # Active-learning loop
    for i in range(config['loops'] + 1):
        if i == 0:   # First loop
            register_datasets(config, init=True)
            cfg = get_init_train_cfg(config)
        else:
            register_datasets(config)
            cfg = get_loop_train_cfg(config)

        trainer = DefaultTrainer(cfg)
        trainer.train()

        if not i == config['loops']:
            # Get the list of files currently not annotated after random pooling (no corresponding .json file)
            pool_not_annot = get_not_annotated(config, extension='.jpeg')
            pool_most_uncertain = uncertainty_pooling(config, pool_not_annot, cfg)
            train2annot(config, pool_most_uncertain)

            # Query to annotate
            input('[Main]: Please annotate ALL data points (Press enter to continue...)')
            while not check_annotated(config):
                input('[Main]: Not all data points are annotated! (Press enter to continue...)')
            annot2train(config)  # Transfer annotation to original train folder




