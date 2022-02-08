import os
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import DatasetCatalog


def register_datasets(conf, init=False):
    print("[Data Load]: Converting labelme json to COCO json format...")
    os.system('python labelme2coco.py --labelme_images ./datasets/train --output ./datasets/json/train.json')
    os.system('python labelme2coco.py --labelme_images ./datasets/test --output ./datasets/json/test.json')
    print("[Data Load]: Converting is Finished")

    train_dataset_name = conf['train_set_name']
    train_json_annot_path = conf['train_json']
    train_images_path = conf['train_dir']

    test_dataset_name = conf['test_set_name']
    test_json_annot_path = conf['test_json']
    test_images_path = conf['test_dir']

    if not init:
        DatasetCatalog.remove(conf['train_set_name'])
        DatasetCatalog.remove(conf['test_set_name'])

        register_coco_instances(name=train_dataset_name, metadata={},
                                json_file=train_json_annot_path, image_root=train_images_path)

        register_coco_instances(name=test_dataset_name, metadata={},
                                json_file=test_json_annot_path, image_root=test_images_path)
    else:
        register_coco_instances(name=train_dataset_name, metadata={},
                                json_file=train_json_annot_path, image_root=train_images_path)

        load_coco_json(dataset_name=train_dataset_name, json_file=train_json_annot_path, image_root=train_images_path)

        register_coco_instances(name=test_dataset_name, metadata={},
                                json_file=test_json_annot_path, image_root=test_images_path)

        load_coco_json(dataset_name=test_dataset_name, json_file=test_json_annot_path, image_root=test_images_path)

    print("[Data Load]: Dataset registration is finished")

