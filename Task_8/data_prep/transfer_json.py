import os
import shutil


def annot2train(conf):    # Copy .json files in annotate folder to train folder
    path_train = conf['train_dir']
    path_annot = conf['train_dir'] + '/annotate'
    all_files = [f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]
    files_json = [f for f in all_files if f.endswith('.json')]
    for f in files_json:
        src = os.path.join(path_annot, f)
        dst = os.path.join(path_train, f)
        shutil.copy(src, dst)
    print('[Data]: .json files are copied to original train dataset folder')


def train2annot(conf, pool):    # Copy (uncertain) image files in train folder to annotate folder
    path_train = conf['train_dir']
    path_annot = conf['train_dir'] + '/annotate'

    if os.path.isdir(path_annot):
        shutil.rmtree(path_annot)

    os.makedirs(path_annot, exist_ok=True)
    for f in pool:
        src = os.path.join(path_train, f)
        dst = os.path.join(path_annot, f)
        shutil.copy(src, dst)
    print('[Data]: (uncertain) image files are copied into annotate folder')