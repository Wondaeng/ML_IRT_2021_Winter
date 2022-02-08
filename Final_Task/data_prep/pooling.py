import random
import sys
import os, shutil


def random_pooling(conf, n=1):
    # Remove directory if 'annotate' folder already exists
    if os.path.isdir(os.path.join(conf['train_dir'], 'annotate')):
        shutil.rmtree(os.path.join(conf['train_dir'], 'annotate'))

    # Get the names of files in the (original) train dataset folder
    all_files = [f for f in os.listdir(conf['train_dir']) if os.path.isfile(os.path.join(conf['train_dir'], f))]
    files_no_annot = [f for f in all_files if not f.endswith('.json')]
    while n > len(files_no_annot):
        keyword = input('[Data]: Samples lager than population, please add more and press enter or type "exit" to finish.')
        if keyword == 'exit':
            sys.exit('')
        else:
            all_files = [f for f in os.listdir(conf['train_dir']) if os.path.isfile(os.path.join(conf['train_dir'], f))]
            files_no_annot = [f for f in all_files if not f.endswith('.json')]
    random_files_no_annot = random.sample(files_no_annot, n)

    # Make new 'annotate' folder and copy the randomly selected data_prep points to the folder
    os.makedirs(os.path.join(conf['train_dir'], 'annotate'), exist_ok=True)
    for f in random_files_no_annot:
        src = os.path.join(conf['train_dir'], f)
        dst = os.path.join(os.path.join(conf['train_dir'], 'annotate'), f)
        shutil.copy(src, dst)
    print("[Data]: Initial random pooling is finished. Please check './train/annotate folder'.")


