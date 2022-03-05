import os


def check_annotated(conf):
    path_annot = conf['train_dir'] + '/annotate'
    all_files = [f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]
    files_img = [f for f in all_files if not f.endswith('.json')]
    files_json = [f for f in all_files if f.endswith('.json')]
    return len(files_img) == len(files_json)


def get_not_annotated(conf, extension='.jpeg'):
    path_annot = conf['train_dir']
    all_files = [f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]
    files_json = [f.rstrip('.json') for f in all_files if f.endswith('.json')]
    imgs_no_ext = [f.rstrip(extension) for f in all_files if f.endswith(extension)]
    imgs_no_annot_no_ext = list(set(imgs_no_ext)-set(files_json))
    imgs_no_annot = [f + extension for f in imgs_no_annot_no_ext]
    return imgs_no_annot

