import os
import image_slicer
import shutil
import json

from pycocotools.coco import COCO
from collections import OrderedDict


def get_names(pth):    # Get list of (image) file names with extension
    file_list = os.listdir(pth)    # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[1] != 'json']
    extension = file_list[0][1]
    file_list = [i[0] + '.' + extension for i in file_list]

    return file_list


def get_names_without_ext(pth):
    file_list = os.listdir(pth)  # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[1] != 'json']
    file_list = [i[0] for i in file_list]

    return file_list


def tiling_img(n, src, dst, names):    # Tiling images into n slices in new folder
    # Copy the folder with original (whole) images
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Remove annotation files
    temp = os.listdir(dst)
    for item in temp:
        if item.endswith(".json"):
            os.remove(os.path.join(dst, item))

    # Slice each original image into n slices and remove original one
    for i in names:
        file_pth = os.path.join(dst, i)
        image_slicer.slice(file_pth, n)
        os.remove(file_pth)
    print(f'{len(names)} images are sliced! (Result in {len(os.listdir(dst))} slices)')
    return None

def tiling_bbox(n, pth_json, dst):
    coco = COCO(pth_json)    # Load COCO annotation

    new_json = OrderedDict()
    new_json['images'] = []
    new_json['annotations'] = []

    # Make new "images" field of COCO json file with sliced images
    sqrt_n = int(n ** (1 / 2))
    id = 0
    ann_id = 1

    for i in coco.imgs:
        h = coco.loadImgs(i)[0]['height'] // sqrt_n
        w = coco.loadImgs(i)[0]['width'] // sqrt_n
        img_name = coco.loadImgs(i)[0]['file_name'].split('.')[0]
        ext = coco.loadImgs(i)[0]['file_name'].split('.')[1]
        for j in range(1, sqrt_n + 1):
            for k in range(1, sqrt_n + 1):
                slice_name = img_name + f'_0{j}_0{k}.{ext}'
                id += 1
                dct_img_temp = {"height":h,"width":w,"id":id,"file_name":slice_name}
                new_json['images'].append(dct_img_temp)

        lst_annots = coco.getAnnIds(i)

        for j in lst_annots:
            annot = coco.loadAnns(j)[0]
            category_id = annot["category_id"]
            x1, y1 = annot["bbox"][0], annot["bbox"][1]

            # COCO bounding box format is [top left x position, top left y position, width, height].
            bbox_w, bbox_h = annot["bbox"][2], annot["bbox"][3]

            # where a & b mean: filename_0(a+1)_0(b+1).ext
            slice_num_a = x1 // w
            slice_num_b = y1 // h
            new_x1, new_y1 = x1 % w, y1 % h

            new_bbox = [new_x1, new_y1, bbox_w, bbox_h]
            
            # Add new annotation if only bbox does not present accross tiling boundaries
            if new_x1 + bbox_w > w or new_y1 + bbox_h > h:
                pass
            else:
                new_annot = {"iscrowd":0,"image_id":4*(i-1) + slice_num_a + slice_num_b*sqrt_n + 1,
                             "bbox":new_bbox,"segmentation":[],
                             "category_id":category_id,"id":ann_id,"area":bbox_w*bbox_h}
                new_json["annotations"].append(new_annot)
            ann_id += 1
    print(json.dumps(new_json, ensure_ascii=False, indent='\t'))    # Print JSON file on memory
    with open(dst + '/new_dataset.json', 'w') as outfile:
        json.dump(new_json, outfile, indent=4)


