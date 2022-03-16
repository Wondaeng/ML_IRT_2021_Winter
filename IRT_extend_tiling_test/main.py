import utils
import os
import labelme2coco

img_src = './test_image'
slice_dst = './images_slice'
slice_num = 4
lst_names = utils.get_names(img_src)

utils.tiling_img(slice_num, img_src, slice_dst, lst_names)

# Get original COCO annotation (.json) file
train_split_rate = 1
labelme2coco.convert(img_src, img_src, train_split_rate)

print('end')
