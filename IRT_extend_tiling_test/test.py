import json, utils

from collections import OrderedDict

new_json = OrderedDict()
new_json['images'] = []
new_json['annotations'] = []
print(json.dumps(new_json, ensure_ascii=False, indent="\t"))

utils.tiling_bbox(4, './test_image/dataset.json', './images_slice' )