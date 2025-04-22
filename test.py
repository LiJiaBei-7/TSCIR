import torch
import json
import os 



# img_path = '/mllm_native/wangyabing.wyb/datasets/mscoco2017_unlabeled/unlabeled2017'

# items = os.listdir(img_path)
# print(items[0])


file_path = '/mnt_rela/wangyabing.wyb/datasets/CRICO/unlabeled2017/image_info_unlabeled2017.json'

with open(file_path) as f:
    data = json.load(f)

print(data['images'][0])