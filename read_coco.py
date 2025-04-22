
import os

# path='/gruntdata/heyuan4/workspace/nayun.xsy/coco/train2017'
path='/mnt/workspace/xuanshiyu.xsy/instruct/coco/coco/val2017'
# path='/mnt/workspace/xuanshiyu.xsy/instruct/coco/coco/train2017'

files = os.listdir(path)  # 获取文件和子目录

print(files)
print('000000572517.jpg' in files)
print('000000533083.jpg' in files)