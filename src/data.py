# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb
import json

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000                                                                                              

from typing import Union
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
from torchvision.datasets.folder import DatasetFolder
import torchvision.datasets as datasets
import torchvision.transforms as T
from third_party.open_clip.clip import tokenize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal






class CIRCODataset(Dataset):
    """
    Copy-paste from https://github.com/miccunifi/SEARLE/blob/main/src/datasets.py
    CIRCO dataset class for PyTorch.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_captions', 'shared_concept',
             'gt_img_ids', 'query_id'] when split == 'val'
            - ['reference_image', 'reference_name', 'relative_captions', 'shared_concept', 'query_id'] when split == test
    """

    def __init__(self, dataset_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        """
        Args:
            dataset_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        dataset_path = Path(dataset_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = dataset_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(dataset_path / 'unlabeled2017' / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        img_path = '/mllm_native/wangyabing.wyb/datasets/mscoco2017_unlabeled/unlabeled2017'
        with open(f'{dataset_path}/val_name.json', 'r', encoding='utf-8') as file:
            data = json.load(file)  # 解析 JSON 文件为 Python 对象
                        
        self.img_paths = [f'{img_path}/{img_info["file_name"]}' for img_info in
                          imgs_info["images"]]

        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]

        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(dataset_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def get_target_img_ids(self, index) -> Dict[str, int]:
        """
        Returns the id of the target image and ground truth images for a given query

        Args:
            index (int): id of the query

        Returns:
             Dict[str, int]: dictionary containing target image id and a list of ground truth image ids
        """

        return {
            'target_img_id': self.annotations[index]['target_img_id'],
            'gt_img_ids': self.annotations[index]['gt_img_ids']
        }

    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id]
            if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            relative_caption_token = tokenize(f'a photo of * that {relative_caption}')[0]         

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(Image.open(reference_img_path))

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                # target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                # target_img = self.preprocess(Image.open(target_img_path))

                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))
                return reference_img, relative_caption_token, shared_concept, target_img_id, gt_img_ids
                # return {
                #     'reference_image': reference_img,
                #     'reference_name': reference_img_id,
                #     'target_image': target_img,
                #     'target_name': target_img_id,
                #     'relative_caption': relative_caption,
                #     'shared_concept': shared_concept,
                #     'gt_img_ids': gt_img_ids,
                #     'query_id': query_id,
                # }

            elif self.split == 'test':
                return reference_img, reference_img_id, relative_caption_token, shared_concept, query_id
                # return {
                #     'reference_image': reference_img,
                #     'reference_name': reference_img_id,
                #     'relative_caption': relative_caption,
                #     'shared_concept': shared_concept,
                #     'query_id': query_id,
                # }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(Image.open(img_path))
            return img, img_id
            # return {
            #     'image': img,
            #     'image_name': img_id
            # }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


## Structure of dataset directory
## CIRR: under ./data/CIRR
## validation images ./dev/
## caption split ./captions/cap.rc2.val.json
## image split ./image_splits/split.rc2.val.json
class CIRR(Dataset):
    def __init__(self, transforms, mode='caps', 
    vis_mode=False, test=False, root='./data'):
        self.mode = mode
        self.transforms = transforms
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = root #os.path.join(root, 'CIRR')
        self.root_img = os.path.join(self.root, 'dev')
        if self.test:
            self.root_img = os.path.join(self.root, 'test1')
            if self.mode == 'caps':
                self.json = os.path.join(self.root , 'captions/cap.rc2.test1.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.test1.json')
        else:
            if self.mode == 'caps':
                self.json = os.path.join(self.root, 'captions/cap.rc2.val.json')
            else:
                self.json = os.path.join(self.root, 'image_splits/split.rc2.val.json')
        logging.debug(f'Loading json data from {self.json}.')
        data = json.load(open(self.json, "r"))                                
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []        
        if self.test:
            self.init_test(data)
        elif self.mode == 'caps':            
            self.init_val(data)                        
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]                    
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))
        logging.info("Use {} imgs".format(len(self.target_imgs)))        

    def init_test(self, data):
        self.pairids = []
        if self.mode == 'caps':
            for d in data:
                ref_path = d['reference']+ ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d['caption']) 
                self.pairids.append(d['pairid'])
                self.target_imgs.append('dummy')
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d['reference']+ ".png"
            tar_path = d['target_hard']+ ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d['caption'])            
    
    def return_testdata(self, idx):
        if self.mode == 'caps':
                ref_path = str(self.ref_imgs[idx])
                img_path = os.path.join(self.root_img, ref_path)
                ref_images = self.transforms(Image.open(img_path))
                target_cap = self.target_caps[idx]
                text_with_blank_raw = 'a photo of * , that {}'.format(target_cap)    
                caption_only = tokenize(target_cap)[0]
                text_with_blank = tokenize(text_with_blank_raw)[0]                 
                return ref_images, text_with_blank, \
                    caption_only, str(self.ref_imgs[idx]), \
                        self.pairids[idx], text_with_blank_raw
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == 'caps' and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = 'a photo of * , that {}'.format(target_cap)
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0]                 
            return ref_images, ref_text_tokens, caption_only, \
                str(self.ref_imgs[idx]), str(self.target_imgs[idx])
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, tar_path

    def __getitem__(self, idx):
        if self.test:                        
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)
    
    def __len__(self):
        return len(self.target_imgs)
        
## Fashion-IQ: under ./data/fashion-iq
## validation images ./images
## caption split ./json/cap.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
## image split ./image_splits/split.{cloth_type}.val.json, cloth_type in [toptee, shirt, dress]
class FashionIQ(Dataset):
    def __init__(self, cloth, transforms, is_train=False, vis_mode=False, \
        mode='caps', is_return_target_path=False, root='./data'):
        root_iq = root #os.path.join(root, 'fashion-iq')
        self.root_img = os.path.join(root_iq, 'images')
        self.vis_mode = vis_mode
        self.mode = mode
        self.is_return_target_path = is_return_target_path
        self.transforms = transforms
        if mode == 'imgs':
            self.json_file = os.path.join(root_iq, 'image_splits', \
                'split.{}.val.json'.format(cloth))
        else:
            self.json_file = os.path.join(root_iq, 'captions', \
                'cap.{}.val.json'.format(cloth))                
        logging.debug(f'Loading json data from {self.json_file}.')

        self.ref_imgs = []
        self.target_imgs = []
        self.ref_caps = []
        self.target_caps = []        
        if mode == 'imgs':
            self.init_imgs()
            logging.info("Use {} imgs".format(len(self.target_imgs)))
        else:
            self.init_data()     
            logging.info("Use {} imgs".format(len(self.target_imgs)))

    def init_imgs(self):
        data = json.load(open(self.json_file, "r"))
        self.target_imgs = [key + ".png" for key in data]        

    def init_data(self):
        def load_data(data):
            for d in data:
                ref_path = os.path.join(self.root_img, d['candidate']+ ".png") 
                tar_path = os.path.join(self.root_img, d['target']+ ".png")            
                try:
                    Image.open(ref_path)
                    Image.open(tar_path)
                    self.ref_imgs.append(ref_path)
                    self.target_imgs.append(tar_path)
                    self.ref_caps.append((d['captions'][0], d['captions'][1]))
                    #self.target_caps.append(d['captions'][1])
                except:                
                    print('cannot load {}'.format(d['candidate']))
        if isinstance(self.json_file, str):
            data = json.load(open(self.json_file, "r"))        
            load_data(data)            
        elif isinstance(self.json_file, list):
            for filename in self.json_file:
                data = json.load(open(filename, "r")) 
                load_data(data)         

    def __len__(self):
        if self.mode == 'caps':
            return len(self.ref_imgs)
        else:
            return len(self.target_imgs)

    def return_imgs(self, idx):
        tar_path = str(self.target_imgs[idx])
        img_path = os.path.join(self.root_img, tar_path)
        target_images = self.transforms(Image.open(img_path))
        return target_images, os.path.join(self.root_img, tar_path)

    def return_all(self, idx):
        if self.vis_mode:
            tar_path = str(self.target_imgs[idx])
            target_images = self.transforms(Image.open(tar_path))
            return target_images, tar_path            
        ref_images = self.transforms(Image.open(str(self.ref_imgs[idx])))
        target_images = self.transforms(Image.open(str(self.target_imgs[idx])))
        cap1, cap2 = self.ref_caps[idx]
        text_with_blank = 'a photo of * , that {} and {}'.format(cap2, cap1)
                
        if self.is_return_target_path:
            token_texts = tokenize(text_with_blank)[0]  
            return ref_images, target_images, token_texts, token_texts, \
                str(self.target_imgs[idx]), str(self.ref_imgs[idx]), \
                    cap1
        else:
            return ref_images, target_images, text_with_blank


    def __getitem__(self, idx):
        if self.mode == 'imgs':            
            return self.return_imgs(idx)
        else:            
            return self.return_all(idx)
        
## COCO: under ./data/coco
## validation images ./val2017
## validation masked images ./val2017_masked
## validation csv file ./coco_eval.csv
class CsvCOCO(Dataset):
    def __init__(self, transforms, transforms_region, sep=",",
                return_data_identifier=False, return_filename=False, 
                root='./data'):
        self.transforms = transforms
        self.transforms_region = transforms_region
        self.root = os.path.join(root, 'coco')
        self.root_img = os.path.join(self.root, 'val2017')
        self.csv_file = os.path.join(self.root, 'coco_eval.csv')
        logging.debug(f'Loading csv data from {self.csv_file}.')
        df = pd.read_csv(self.csv_file, sep=sep)                
        self.images = df['id'].tolist()
        ## query_region contains the box of query regions.
        regions = df['query_regions'].tolist()
        self.regions = []
        for region in regions:
            x1, y1, x2, y2 = map(lambda x: int(float(x)), region.split(";"))
            self.regions.append([x1, y1, x2, y2])

        ## query_classes contains the class of query region in the target.
        self.query_classes = df['query_class'].tolist()
        self.classes = []
        ## classes contains the list of classes in the target.
        for list_class in df['classes'].tolist():
            if isinstance(list_class, str):
                list_class = list_class.split(";")
                self.classes.append(list_class)
            else:
                self.classes.append([""])        
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img, str(self.images[idx]))
        image = Image.open(img_path)        
        masked_path = os.path.join(self.root_img.replace('val2017', 'val2017_masked'), \
            str(self.images[idx]))
        image_masked = Image.open(masked_path)
        
        ## extract query region.
        x1, y1, x2, y2 = self.regions[idx]        
        region_image = image_masked.crop((x1, y1, x2, y2)) 

        image = self.transforms(image)
        ## no cropping is applied to query region.
        region_image = self.transforms_region(region_image)
        query_class = self.query_classes[idx]
        other_classes = self.classes[idx]        
        text_with_blank = 'a photo of * and {}'.format(" and ".join(other_classes))
        text_with_queryclass = 'a photo of * and {} and {}'.format(query_class, \
            " and ".join(other_classes))
        raw_text = text_with_queryclass
        text_full = 'a photo of {} and {}'.format(query_class, " and ".join(other_classes))        
        text_with_blank = tokenize(text_with_blank)[0]
        text_with_queryclass = tokenize(text_with_queryclass)[0]
        text_full = tokenize(text_full)[0]
        return image, region_image, text_full, text_with_blank, \
            text_with_queryclass, str(self.images[idx]), raw_text


class ImageList(Dataset):
    def __init__(self, input_filename, transforms, root=None, 
                 return_filename=False, is_labels=False):
        logging.debug(f'Loading txt data from {input_filename}.')
        with open(input_filename, 'r') as f:
            lines = f.readlines()
        if not is_labels:
            self.images = [line.strip() for line in lines]
        else:
            filenames = [line.strip() for line in lines]
            self.images = [name.split(" ")[0] for name in filenames] 
            self.labels = [int(name.split(" ")[1]) for name in filenames] 
        self.is_labels = is_labels
        self.transforms = transforms
        self.root = root
        logging.debug('Done loading data.')
        self.return_filename = return_filename

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.root is not None:
            img_path = os.path.join(self.root, str(self.images[idx]))
        else:
            img_path = str(self.images[idx])
        images = self.transforms(Image.open(img_path))
        if self.return_filename:
            return images, img_path
        elif self.is_labels:
            target = self.labels[idx]
            return images, target       
        else:
            return images


class CustomFolder(Dataset):
    def __init__(self, folder, transform):
        image_lists = os.listdir(folder)
        self.samples = [os.path.join(folder, name) for name in image_lists]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = Image.open(str(path))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        if self.return_filename:
            return images, str(self.images[idx])
        texts = tokenize([str(self.captions[idx])])[0]

        if self.return_data_identifier:
            return images, texts, 0
        return images, texts

import json
import oss2
import tqdm
import io

class OssDataset(Dataset):
    def __init__(self, bucket, input_filename, transforms, img_key, caption_key, sep="\t",
                 return_data_identifier=False, return_filename=False):
        self.bucket = bucket
        self.object_name = f'public_data/cc/training/'
        input_filename = '/mnt_rela/wangyabing.wyb/datasets/CC3M/CC3M_train.jsonl'
        self.anns = []
        logging.debug(f'Loading jsonl data from {input_filename}.')
        # {'caption': 'a very typical bus station', 'clip_name': 'e6907b4b11a9bf76f739ed442b0281ab/cc/training/00000000.jpg', 'sen_id': 0, 'type': 'image'}
        with open(input_filename, 'r') as input_file:
            for line in input_file:
                # 解析 JSON 数据
                self.anns.append(json.loads(line))

        self.transforms = transforms
        self.return_data_identifier = return_data_identifier
        # logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        data = self.anns[idx]
        
        image_path = data['clip_name'].rsplit('/', 1)[-1]
        image_path = f'public_data/cc/training/{image_path}' 
        result = self.bucket.get_object(image_path)
        image_stream = io.BytesIO(result.read())
        images = self.transforms(Image.open(image_stream))

        if self.return_filename:
            return images, str(self.images[idx])
        
        caption = data['caption']
        texts = tokenize([str(caption)], truncate=True)[0]

        if self.return_data_identifier:
            return images, texts, 0
        return images, texts


def train_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # 过滤掉为None的数据项
    return torch.utils.data.dataloader.default_collate(batch)


import time

class OssDataset_with_caption(Dataset):
    def __init__(self, input_filename, transforms_strong, transforms, sep="\t",
                 return_data_identifier=False, return_filename=False):

        generated_caption_dir = '/mnt_rela/wangyabing.wyb/datasets/CC3M/llm_data_p2.jsonl'
        input_filename = '/mnt_rela/wangyabing.wyb/datasets/CC3M/CC3M_train.jsonl'
        self.anns = []
        logging.debug(f'Loading jsonl data from {input_filename}.')
        # {'caption': 'a very typical bus station', 'clip_name': 'e6907b4b11a9bf76f739ed442b0281ab/cc/training/00000000.jpg', 'sen_id': 0, 'type': 'image'}
        
        start_t = time.time()
        self.llm_caption_ls = []
        with open(generated_caption_dir, 'r') as input_file:
            for line in input_file:
                # 解析 JSON 数据
                llm_data = json.loads(line)
                self.llm_caption_ls.append(llm_data)

        end_t = time.time()
        if torch.distributed.get_rank() == 0:
            print(f'reading llm data is {(end_t-start_t)/60} min')

        self.transforms = transforms
        self.transforms_strong = transforms_strong
        self.return_data_identifier = return_data_identifier
        # logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.llm_caption_ls)

    def __getitem__(self, idx):
        data = self.llm_caption_ls[idx]
        key = list(data.keys())[0]
        llm_caption = data[key]
        image_path = f'{key}.jpg'
        image_path = f'public_data/cc/training/{image_path}' 
        image_stream = io.BytesIO(image_path)
        images = self.transforms(Image.open(image_stream))
        images_strong = self.transforms_strong(Image.open(image_stream))

        if self.return_filename:
            return images, str(self.images[idx])

        # caption = data['caption']
        # texts = tokenize([str(caption)], truncate=True)[0]
        llm_texts = tokenize([llm_caption], truncate=True)[0]

        if self.return_data_identifier:
            return images_strong, images, llm_texts, 0
        return images_strong, images, llm_texts



class OssDataset_with_transagg(Dataset):
    def __init__(self, input_filename, transforms_strong, transforms, sep="\t",
                 return_data_identifier=False, return_filename=False):

        self.file_path_template = '/mnt_rela/wangyabing.wyb/datasets/laion_cir/laion_cir_template'
        self.file_path_llm = '/mllm_native/wangyabing.wyb/datasets/laion_cir/laion_chatgpt_16k'
        # 读取 JSON 文件
        with open(f'/mnt_rela/wangyabing.wyb/datasets/laion_cir/laion_combined_info.json', 'r') as f:
            self.data_ls = json.load(f)

        llm_caption_path = '/mnt_rela/wangyabing.wyb/datasets/laion_cir/combined_llm_caption.json'
        with open(llm_caption_path) as llm_json:
            self.llm_caption_dict = json.load(llm_json)

        self.transforms = transforms
        self.transforms_strong = transforms_strong
        self.return_data_identifier = return_data_identifier
        # logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

    def __len__(self):
        return len(self.data_ls)

    def __getitem__(self, idx):
        data = self.data_ls[idx]
        ref_image_id = data['ref_image_id']
        relative_cap = data['relative_cap']
        tgt_image_id = data['tgt_image_id']
        data_type = data['type']
        if data_type == 'llm':
            img_path = self.file_path_llm
        else:
            img_path = self.file_path_template
        # /mnt_rela/wangyabing.wyb/datasets/laion_cir/laion_cir_templat

        ref_img_id_str = str(ref_image_id).zfill(7)
        tgt_img_id_str = str(tgt_image_id).zfill(7)
        ref_img_path = os.path.join(img_path, f'{ref_img_id_str}.png')
        tgt_img_path = os.path.join(img_path, f'{tgt_img_id_str}.png')

        ref_llm_caption, tgt_llm_caption = self.llm_caption_dict[f'{ref_img_id_str}.png'].split('.', 1)[0],  self.llm_caption_dict[f'{tgt_img_id_str}.png'].split('.', 1)[0]
        
        ref_llm_caption = tokenize([ref_llm_caption], truncate=True)[0]
        tgt_llm_caption = tokenize([tgt_llm_caption], truncate=True)[0]

        ref_img = Image.open(ref_img_path)
        tgt_img = Image.open(tgt_img_path)
        
        if ref_img.mode == 'RGB':
            ref_img = ref_img.convert('RGB')
        else:
            ref_img = ref_img.convert('RGBA')

        ref_images = self.transforms(ref_img)

        if tgt_img.mode == 'RGB':
            tgt_img = tgt_img.convert('RGB')
        else:
            tgt_img = tgt_img.convert('RGBA')

        tgt_images = self.transforms(tgt_img)

        if self.return_filename:
            return images, str(self.images[idx])

        relative_texts = f'a photo of * that {relative_cap}'
        relative_texts = tokenize([relative_texts], truncate=True)[0]

        if self.return_data_identifier:
            return ref_images, tgt_images, relative_texts, 0
        return ref_images, tgt_images, relative_texts, ref_llm_caption, tgt_llm_caption




class OssDataset_with_synthtriplet(Dataset):
    def __init__(self, input_filename, transforms_strong, transforms, sep="\t",
                 return_data_identifier=False, return_filename=False):
        self.id_path = '/mllm_native/wangyabing.wyb/datasets/Compodiff_dataset_process/filtered_ids_8.7w.txt'

        with open(self.id_path) as f:
            self.id_list = f.readlines()
        

        self.transforms = transforms
        self.transforms_strong = transforms_strong
        self.return_data_identifier = return_data_identifier
        # logging.debug('Done loading data of {} samples'.format(len(self.images)))
        self.return_filename = return_filename

        self.image_root = '/mllm_native/wangyabing.wyb/datasets/Compodiff_dataset'
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        f_id = self.id_list[idx].strip('\n')
      

        ref_img_path = os.path.join(self.image_root, f'{f_id}.source_image.jpg')
        tgt_img_path = os.path.join(self.image_root, f'{f_id}.target_image.jpg')
        cap_json =  os.path.join(self.image_root, f'{f_id}.json')
        with open(cap_json, 'r', encoding='utf-8') as f:
            cap_data = json.load(f)
        ref_caption = cap_data.get("source_caption", "")
        ref_caption = tokenize([ref_caption], truncate=True)[0]

        tgt_caption = cap_data.get("target_caption", "")
        relative_texts = f'a photo of * that {tgt_caption}'
        relative_texts = tokenize([relative_texts], truncate=True)[0]

        ref_img = Image.open(ref_img_path)
        tgt_img = Image.open(tgt_img_path)
        
        if ref_img.mode == 'RGB':
            ref_img = ref_img.convert('RGB')
        else:
            ref_img = ref_img.convert('RGBA')
        
        if tgt_img.mode == 'RGB':
            tgt_img = tgt_img.convert('RGB')
        else:
            tgt_img = tgt_img.convert('RGBA')

        ref_images = self.transforms(ref_img)
        tgt_images = self.transforms(tgt_img)

        if self.return_data_identifier:
            return ref_images, tgt_images, relative_texts, 0
        return ref_images, tgt_images, relative_texts, ref_caption, tgt_caption




@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes = json.load(open(sizes_filename, 'r'))
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )
    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_csv_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_oss_dataset(args, preprocess_fn_strong, preprocess_fn, is_train, input_filename=None):
    

    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.stage == 2:
        if args.ft_dataset == 'transAgg':
            dataset = OssDataset_with_transagg(
            input_filename,
            preprocess_fn_strong,
            preprocess_fn,
            sep=args.csv_separator)
        elif args.ft_dataset == 'compodiff':
            dataset = OssDataset_with_synthtriplet(
                input_filename,
                preprocess_fn_strong,
                preprocess_fn,
                sep=args.csv_separator)
    else:
        dataset = OssDataset_with_caption(
        input_filename,
        preprocess_fn_strong,
        preprocess_fn,
        sep=args.csv_separator)
        
        
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=train_collate_fn,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



#
def get_imgnet_r(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    path_data = os.path.join(args.root_data, 'imgnet/imagenet-r')
    dataset = CustomFolder(path_data, transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sampler)


def get_directory_dataset(args, preprocess_fn, is_train, input_filename=None):
    if input_filename is None:
        input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CustomFolder(
        input_filename,
         transform=preprocess_fn)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'imgnet_r':
        return get_imgnet_r
    elif dataset_type == 'fashion-iq':
        return get_fashion_iq
    elif dataset_type == 'cirr':
        return get_cirr
    elif dataset_type == 'directory':
        return get_directory_dataset
    elif dataset_type == 'oss':
        return get_oss_dataset
    elif dataset_type == "csv":
        return get_csv_dataset        
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train_strong, preprocess_train, preprocess_val = preprocess_fns
    data = {}
    dataset_type_val = getattr(args, 'dataset_type_val', args.dataset_type)
    
    args.dataset_type = 'oss'
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train_strong, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type_val)(
            args, preprocess_val, is_train=False)
    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")
    return data
