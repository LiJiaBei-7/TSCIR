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
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import partial
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
import pickle

from utils import is_master

def prepare_img(img_file, transform):
    return transform(Image.open(img_file))

def visualize_composed_results(lora, model, img2text, args, prompt, dataloader):
    model.eval()
    img2text.eval()
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    # text = []
    # id_split = tokenize(["*"])[0][1]
    # for p in prompt:
    #     text_tokens = tokenize(p)
    #     text.append(text_tokens)
    #     assert id_split in text_tokens
    # text = torch.cat(text, dim=0)
    # text = text.cuda(args.gpu, non_blocking=True)
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images.
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            # dict_save = {}
            # dict_save['feats'] = all_image_features.data.cpu().numpy()
            # dict_save['path'] = all_image_filenames
            # with open(path_save,"wb") as f:
            #     pickle.dump(dict_save,f)
    # f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    import json

    with open(query_file, 'r') as file:
        query_data = json.load(file)
    data_list = []
    for query in query_data:
        prompt = []
        ref_image = query['candidate']
        caption = query['captions']
        target_image = query['target']
        query = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', ref_image+'.png')
        target = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', target_image+'.png')

        id_split = tokenize(["*"])[0][1]
        text = f'a photo of * , {caption[0]} and {caption[1]}'
        prompt.append(text)
        logging.info("Prompt is {}".format(text))

        text = torch.tensor(tokenize(text))
        assert id_split in text
        text = text.cuda(args.gpu, non_blocking=True)

        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)
        query_img = query_img.cuda(args.gpu, non_blocking=True)

        img_feature, image_hidden = m.encode_image(query_img, return_hidden=True)
        query_img_feature = img2text(img_feature)
        adapters = img2text.get_adapters()
        image_hidden = img2text.map_to_text_space(image_hidden) 
        composed_feature = m.encode_text_img_retrieval(lora.lora_layers, adapters, text, query_img_feature, image_hidden, split_ind=id_split, repeat=False)

        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        sim_value, indices = torch.sort(similarity, descending=True)
        logging.info("Composed feature result")
        num=50
        # for i, caption in enumerate(prompt):
        #     logging.info("for prompt {}".format(caption))
        #     for j, ind in enumerate(indices[i][:8]):
        #         logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:num])]
                        for i, caption in enumerate(prompt)]
        data_map = {"modified_text":caption, "image_paths":image_paths, "query_path":query, "target_path": target, "similarity": sim_value[0][:num].tolist()}
        data_list.append(data_map)
        # html_txt += make_html(prompt, query, image_paths, args.demo_out, target=target)
    # f.write(html_txt)

    # 指定 JSON 文件的路径
    json_file_path = f'{args.demo_out}/composed_result_output_temp_0.07.json'
    # 将 data_list 写入 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
    print('Finished!')

def visualize_clip_vis(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            # dict_save = {}
            # dict_save['feats'] = all_image_features.data.cpu().numpy()
            # dict_save['path'] = all_image_filenames
            # with open(path_save,"wb") as f:
            #     pickle.dump(dict_save,f)
        
        sim = all_image_features @ all_image_features.t()
        data_list = {"image_paths":all_image_filenames, "similarity": sim}

        # 指定 JSON 文件的路径
        json_file_path = f'{args.demo_out}/CLIP_vis_results.pth'
        torch.save(data_list, json_file_path)
        # 将 data_list 写入 JSON 文件
        # with open(json_file_path, 'w') as json_file:
        #     json.dump(data_list, json_file, indent=4)

        print('Finished!')
        exit()



def visualize_results_only_token(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            # dict_save = {}
            # dict_save['feats'] = all_image_features.data.cpu().numpy()
            # dict_save['path'] = all_image_filenames
            # with open(path_save,"wb") as f:
            #     pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    import json
    with open(query_file, 'r') as file:
        query_data = json.load(file)
    out_data = []
    data_list = []
    with torch.no_grad():
        for query in all_image_filenames:
            
            # query = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', ref_image+'.png')

            # ref_image = query['candidate']
            # caption = query['captions']
            # target_image = query['target']
            # query = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', ref_image+'.png')
            logging.info("retrieve image of {}".format(query))
            transform = _transform(model.visual.input_resolution)
            query_img = prepare_img(query, transform)
            query_img = torch.unsqueeze(query_img, 0)    
            query_img = query_img.cuda(args.gpu, non_blocking=True)

            img_feature, image_hidden = m.encode_image(query_img, return_hidden=True)
            query_img_feature = img2text(img_feature)
            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 
            composed_feature = m.encode_text_img_retrieval(adapters, text, query_img_feature, image_hidden, split_ind=id_split, repeat=False)

            # composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
            # text_feature = m.encode_text(text)
            # text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            similarity = composed_feature @ all_image_features.T
            sim_value, indices = torch.sort(similarity, descending=True)        
            logging.info("Composed feature result")
            num=50
            # for i, caption in enumerate(prompt):
            #     logging.info("for prompt {}".format(caption))
            #     for j, ind in enumerate(indices[i][:num]):
            #         logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
            image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:num])] 
                            for i, caption in enumerate(prompt)]
            data_map = {"image_paths":image_paths, "query_path":query, "similarity": sim_value[0][:num].tolist()}
            data_list.append(data_map)
            # # 指定 JSON 文件的路径
            # json_file_path = f'{args.demo_out}/result_output.json'
            # # 将 data_list 写入 JSON 文件
            # with open(json_file_path, 'w') as json_file:
            #     json.dump(data_list, json_file, indent=4)
            # exit()
        # html_txt += make_html(prompt, query, image_paths, args.demo_out)
    # f.write(html_txt)

    # 指定 JSON 文件的路径
    json_file_path = f'{args.demo_out}/re_result_output.json'
    # 将 data_list 写入 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

    print('Finished!')

    


def visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            # dict_save = {}
            # dict_save['feats'] = all_image_features.data.cpu().numpy()
            # dict_save['path'] = all_image_filenames
            # with open(path_save,"wb") as f:
            #     pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    import json
    with open(query_file, 'r') as file:
        query_data = json.load(file)
    out_data = []
    data_list = []
    with torch.no_grad():
        for query in query_data:
            ref_image = query['candidate']
            caption = query['captions']
            target_image = query['target']
            query = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', ref_image+'.png')
            logging.info("retrieve image of {}".format(query))
            transform = _transform(model.visual.input_resolution)
            query_img = prepare_img(query, transform)
            query_img = torch.unsqueeze(query_img, 0)    
            query_img = query_img.cuda(args.gpu, non_blocking=True)

            img_feature, image_hidden = m.encode_image(query_img, return_hidden=True)
            query_img_feature = img2text(img_feature)
            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 
            composed_feature = m.encode_text_img_retrieval(adapters, text, query_img_feature, image_hidden, split_ind=id_split, repeat=False)

            # composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
            # text_feature = m.encode_text(text)
            # text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            similarity = composed_feature @ all_image_features.T
            sim_value, indices = torch.sort(similarity, descending=True)        
            logging.info("Composed feature result")
            for i, caption in enumerate(prompt):
                logging.info("for prompt {}".format(caption))
                for j, ind in enumerate(indices[i][:8]):
                    logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
            image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                            for i, caption in enumerate(prompt)]
            data_map = {"caption":caption, "image_paths":image_paths, "query_path":query, "similarity": sim_value[0][:8].tolist()}
            data_list.append(data_map)
            # # 指定 JSON 文件的路径
            # json_file_path = f'{args.demo_out}/result_output.json'
            # # 将 data_list 写入 JSON 文件
            # with open(json_file_path, 'w') as json_file:
            #     json.dump(data_list, json_file, indent=4)
            # exit()
        # html_txt += make_html(prompt, query, image_paths, args.demo_out)
    # f.write(html_txt)

    # 指定 JSON 文件的路径
    json_file_path = f'{args.demo_out}/result_output.json'
    # 将 data_list 写入 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

    print('Finished!')



def _visualize_results(model, img2text, args, prompt, dataloader):        
    model.eval()
    img2text.eval()   
    if not os.path.exists(args.demo_out):
        os.makedirs(args.demo_out)        
    if not os.path.exists(os.path.join(args.demo_out, "images")):
        os.makedirs(os.path.join(args.demo_out, "images"))
    text = []
    id_split = tokenize(["*"])[0][1]
    for p in prompt:
        text_tokens = tokenize(p)
        text.append(text_tokens)
        assert id_split in text_tokens
    text = torch.cat(text, dim=0)    
    text = text.cuda(args.gpu, non_blocking=True)    
    all_image_features, all_image_filenames = [], []
    m = model.module if args.distributed or args.dp else model
    query_file = args.query_file
    path_save = os.path.join("./data", args.retrieval_data.split('/')[-1].split('.')[0]+".pkl")
    if os.path.exists(path_save):
        with open(path_save, 'rb') as f:
            data = pickle.load(f)
        all_image_features = data['feats']
        all_image_filenames = data['path']
        all_image_features = torch.from_numpy(all_image_features).cuda(args.gpu, non_blocking=True)
    else:
        ## Extract features of target images. 
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, filenames = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                image_features = m.encode_image(images)           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
                all_image_features.append(image_features)
                for name in filenames:
                    all_image_filenames.append(name)
            all_image_features = torch.cat(all_image_features, dim=0)
            dict_save = {}
            dict_save['feats'] = all_image_features.data.cpu().numpy()
            dict_save['path'] = all_image_filenames
            with open(path_save,"wb") as f:
                pickle.dump(dict_save,f)
    f = open(os.path.join(args.demo_out, "index.html"), 'w')
    html_txt = """"""
    ## For each domain, compute composed features and evaluate.
    import json
    with open(query_file, 'r') as file:
        query_data = json.load(file)
    out_data = []
    for query in query_data:
        query = os.path.join('/mnt_rela/wangyabing.wyb/datasets/Fashion-IQ/images', query+'.png')
        logging.info("retrieve image of {}".format(query))
        transform = _transform(model.visual.input_resolution)
        query_img = prepare_img(query, transform)
        query_img = torch.unsqueeze(query_img, 0)    
        query_img = query_img.cuda(args.gpu, non_blocking=True)
        img_feature = m.encode_image(query_img) 
        query_img_feature = img2text(img_feature)
        composed_feature = m.encode_text_img_vis(text, query_img_feature, split_ind=id_split)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_feature = m.encode_text(text)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        similarity = composed_feature @ all_image_features.T
        _, indices = torch.sort(similarity, descending=True)        
        logging.info("Composed feature result")
        for i, caption in enumerate(prompt):
            logging.info("for prompt {}".format(caption))
            for j, ind in enumerate(indices[i][:8]):
                logging.info("top {} filename {}".format(j, all_image_filenames[ind]))
        image_paths = [[all_image_filenames[ind] for j, ind in enumerate(indices[i][:8])] 
                        for i, caption in enumerate(prompt)]
        html_txt += make_html(prompt, query, image_paths, args.demo_out)
    f.write(html_txt)

def make_html(prompts, query_image, images, path_html, target=None):
    import shutil
    html_all = """"""        
    for i in range(len(prompts)):
        prompt = prompts[i]            
        query_image_local = os.path.join(path_html, "images", query_image.split("/")[-1])
        query_image_local_path = os.path.join("images", query_image.split("/")[-1])
        target_image_local_path = os.path.join("images", target.split("/")[-1])
        shutil.copy(query_image, query_image_local)
        image_list = images[i]        
        html = """<table><tr>"""    
        html += """<td><p style="display:inline-block;vertical-align;font-size:20px">%s</p></td>"""%(prompt)
        html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(query_image_local_path)
        if target is not None:
            html += """<td><p style="margin-right: 50px;"><img src="%s" height="100"></p></td>"""%(target_image_local_path)
        for image in image_list:
            image_local = os.path.join(path_html, "images", image.split("/")[-1])
            image_path = os.path.join("images", image.split("/")[-1])
            shutil.copy(image, image_local)
            html += """<td><img src="%s" height=%s></td>"""%(image_path, 200)
        html += """</tr></table>"""
        html_all += html
    return html_all
    #f.write(html_all)




def evaluate_cirr_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids, text_with_blank_raw = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            # if args.eval_combiner:
            #     composed_feature = img2text(query_image_features, caption_features)
            # else:
            #     query_image_tokens = img2text(query_image_features)
            #     composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]
            caption_features = m.encode_text(caption_only)
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden)

            composed_feature = m.encode_text_img_retrieval(args, adapters, text_with_blank, query_image_tokens, image_hidden,
                                                           split_ind=id_split, repeat=False)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)
            all_mixture_features.append(mixture_features)

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}  
        
        results = []      
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)
            # result = f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            # logging.info(result)
            # results.append(result)

        # print('writting...')
        # with open(args.output_acc_log, 'a') as f:
        #     f.write(f'{args.cur_epoch} Epoch' + '\n')
        #     f.write(results[0])
        #     f.write('\n')
        #     f.write('------------------' + '\n')
            
    return res_all




def evaluate_cirr_test_stage2(model, img2text, lora, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()

    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_composed_plus_image_features = []  
    all_mixture_features = []  
    all_caption_features = []  
    all_ref_paths = []
    all_target_paths = []
    all_answer_paths = []
    all_ids = []

    m = model.module if args.distributed or args.dp else model   
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean()   

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

        for batch in tqdm(query_loader):
            ref_images, text_with_blank, caption_only, ref_paths, pairids, text_with_blank_raw = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                text_with_blank = text_with_blank.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            id_split = tokenize(["*"])[0][1]                        
            for ids in pairids:
                all_ids.append(ids)
            for path in ref_paths:
                all_ref_paths.append(path)

            # if args.eval_combiner:
            #     composed_feature = img2text(query_image_features, caption_features)
            # else:
            #     query_image_tokens = img2text(query_image_features)
            #     composed_feature = m.encode_text_img_retrieval(text_with_blank, query_image_tokens, split_ind=id_split, repeat=False)

            
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]
            caption_features = m.encode_text(caption_only)
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden)

            composed_feature = m.encode_text_img_retrieval(args, lora.lora_layers, adapters, text_with_blank, query_image_tokens, image_hidden,
                                                           split_ind=id_split, repeat=False)
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)
            all_mixture_features.append(mixture_features)

        all_target_paths = np.array(all_target_paths)
        all_ref_paths = np.array(all_ref_paths)
        all_answer_paths = np.array(all_answer_paths)
        res_all = {}
        metrics_func = partial(get_cirr_testoutput, 
                               image_features=torch.cat(all_image_features),
                               reference_names=all_ref_paths,
                               index_names=all_target_paths,
                               id_names=all_ids)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}  
        
        results = []      
        for key, value in feats.items():
            res_all[key] = metrics_func(ref_features=value)
            # result = f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            # logging.info(result)
            # results.append(result)

        # print('writting...')
        # with open(args.output_acc_log, 'a') as f:
        #     f.write(f'{args.cur_epoch} Epoch' + '\n')
        #     f.write(results[0])
        #     f.write('\n')
        #     f.write('------------------' + '\n')
            
    return res_all


def evaluate_circo_test_stage2(model, img2text, lora, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    lora.eval()
    gts_img_ids_list = []
    target_names_list = []
    all_image_features = []  
    all_composed_features = [] 
    all_target_paths = [] 
    query_ids_list = []

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
        
        all_image_features = torch.cat(all_image_features)

    print('Finish.....')

    with torch.no_grad():
        for batch in tqdm(query_loader):
            ref_images, ref_img_ids, relative_caption_token, shared_concept, query_id = batch

            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                # target_images = target_images.cuda(args.gpu, non_blocking=True)
                relative_caption_token = relative_caption_token.cuda(args.gpu, non_blocking=True)
                # caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            # image_features = m.encode_image(target_images)
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]            
            # caption_features = m.encode_text(caption_only)                            
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 

            composed_feature = m.encode_text_img_retrieval(lora.lora_layers, adapters, relative_caption_token, query_image_tokens, image_hidden, split_ind=id_split, repeat=False)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            all_composed_features.append(composed_feature)
            query_ids_list.extend(query_id)
        
        all_composed_features = torch.cat(all_composed_features)
    
    # Compute the similarity
    similarity = all_composed_features @ all_image_features.t()
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu().numpy()  
    sorted_index_names = np.array(all_target_paths)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids_list, sorted_index_names)}

    return queryid_to_retrieved_images


def evaluate_circo_test(model, img2text, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    gts_img_ids_list = []
    target_names_list = []
    all_image_features = []  
    all_composed_features = [] 
    all_target_paths = [] 
    query_ids_list = []

    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)
        
        all_image_features = torch.cat(all_image_features)

    print('Finish.....')

    with torch.no_grad():
        for batch in tqdm(query_loader):
            
            # ref_images, ref_img_ids, relative_caption_token, shared_concept, query_id
            ref_images, ref_img_ids, relative_caption_token, shared_concept, query_id = batch
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                # target_images = target_images.cuda(args.gpu, non_blocking=True)
                relative_caption_token = relative_caption_token.cuda(args.gpu, non_blocking=True)
           
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]            
            # caption_features = m.encode_text(caption_only)                            
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden)

            composed_feature = m.encode_text_img_retrieval(args, adapters, relative_caption_token, query_image_tokens, image_hidden, split_ind=id_split, repeat=False)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            all_composed_features.append(composed_feature)
            query_ids_list.extend(query_id)
        
        all_composed_features = torch.cat(all_composed_features)
    
    # Compute the similarity
    similarity = all_composed_features @ all_image_features.t()
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu().numpy()  
    sorted_index_names = np.array(all_target_paths)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids_list, sorted_index_names)}

    return queryid_to_retrieved_images


def evaluate_circo(model, img2text, lora, args, query_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    lora.eval()
    gts_img_ids_list = []
    target_names_list = []
    all_image_features = []  
    all_composed_features = [] 
    all_target_paths = [] 

    
    all_answer_paths = []
    all_query_image_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            all_target_paths.extend(target_paths)
        
        all_image_features = torch.cat(all_image_features)

    print('Finish.....')

    with torch.no_grad():
        for batch in tqdm(query_loader):
            ref_images, relative_caption_token, shared_concept, target_img_id, gt_img_ids  = batch

            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                # target_images = target_images.cuda(args.gpu, non_blocking=True)
                relative_caption_token = relative_caption_token.cuda(args.gpu, non_blocking=True)
                # caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            # image_features = m.encode_image(target_images)
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]            
            # caption_features = m.encode_text(caption_only)                            
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 

            composed_feature = m.encode_text_img_retrieval(lora.lora_layers, adapters, relative_caption_token, query_image_tokens, image_hidden, split_ind=id_split, repeat=False)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
            
            precisions = get_metrics_circo(composed_feature, all_image_features, all_target_paths, gt_img_ids, target_img_id)

            ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
            ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
            ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
            ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

            assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
            single_gt_labels = torch.tensor(sorted_index_names == target_name)
            recall_at5.append(float(torch.sum(single_gt_labels[:5])))
            recall_at10.append(float(torch.sum(single_gt_labels[:10])))
            recall_at25.append(float(torch.sum(single_gt_labels[:25])))
            recall_at50.append(float(torch.sum(single_gt_labels[:50])))

        map_at5 = np.mean(ap_at5) * 100
        map_at10 = np.mean(ap_at10) * 100
        map_at25 = np.mean(ap_at25) * 100
        map_at50 = np.mean(ap_at50) * 100
        recall_at5 = np.mean(recall_at5) * 100
        recall_at10 = np.mean(recall_at10) * 100
        recall_at25 = np.mean(recall_at25) * 100
        recall_at50 = np.mean(recall_at50) * 100
    
        result = f'map_at5: {map_at5}, map_at10: {map_at10}, map_at25: {map_at25}, map_at50: {map_at50} \n'
        result += f'recall_at5: {recall_at5}, recall_at10: {recall_at10}, recall_at25: {recall_at25}, recall_at50: {recall_at50}'
        logging.info(result)

        print('writting...')
        with open(args.output_acc_log, 'a') as f:
            f.write(f'{args.cur_epoch} Epoch' + '\n')
            f.write(result)
            f.write('\n')
            f.write('------------------' + '\n')

        

def get_metrics_circo(predicted_feature, all_image_features, index_names, gt_img_ids, target_name):
    gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
    similarity = predicted_feature @ all_image_features.t()
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu().numpy()
    print('index_names', np.array(index_names).shape)
    print('sorted_indices', sorted_indices.shape)
    sorted_index_names = np.array(index_names)[sorted_indices]
    print('sorted_index_names', sorted_index_names.shape)
    map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
    precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
    print('precisions', precisions.shape)
    print(torch.arange(1, map_labels.shape[0] + 1).shape)
    precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

    return precisions
    


def evaluate_fashion(model, img2text, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]            
            caption_features = m.encode_text(caption_only)                            
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 

            composed_feature = m.encode_text_img_retrieval(args, adapters, target_caption, query_image_tokens, image_hidden, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                         

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        results = []
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            result = f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logging.info(result)
            results.append(result)

        print('writting...')
        with open(args.output_acc_log, 'a') as f:
            f.write(f'{args.cur_epoch} Epoch' + '\n')
            f.write(f'{args.source_data}' + '\n')
            f.write(results[0])
            f.write('\n')
            f.write(results[-1])
            f.write('\n')
            f.write('------------------' + '\n')

    return metrics


def evaluate_fashion_stage2(model, img2text, lora, args, source_loader, target_loader):
    if not is_master(args):
        return
    model.eval()
    img2text.eval()
    lora.eval()
    all_target_paths = []
    all_answer_paths = []
    all_image_features = []  
    all_query_image_features = []  
    all_composed_features = []  
    all_caption_features = []  
    all_mixture_features = []  
    all_reference_names = []
    all_captions = []     
    m = model.module if args.distributed or args.dp else model
    logit_scale = m.logit_scale.exp()
    logit_scale = logit_scale.mean() 

    with torch.no_grad():
        for batch in tqdm(target_loader):
            target_images, target_paths = batch
            if args.gpu is not None:
                target_images = target_images.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)
            for path in target_paths:
                all_target_paths.append(path)

    with torch.no_grad():
        for batch in tqdm(source_loader):
            ref_images, target_images, target_caption, caption_only, answer_paths, ref_names, captions = batch
            for path in answer_paths:
                all_answer_paths.append(path)
            all_reference_names.extend(ref_names)
            all_captions.extend(captions)
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                target_images = target_images.cuda(args.gpu, non_blocking=True)
                target_caption = target_caption.cuda(args.gpu, non_blocking=True)
                caption_only = caption_only.cuda(args.gpu, non_blocking=True)
            image_features = m.encode_image(target_images)
            query_image_features, image_hidden = m.encode_image(ref_images, return_hidden=True)
            id_split = tokenize(["*"])[0][1]            
            caption_features = m.encode_text(caption_only)                            
            query_image_tokens = img2text(query_image_features)

            adapters = img2text.get_adapters()
            image_hidden = img2text.map_to_text_space(image_hidden) 

            composed_feature = m.encode_text_img_retrieval(args, lora.lora_layers, adapters, target_caption, query_image_tokens, image_hidden, split_ind=id_split, repeat=False)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)            
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)                       
            query_image_features = query_image_features / query_image_features.norm(dim=-1, keepdim=True)   
            mixture_features = query_image_features + caption_features
            mixture_features = mixture_features / mixture_features.norm(dim=-1, keepdim=True)
            composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)

            all_caption_features.append(caption_features)
            all_query_image_features.append(query_image_features)
            all_composed_features.append(composed_feature)            
            all_mixture_features.append(mixture_features)                         

        metric_func = partial(get_metrics_fashion, 
                              image_features=torch.cat(all_image_features),
                              target_names=all_target_paths, answer_names=all_answer_paths)
        feats = {'composed': torch.cat(all_composed_features), 
                 'image': torch.cat(all_query_image_features),
                 'text': torch.cat(all_caption_features),
                 'mixture': torch.cat(all_mixture_features)}
        
        results = []
        for key, value in feats.items():
            metrics = metric_func(ref_features=value)
            result = f"Eval {key} Feature" + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logging.info(result)
            results.append(result)

        print('writting...')
        with open(args.output_acc_log, 'a') as f:
            f.write(f'{args.cur_epoch} Epoch' + '\n')
            f.write(f'{args.source_data}' + '\n')
            f.write(results[0])
            f.write('\n')
            f.write(results[-1])
            f.write('\n')
            f.write('------------------' + '\n')

    return metrics




def get_metrics_coco(image_features, ref_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale.cpu() * image_features @ ref_features.t()).detach().cpu()
    logits_per_ref = logits_per_image.t().detach().cpu()
    logits = {"image_to_ref": logits_per_image, "ref_to_image": logits_per_ref}
    ground_truth = torch.arange(len(ref_features)).view(-1, 1)
    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10, 50, 100]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)
    return metrics


def get_metrics_fashion(image_features, ref_features, target_names, answer_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(target_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(answer_names), len(target_names)).reshape(len(answer_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(answer_names)).int())
    # Compute the metrics
    for k in [1, 5, 10, 50, 100]:
        metrics[f"R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100
    return metrics


def get_metrics_cirr(image_features, ref_features, reference_names, index_names, target_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), 
        len(index_names)).reshape(len(target_names), -1))        
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0], sorted_index_names.shape[1]-1)
                                                                    #sorted_index_names.shape[1] - 1)

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), 
        len(index_names)-1).reshape(len(target_names), -1))

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    for k in [1, 5, 10, 50, 100]:
        metrics[f"recall_R@{k}"] = (torch.sum(labels[:, :k]) / len(labels)).item() * 100

    return metrics


def get_cirr_testoutput(image_features, ref_features, reference_names, index_names, id_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    result_dict = {"version": "rc2", "metric": "recall"}
    for ind in range(len(id_names)):
        pairid = str(id_names[ind].item())
        result_dict[pairid] = []
        for t in range(50):
            result_dict[pairid].append(sorted_index_names[ind][t].replace(".png", ""))
    return result_dict


def get_metrics_imgnet(query_features, image_features, query_labels, target_labels):
    metrics = {}
    num_classes = 7000
    query_onehot = F.one_hot(query_labels, num_classes=num_classes).float()
    target_onehot = F.one_hot(target_labels, num_classes=num_classes).float()
    batches = [(query_features[x:x+100], query_onehot[x:x+100]) for x in range(0, len(query_features), 100)]
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] = 0
        metrics[f"Real2Sketch_P@{k}"] = 0
    for batch in batches:
        feats, labels = batch[0], batch[1]
        logits_per_query = (feats @ image_features.t()).detach().cpu()
        label_matrix = (labels @ target_onehot.t()).detach().cpu()                
        ranking = torch.argsort(logits_per_query, descending=True)
        for k in [1, 5, 10, 50, 100, 200]:
            matrix_k = torch.zeros_like(label_matrix)
            rank_k = ranking[:, :k]
            matrix_k[torch.arange(matrix_k.size(0)).unsqueeze(1), rank_k] = 1
            consistency = matrix_k * label_matrix
            num_correct = torch.sum(consistency, dim=1)
            num_predicted = torch.sum(matrix_k, dim=1)            
            num_total = torch.sum(label_matrix, dim=1)
            recall = torch.mean(num_correct / (num_total+1e-5))
            precision = torch.mean(num_correct / num_predicted)
            metrics[f"Real2Sketch_R@{k}"] += recall * len(feats)
            metrics[f"Real2Sketch_P@{k}"] += precision * len(feats)
    for k in [1, 5, 10, 50, 100, 200]:
        metrics[f"Real2Sketch_R@{k}"] /= len(query_features)
        metrics[f"Real2Sketch_P@{k}"] /= len(query_features)
    return metrics


# CIRCO
@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset, clip_model, img2text, index_features, index_names):
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, img2text,
                                                                                   relative_val_dataset)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_generate_val_predictions(clip_model, img2text, relative_val_dataset):
    """
    Generates features predictions for the validation set of CIRCO
    """
    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']
        reference_image = batch['reference_image'].cuda()

        gt_img_ids = np.array(gt_img_ids).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_img_feature = img2text.img_to_text(reference_image, clip_model, relative_captions)

        predicted_feature = F.normalize(text_img_feature)

        predicted_features_list.append(predicted_feature)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list

