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

from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import pdb
import wandb
import logging
import torch.nn.functional as F
from third_party.open_clip.clip import tokenize, _transform
from third_party.open_clip.simple_tokenizer import SimpleTokenizer
from utils import is_master


def gather(x, world_size, dist, rank):
    gathered_x = [
        torch.zeros_like(x) for _ in range(world_size)
    ]
    dist.all_gather(gathered_x, x)
    all_x = torch.cat(
        [x]
        + gathered_x[:rank]
        + gathered_x[rank + 1:]
    )
    return all_x

def get_loss(model, images, texts, loss_img, loss_txt, args, data_identifier=-1):
    if data_identifier == 1:
        # ImageNet dataset
        image_features, text_features, logit_scale = model(images, texts, extra=True)
    else:
        image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # this is needed to send gradients back everywhere.
        # Image loss.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss


def get_text_features_stage2(lora, adapters, model, token_features, image_hidden, relative_texts, args):
    text = tokenize("a photo of * ")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)

    text = torch.cat([text, relative_texts], dim=0)
    token_features = torch.cat([token_features, token_features], dim=0)
    image_hidden = torch.cat([image_hidden, image_hidden], dim=0)
    id_split = tokenize(["*"])[0][1]    
    text_features = model.encode_text_img_retrieval(args, adapters, text, token_features, image_hidden, split_ind=id_split, repeat=False, lora_layers=lora)
    # text_features = model.encode_text_img(text, token_features)
    return text_features


def get_text_features(adapters, model, token_features, image_hidden, args):
    text = tokenize("a photo of * ")
    text = text.cuda(args.gpu, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    id_split = tokenize(["*"])[0][1]    
    text_features = model.encode_text_img_retrieval(args, adapters, text, token_features, image_hidden, split_ind=id_split, repeat=False)
    # text_features = model.encode_text_img(text, token_features)
    return text_features


def get_loss_img2text_stage2(model, lora, img2text, ref_images, tgt_images, ref_llm_caption, tgt_llm_caption, loss_img, loss_txt, loss_triplet, relative_texts, args, memory=None):
    with torch.no_grad():
        ref_image_features, image_hidden = model.encode_image(ref_images, return_hidden=True)
        tgt_image_features, _ = model.encode_image(tgt_images, return_hidden=True)
        ref_llm_features = model.encode_text(ref_llm_caption.to(ref_images.device))

    token_features = img2text(ref_image_features)
    adapters = img2text.module.get_adapters()
    image_hidden = img2text.module.map_to_text_space(image_hidden)

    text_features_all = get_text_features_stage2(lora.module.lora_layers, adapters, model, token_features, image_hidden, relative_texts, args)

    bsz = len(text_features_all)
    text_features_map, text_features = text_features_all[:bsz//2], text_features_all[bsz//2:] 

    image_features = tgt_image_features / tgt_image_features.norm(dim=-1, keepdim=True)
    ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)   
    ref_llm_features = ref_llm_features / ref_llm_features.norm(dim=-1, keepdim=True)   
    text_features_map = text_features_map / text_features_map.norm(dim=-1, keepdim=True)   
    
    logit_scale = 1. / args.temperature
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_image_features, image_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
  
        all_ref_image_features = gather(ref_image_features, world_size, dist, rank)
        all_text_features_norm = gather(text_features_norm, world_size, dist, rank)
        all_ref_llm_features = gather(ref_llm_features, world_size, dist, rank)
        all_text_features_map = gather(text_features_map, world_size, dist, rank)

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)


        logits_per_image = logit_scale * all_image_features @ all_text_features_norm.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t() #torch.cat([logits_per_image.t(), sim], dim=1)


        # 设置对角线元素为一个非常小的值（例如负无穷）
        bsz = len(logits_per_text)
        k = args.topk
        mask = torch.eye(bsz, dtype=torch.bool).to(ref_image_features.device)
        matrix_no_diag = logits_per_text.masked_fill(mask, float('-inf'))
        top_values, top_indices = torch.topk(matrix_no_diag, k, dim=1)

        sim = all_image_features @ all_image_features.t()

        min_vals, _ = sim.min(dim=1, keepdim=True)  # 每行最小值
        max_vals, _ = sim.max(dim=1, keepdim=True)  # 每行最大值

        eps = 1e-8
        matrix_scaled = (sim - min_vals) / (max_vals - min_vals + eps)

        mix_weight = torch.gather(matrix_scaled, dim=1, index=top_indices).unsqueeze(-1)
        image_features_expand = all_image_features.unsqueeze(1).expand(bsz, bsz, all_image_features.size(-1))

        indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, all_image_features.size(-1))  # shape: [1024, 10, 768]
        _img_feat_mix = torch.gather(image_features_expand, 1, indices_expanded)  # shape: [bsz, top, dim]

        img_feat_mix = (mix_weight * all_image_features.unsqueeze(1)) + (1. - mix_weight) * _img_feat_mix
        mix_sim = torch.matmul(img_feat_mix, all_text_features_norm.unsqueeze(-1)).squeeze(-1)
        logits_per_text = torch.cat([logits_per_text, mix_sim], dim=-1)
        loss_txt_val = loss_txt(logits_per_text, ground_truth)


        loss_map = contrastive_loss(all_ref_image_features, all_text_features_map, logit_scale, loss_img, loss_txt, ground_truth)

        logits_llm_t = logit_scale * all_ref_llm_features @ all_text_features_map.t()
        logist_llm_v = logit_scale * all_ref_llm_features @ all_image_features.t()
        p = torch.softmax(logits_llm_t, dim=-1)
        q = torch.softmax(logist_llm_v, dim=-1)
        kl_loss = F.kl_div(torch.log(p), q, reduction='batchmean')
    
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2 + loss_map + kl_loss * args.loss_consistency
    
    return total_loss



def get_loss_img2text(model, img2text, images_strong, images, loss_img, loss_txt, llm_texts, args, memory=None):
    with torch.no_grad():
        ref_image_features, image_hidden = model.encode_image(images_strong, return_hidden=True)
        image_features, _ = model.encode_image(images, return_hidden=True)
        llm_features = model.encode_text(llm_texts.to(images.device))
    token_features = img2text(ref_image_features)
    adapters = img2text.module.get_adapters()
    image_hidden = img2text.module.map_to_text_space(image_hidden)

    text_features = get_text_features(adapters, model, token_features, image_hidden, args)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    ref_image_features = ref_image_features / ref_image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)    
    llm_features = llm_features / llm_features.norm(dim=-1, keepdim=True)    
    
    logit_scale=1./args.temperature
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        all_llm_features = gather(llm_features, world_size, dist, rank)
        all_ref_image_features = gather(ref_image_features, world_size, dist, rank)

        ground_truth = torch.arange(len(all_image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logits_per_image.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

        logits_llm_t = logit_scale * all_llm_features @ all_text_features.t()
        logist_llm_v = logit_scale * all_llm_features @ all_image_features.t()
        p = torch.softmax(logits_llm_t, dim=-1)
        q = torch.softmax(logist_llm_v, dim=-1)
        kl_loss = F.kl_div(torch.log(p), q, reduction='batchmean')

        if args.loss_ref_txt > 0.:
            loss_ref_txt = contrastive_loss(all_ref_image_features, all_text_features, logit_scale, loss_img, loss_txt, ground_truth)
        else:
            loss_ref_txt = 0.
    
    else:
        ground_truth = torch.arange(len(image_features)).long()
        if args.gpu is not None:
            ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        # Image loss.
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss_img_val = loss_img(logits_per_image, ground_truth)
        logits_per_text = logit_scale * text_features @ image_features.t()
        loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2 + kl_loss * args.loss_consistency + loss_ref_txt * args.loss_ref_txt
    return total_loss




def contrastive_loss(all_image_features, all_text_features, logit_scale, loss_img, loss_txt, ground_truth):
    logits_per_image = logit_scale * all_image_features @ all_text_features.t()
    loss_img_val = loss_img(logits_per_image, ground_truth)
    logits_per_text = logits_per_image.t()
    loss_txt_val = loss_txt(logits_per_text, ground_truth)
    return (loss_img_val + loss_txt_val) / 2

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def train(model, img2text, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None, loral=None):
    os.environ["WDS_EPOCH"] = str(epoch)
    model.eval()
    dataloader, sampler = data['train'].dataloader,  data['train'].sampler
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_triplet = nn.TripletMarginLoss(margin=0.1, p=2)

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        if args.stage == '2':
            ref_images, tgt_images, relative_texts, ref_llm_caption, tgt_llm_caption = batch[0], batch[1], batch[2], batch[3], batch[4]

            if len(batch) == 3 and args.use_debiased_sampler:
                data_identifier = torch.unique(batch[2])[0].numpy()
            else:
                data_identifier = -1
            if args.gpu is not None:
                ref_images = ref_images.cuda(args.gpu, non_blocking=True)
                tgt_images = tgt_images.cuda(args.gpu, non_blocking=True)
                relative_texts = relative_texts.cuda(args.gpu, non_blocking=True)
                ref_llm_caption = ref_llm_caption.cuda(args.gpu, non_blocking=True)
                # tgt_llm_caption = tgt_llm_caption.cuda(args.gpu, non_blocking=True)

        else:
            images_strong, images, llm_texts = batch[0], batch[1], batch[2]
            if len(batch) == 3 and args.use_debiased_sampler:
                data_identifier = torch.unique(batch[2])[0].numpy()
            else:
                data_identifier = -1
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                images_strong = images_strong.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model
        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                if args.stage == '2':
                    total_loss = get_loss_img2text_stage2(m, loral, img2text, ref_images, tgt_images, ref_llm_caption, tgt_llm_caption, loss_img, loss_txt, loss_triplet, relative_texts, args, data_identifier)
                else:
                    total_loss = get_loss_img2text(m, img2text, images_strong, images, loss_img, loss_txt, llm_texts, args, data_identifier)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss = get_loss_img2text(m, img2text, images, loss_img, loss_txt, args, data_identifier)
            total_loss.backward()
            optimizer.step()
        
                

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 5) == 0:
            num_samples = i * len(ref_images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {1. / args.temperature:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})