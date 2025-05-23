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
import logging
from time import gmtime, strftime
from pathlib import Path
import json
from functools import partial
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, LoRA
from eval_utils import evaluate_imgnet_retrieval, evaluate_coco, evaluate_fashion, evaluate_cirr, evaluate_cirr_test, evaluate_circo, evaluate_circo_test
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR, CIRCODataset
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, TargetPad

def load_model(args):
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location=loc)
    opt = checkpoint['args']
    args.ca_layers = opt.ca_layers
    try:
        args.va_layers = opt.va_layers
    except:
        args.va_layers = [4,7,10]


    model, _,_, preprocess_val = load(
            args.model,
            jit=False)
    img2text = IM2TEXT(args=args, embed_dim=model.embed_dim, 
                       middle_dim=args.middle_dim, 
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer) 
    
    if args.stage == 2:
        lora = LoRA(args, original_dim=model.embed_dim, lora_rank=args.lora_rank, layer_num=6)
    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        if args.stage == 2:
            lora.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        if args.stage == 2:
            lora.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            if args.stage == 2:
                convert_weights(lora)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
            if args.stage == 2:
                lora = torch.nn.parallel.DistributedDataParallel(lora, 
                    device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)
            if args.stage == 2:
                lora = torch.nn.DataParallel(lora, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
            if args.stage == 2:
                convert_weights(lora)
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        if args.stage == 2:
            sd_lora = checkpoint["state_dict_lora"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
            sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
        if args.stage == 2:
            if not args.distributed and next(iter(sd_lora.items()))[0].startswith('module'):
                sd_lora = {k[len('module.'):]: v for k, v in sd_lora.items()}
        model.load_state_dict(sd)
        img2text.load_state_dict(sd_img2text)
        if args.stage == 2:
            lora.load_state_dict(sd_lora)
        logging.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
        )
        args.cur_epoch = checkpoint['epoch']
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.stage == 2:
        return model, img2text, lora, preprocess_val
    else:
        return model, img2text, preprocess_val


def setup_log_save(args):
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    if args.dp:
        args.batch_size *= args.world_size
    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)


def main_worker(gpu, ngpus_per_node, log_queue, args):

    import random
    import numpy as np
    seed=args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    # Log and save params.
    setup_log_save(args)
    # Load trained model
    if args.stage == 2:
        model, img2text, lora, preprocess_val = load_model(args)
    else:
        model, img2text, preprocess_val = load_model(args)
    
    cudnn.benchmark = True
    cudnn.deterministic = False   
    root_project = args.dataset_root
    ## Padding option
    if args.target_pad:
        trans_tmp = preprocess_val.transforms
        trans_tmp = [TargetPad(1.25)] + trans_tmp
        preprocess_train = T.Compose(trans_tmp)
        preprocess_val = preprocess_train

     ## Load data for each evaluation dataset and perform evaluation.

    elif args.eval_mode == 'circo':
        # Extract the index features
        classic_val_dataset = CIRCODataset(root_project, 'val', 'classic', preprocess_val)

        # Define the relative validation dataset
        relative_val_dataset = CIRCODataset(root_project, 'val', 'relative', preprocess_val)

        classic_val_dataloader = DataLoader(
            classic_val_dataset,
            batch_size=args.batch_size * 10,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        
        relative_val_dataloader = DataLoader(
            relative_val_dataset,
            batch_size=args.batch_size * 4,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        if args.stage == '2':
            evaluate_circo_stage2(model, img2text, lora, args, relative_val_dataloader, classic_val_dataloader)
        else:
            evaluate_circo(model, img2text, args, relative_val_dataloader, classic_val_dataloader)


    elif args.eval_mode == 'circo_test':
        # Extract the index features
        classic_val_dataset = CIRCODataset(root_project, 'test', 'classic', preprocess_val)

        # Define the relative validation dataset
        relative_val_dataset = CIRCODataset(root_project, 'test', 'relative', preprocess_val)

        classic_val_dataloader = DataLoader(
            classic_val_dataset,
            batch_size=args.batch_size * 10,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        
        relative_val_dataloader = DataLoader(
            relative_val_dataset,
            batch_size=args.batch_size * 4,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        if args.stage == '2':
            queryid_to_retrieved_images = evaluate_circo_test_stage2(model, img2text, lora, args, relative_val_dataloader, classic_val_dataloader)
        else:
            queryid_to_retrieved_images = evaluate_circo_test(model, img2text, lora, args, relative_val_dataloader, classic_val_dataloader)


        submissions_folder_path = Path(args.output_acc_log + f"/circo_test")
        submissions_folder_path.mkdir(exist_ok=True, parents=True)

        with open(submissions_folder_path / f"ciror_test_{args.epoch}.json", 'w+') as file:
            json.dump(queryid_to_retrieved_images, file, sort_keys=True)

    elif args.eval_mode == 'cirr':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs')
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        evaluate_cirr(model, 
                      img2text, 
                      args, 
                      source_dataloader, 
                      target_dataloader)

    elif args.eval_mode == 'cirr_test':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, test=True)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs', 
                              test=True)
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        
        if args.stage == '2':
            results = evaluate_cirr_test_stage2(model, 
                                        img2text, 
                                        lora,
                                        args, 
                                        source_dataloader, 
                                        target_dataloader)
        else:
            results = evaluate_cirr_test(model, 
                                        img2text, 
                                        args, 
                                        source_dataloader, 
                                        target_dataloader)

        out_path = args.output_acc_log  
        out_dir = f'{out_path}/cirr_acc_test'
        os.makedirs(out_dir, exist_ok=True)    
        for key, value in results.items():
            if key == 'composed':
                write_path = f'{out_dir}/{args.epoch}_{key}.json'
                with open(write_path, 'a') as f:
                    json.dump(value, f)
    
    elif args.eval_mode == 'fashion':

        assert args.source_data in ['dress', 'shirt', 'toptee']
        source_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   is_return_target_path=True)
        target_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   mode='imgs')
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        if args.stage == '2':
            evaluate_fashion_stage2(model, img2text, args, source_dataloader, target_dataloader)
        else:
            evaluate_fashion(model, img2text, lora, args, source_dataloader, target_dataloader)

    
def main():
    args = parse_args()

    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    # get the name of the experiments
    if os.path.isfile(args.model):
        model_name = os.path.splitext(os.path.basename(args.model))[0]
    if args.name is None:
        args.name = (f"lr={args.lr}_"
             f"wd={args.wd}_"
             f"agg={args.aggregate}_"
             f"model={model_name}_"
             f"batchsize={args.batch_size}_workers={args.workers}")
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    args.world_size = 1
    main_worker(args.gpu, None, log_queue, args)
    # try:
    #     main_worker(args.gpu, None, log_queue, args)
    # except:
    #     print('evaluation done')


if __name__ == "__main__":
    main()
