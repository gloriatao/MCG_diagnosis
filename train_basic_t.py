import datetime, os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.load_train import load_train
from datasets.load_val import load_val
from engines.engine_basic import train_one_epoch
from models.MCGNet_B_single import backbone
from timm.scheduler.cosine_lr import CosineLRScheduler
from apex import amp
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# number of trainable params: 5,144,974

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int) # 8
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # parameters
    parser.add_argument('--output_dir', default='checkpoints/basic_basic_t', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0, 1], help='device to use for training / testing')
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic_basic_t/checkpoint_0.8294314381270903.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    print(os.environ)
    device = torch.device(args.device)

 
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = backbone().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=1e-4, weight_decay=0.05)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = torch.nn.DataParallel(model, device_ids=args.gpus)

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad==False)
    print('number of frozen params:', n_parameters)

    dataset_train = load_train(fold=0, bs=2)    
    dataset_val_ni = load_val(fold=0, type='NI')
    dataset_val_cta = load_val(fold=0, type='CTA')
    dataset_val_spect = load_val(fold=0, type='SPECT')
    
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, drop_last=True, num_workers=args.num_workers)
    data_loader_val_ni = DataLoader(dataset_val_ni, batch_size=args.batch_size*8, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_spect = DataLoader(dataset_val_spect, batch_size=args.batch_size*8, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_cta = DataLoader(dataset_val_cta, batch_size=args.batch_size*8, drop_last=False, shuffle=False, num_workers=args.num_workers)

    num_steps = int(args.epochs * len(data_loader_train))
    warmup_steps = int(1 * len(data_loader_train))
    lr_scheduler = CosineLRScheduler(
                                    optimizer,
                                    t_initial=num_steps,
                                    lr_min=5e-6,
                                    warmup_lr_init=5e-7,
                                    warmup_t=warmup_steps,
                                    cycle_limit=4,
                                    t_in_epochs=False,
                                )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # input = checkpoint['input']
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']
            args.start_iteration = checkpoint['iteration']+1

    print("Start training")
    start_time = time.time()

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    train_log_dir = Path(args.output_dir+'/train_log')
    val_log_dir = Path(args.output_dir + '/val_log')
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)
    if os.path.isdir(train_log_dir) == False:
        os.mkdir(train_log_dir)
    if os.path.isdir(val_log_dir) == False:
        os.mkdir(val_log_dir)

    # train loop--------------------------------------
    benchmark_metric, steps = 0.0, 0.0
    iteration = args.start_iteration
    print('start iteration:', iteration)
    for epoch in range(args.start_epoch, args.epochs):
        benchmark_metric, iteration = train_one_epoch(model, data_loader_train, optimizer, device, 
                                                      epoch, train_log_dir, output_dir, lr_scheduler, steps, args.clip_max_norm,
                                                    data_loader_val_ni, data_loader_val_spect, data_loader_val_cta,
                                                    val_log_dir, benchmark_metric, iteration)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
