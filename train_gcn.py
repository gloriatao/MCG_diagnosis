import datetime, os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import numpy as np
# from datasets.load_train import load_train
from datasets.load_train import load_train
from datasets.load_val import load_val
from engines.engine_gcn import train_one_epoch
from models.MCGNet_loc import backbone as Net_SPECT
from models.MCGNet_CTA import backbone as  Net_CTA
from models.MCGNet_B_single import backbone as Net_ISCHEMIA
from models.MCGNet_GCN2 import Fusion

from timm.scheduler.cosine_lr import CosineLRScheduler
from apex import amp
import warnings 
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# number of trainable params: 5,144,974

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int) # 8
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # parameters
    parser.add_argument('--output_dir', default='checkpoints/basic_gcn2_ischemia', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0, 1], help='device to use for training / testing')
    parser.add_argument('--resume_spect', default='checkpoints/basic_loc/checkpoint_0.6917864892245696.pth', help='resume from checkpoint')
    parser.add_argument('--resume_cta', default='checkpoints/loc_LCX/checkpoint_0.6366666666666666.pth', help='resume from checkpoint')
    parser.add_argument('--resume_ischemia', default='checkpoints/basic_basic_t/checkpoint_0.8294314381270903.pth', help='resume from checkpoint')
    parser.add_argument('--resume_fusion', default=None, help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    print(os.environ)
    device = torch.device(args.device)

    # the Answer to Life, the Universe and Everything :)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model_spect = Net_SPECT().cuda()
    model_cta = Net_CTA().cuda()
    model_ischemia = Net_ISCHEMIA().cuda()
    with open('/media/cygzz/data/rtao/projects/MCG-NC/data/data001/ADJ_matrix2.csv') as f:
        adj_matrix = torch.tensor(np.loadtxt(f, delimiter=",").astype(np.float32)).unsqueeze(0).unsqueeze(0)
    f.close()
    model_fusion = Fusion(adj=adj_matrix).cuda()
    
    optimizer = torch.optim.AdamW(model_fusion.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=1e-4, weight_decay=0.05)

    model_fusion, optimizer = amp.initialize(model_fusion, optimizer, opt_level="O1")
    model_fusion = torch.nn.DataParallel(model_fusion, device_ids=args.gpus)

    dataset_train = load_train(fold=0, bs=6)
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

    # load 
    model_spect.load_state_dict(torch.load(args.resume_spect, map_location='cpu')['model'])
    model_cta.load_state_dict(torch.load(args.resume_cta, map_location='cpu')['model'])
    model_ischemia.load_state_dict(torch.load(args.resume_ischemia, map_location='cpu')['model'])
    if args.resume_fusion:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_fusion.module.load_state_dict(checkpoint['model'])
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
        benchmark_metric, iteration = train_one_epoch(model_spect, model_cta, model_ischemia, model_fusion, data_loader_train, optimizer, device, 
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
