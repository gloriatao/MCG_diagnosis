import os
import random
from pathlib import Path
import numpy as np
import torch 
from torch.utils.data import DataLoader
from datasets.load_val_ischemia import load_val
# from datasets.load_val_ischemia_backup import load_val
from engines.engine_gcn_inference import inference
from models.Net_SPECT_old import backbone as Net_SPECT
from models.Net_CTA import backbone as Net_CTA
from models.MCGNet_B_single import backbone as Net_ISCHEMIA
from models.MCGNet_GCN3 import Fusion
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# number of trainable params: 5,144,974

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int) # 8
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # parameters
    parser.add_argument('--fold', default=3, type=int)
    parser.add_argument('--output_dir', default='checkpoints/basic', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0], help='device to use for training / testing')
    parser.add_argument('--resume_spect', default='checkpoints_new/checkpoint_spect.pth', help='resume from checkpoint')
    parser.add_argument('--resume_cta', default='checkpoints_new/checkpoint_cta.pth', help='resume from checkpoint')
    parser.add_argument('--resume_ischemia', default='checkpoints_new/checkpoint_ischemia.pth', help='resume from checkpoint')
    parser.add_argument('--resume_fusion', default='checkpoints_new/checkpoint_fusion.pth', help='resume from checkpoint')

    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    
    device = torch.device(args.device)

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

    dataset_val_ni = load_val(fold=args.fold, type='NI')
    dataset_val_cta = load_val(fold=args.fold, type='CAG')
    dataset_val_spect = load_val(fold=args.fold, type='SPECT')
    
    data_loader_val_ni = DataLoader(dataset_val_ni, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_spect = DataLoader(dataset_val_spect, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_cta = DataLoader(dataset_val_cta, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)

    model_spect.load_state_dict(torch.load(args.resume_spect, map_location='cpu')['model'])
    model_cta.load_state_dict(torch.load(args.resume_cta, map_location='cpu')['model'])
    model_ischemia.load_state_dict(torch.load(args.resume_ischemia, map_location='cpu')['model'])
    model_fusion.load_state_dict(torch.load(args.resume_fusion, map_location='cpu')['model'])

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    inference(model_spect, model_cta, model_ischemia, model_fusion, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device, args.fold)

        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
