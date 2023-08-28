import datetime, os
import random
import time
from pathlib import Path
import numpy as np
import torch 
from torch.utils.data import DataLoader
from datasets.load_val import load_val
from engines.engine_basic import inference
# from models.MCGNet_L import backbone
# from models.MCGNet import backbone
# from models.MCGNet_B import backbone
from models.MCGNet_B_single import backbone
# from models.MCGNet_B_mult import backbone
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# number of trainable params: 5,144,974
# {'acc': 0.8294314381270903, 'sens': 0.8633879781420765, 'spec': 0.7758620689655172}

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int) # 8
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # parameters
    parser.add_argument('--output_dir', default='checkpoints/basic', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--gpus', default=[0], help='device to use for training / testing')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic/checkpoint_0.7692307692307693.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic/checkpoint_sens0.9289617486338798.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic_B/checkpoint_0.8051470588235294.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='checkpoints/basic_basic_t/checkpoint_0.8294314381270903.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic_basic_t_multask/checkpoint_0.7692307692307693.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/media/cygzz/data/rtao/projects/MCG-NC/checkpoints/basic_basic_t_mult/checkpoint_0.8846153846153846.pth', help='resume from checkpoint')   
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_iteration', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser

def main(args):
    print(os.environ)
    device = torch.device(args.device)

    # "the Answer to Life, the Universe and Everything"
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = backbone().cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpus)

    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)

    dataset_val_ni = load_val(fold=0, type='NI')
    dataset_val_cta = load_val(fold=0, type='CTA')
    dataset_val_spect = load_val(fold=0, type='SPECT')
    
    data_loader_val_ni = DataLoader(dataset_val_ni, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_spect = DataLoader(dataset_val_spect, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)
    data_loader_val_cta = DataLoader(dataset_val_cta, batch_size=args.batch_size*10, drop_last=False, shuffle=False, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    # path---------------------------------------------
    output_dir = Path(args.output_dir)
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    results = inference(model, data_loader_val_ni, data_loader_val_spect, data_loader_val_cta, device)
    # {'acc': 0.7692307692307693, 'sens': 0.6939890710382514, 'spec': 0.8879310344827587}
    # {'acc': 0.6722408026755853, 'sens': 0.9289617486338798, 'spec': 0.2672413793103448}
    # {'acc': 0.7424749163879598, 'sens': 0.8524590163934426, 'spec': 0.5689655172413793}
    print()
        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
