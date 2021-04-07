import os
import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import random
import time
import numpy as np
from tqdm import tqdm
from importlib import import_module

import torch
from torch import nn

from torchocr.networks import build_model, build_loss
from torchocr.postprocess import build_post_process
from torchocr.datasets import build_dataloader
from torchocr.utils import get_logger, weight_init, load_checkpoint, save_checkpoint
from torchocr.metrics import DetMetric

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/det_train_db_config.py', help='train config file path')
    args = parser.parse_args()
    # 解析.py文件
    config_path = os.path.abspath(os.path.expanduser(args.config))
    assert os.path.isfile(config_path)
    if config_path.endswith('.py'):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        return mod.config
        # cfg_dict = {
        #     name: value
        #     for name, value in mod.__dict__.items()
        #     if not name.startswith('__')
        # }
        # return cfg_dict
    else:
        raise IOError('Only py type are supported now!')

def build_optimizer(params, config):
    """
    优化器
    Returns:
    """
    from torch import optim
    opt_type = config.pop('type')
    opt = getattr(optim, opt_type)(params, **config)
    return opt

cfg = parse_args()
train_options = cfg.train_options

#################################
net = build_model(cfg['model'])
net = nn.DataParallel(net)
to_use_device = torch.device(train_options['device'] if torch.cuda.is_available() and ('cuda' in train_options['device']) else 'cpu')
net = net.to(to_use_device)
net.train()

resume_from = './checkpoints/ch_det_server_db_res18.pth'
optimizer = build_optimizer(net.parameters(), cfg['optimizer'])

ckpt = torch.load(resume_from, map_location='cpu')
model_dict = net.state_dict()
pretrained_dict = ckpt['state_dict']

txt_file = os.path.join('/home/junlin/Git/github/dbnet_pytorch/test_results', 'model_state.txt')
txt_f = open(txt_file, 'w')

for j in model_dict:
    txt_f.write(j)
txt_f.write('######')
for k in pretrained_dict:
    txt_f.write(k)

txt_f.close()

# net, _resumed_optimizer, global_state = load_checkpoint(net, resume_from, to_use_device, optimizer, third_name=train_options['third_party_name'])
# print(net)