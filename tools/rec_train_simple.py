import os
import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
import random
import time
import shutil
import traceback
from importlib import import_module

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torchocr.networks import build_model, build_loss
from torchocr.datasets import build_dataloader
from torchocr.utils import get_logger, weight_init, load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter

'''0. Utils function'''
import torch.nn.init as init
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

'''1. Load Config'''
config_path = 'config/rec_train_config.py'
config_path = os.path.abspath(os.path.expanduser(config_path))
module_name = os.path.basename(config_path)[:-3]
config_dir = os.path.dirname(config_path)
sys.path.insert(0, config_dir)
mod = import_module(module_name)
cfg = mod.config
os.makedirs(cfg.train_options['checkpoint_save_dir'], exist_ok=True)
logger = get_logger('torchocr', log_file=os.path.join(cfg.train_options['checkpoint_save_dir'], 'train.log'))
# logger.info(cfg)   ## Checked!

'''2. Set CUDA '''
to_use_device = 'cuda'
seed = cfg['SEED']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

'''3. Build Model '''
import copy
from addict import Dict
from torchocr.networks.architectures.RecModel import RecModel
config_model = copy.deepcopy(cfg['model'])
net = RecModel(Dict(config_model))
net.apply(weight_init)
net = net.to(to_use_device)
net.train()
#logger.info(net)
print(net.parameters())
logger.info(net.parameters())

'''3.1 Get finetune layers '''
### to do 

'''4. Build optimizer '''


'''5. Build Scheduler '''

'''6. Build Loss '''

'''7. Build Dataset & DataLoader '''

'''8. Start Training'''
