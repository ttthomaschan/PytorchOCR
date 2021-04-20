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
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

'''5. Build Scheduler '''
step_size = 60 
gamma = 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

'''6. Build Loss '''
class CTCLoss(nn.Module):
    
    def __init__(self, blank_idx=0, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        batch_size = pred.size(0)
        label, label_length = args['targets'], args['targets_lengths']
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        return {'loss': loss}

loss_dict = copy.deepcopy(cfg['loss'])
loss_func = CTCLoss()
loss_func = loss_func.to(to_use_device)

'''7. Build Dataset & DataLoader '''
with open(cfg.dataset.alphabet, 'r', encoding='utf-8') as file:
    cfg.dataset.characters = ''.join([s.strip('\n') for s in file.readlines()])
from torchocr.datasets.RecDataSet import RecDataLoader, RecTextLineDataset
cfg.dataset.train.dataset.characters = cfg.dataset.characters
train_dataset_dict = Dict(copy.deepcopy(cfg.dataset.train))
train_dataset = RecTextLineDataset(train_dataset_dict.dataset)

class RecCollateFn:
    def __init__(self, *args, **kwargs):
        self.process = kwargs['dataset'].process
        self.t = transforms.ToTensor()

    def __call__(self, batch):
        resize_images = []

        all_same_height_images = [self.process.resize_with_specific_height(_['img']) for _ in batch]
        max_img_w = max({m_img.shape[1] for m_img in all_same_height_images})
        # make sure max_img_w is integral multiple of 8
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        labels = []
        for i in range(len(batch)):
            _label = batch[i]['label']
            labels.append(_label)
            img = self.process.width_pad_img(all_same_height_images[i], max_img_w)
            img = self.process.normalize_img(img)
            img = img.transpose([2, 0, 1])
            resize_images.append(torch.tensor(img, dtype=torch.float))
        resize_images = torch.stack(resize_images)
        return {'img': resize_images, 'label': labels}

train_dataset_dict.loader.collate_fn['dataset'] = train_dataset
collate_fn = RecCollateFn( **train_dataset_dict.loader.collate_fn)  
train_loader = DataLoader(dataset= train_dataset,batch_size = 32, shuffle=True, num_workers = 2, collate_fn=collate_fn)
# train_iter = iter(train_loader)
# logger.info(next(train_iter))

'''8. Check the input(image and label in Dataloader) '''
inputCheck = False
if inputCheck == True:
    img_batch = next(train_iter)['img']
    img_batch = img_batch.to(to_use_device)
    img_grid = torchvision.utils.make_grid(img_batch)
    print(type(img_batch))
    print(img_grid.is_cuda)
    writer = SummaryWriter(comment ='InputData')
    with writer:
        writer.add_graph(net, (img_batch,))
        
        writer.add_image('Image_grid', img_grid)#, dataformats='HWC')
        #writer.add_images("Input Images", img_batch, global_step = 0)

'''9. Start Training'''
from torchocr.metrics import RecMetric
from torchocr.utils import CTCLabelConverter

converter = CTCLabelConverter(cfg.dataset.alphabet)
train_options = cfg.train_options
metric = RecMetric(converter)

logger.info('Training...')

all_step = len(train_loader)
logger.info(f'train dataset has {train_loader.dataset.__len__()} samples,{all_step} in dataloader')
# logger.info(f'eval dataset has {eval_loader.dataset.__len__()} samples,{len(eval_loader)} in dataloader')

best_model = {'best_acc': 0, 'eval_loss': 0, 'model_path': '', 'eval_acc': 0., 'eval_ned': 0.}
start_epoch = 0
global_step = 0

## 开始训练
for epoch in range(start_epoch, train_options['epochs']):
    net.train()
    start = time.time()

    for i, batch_data in enumerate(train_loader):
        current_lr = optimizer.param_groups[0]['lr']
        cur_batch_size = batch_data['img'].shape[0]
        targets, targets_lengths = converter.encode(batch_data['label'])
        batch_data['targets'] = targets
        batch_data['targets_lengths'] = targets_lengths
        # 清零梯度及方向传播
        optimizer.zero_grad()
        batch_data['img'] = batch_data['img'].to(to_use_device)
        output = net.forward(batch_data['img'])
        logger.info(f"output:{output}")
        loss_dict = loss_func(output, batch_data)
        logger.info(loss_dict)
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),5)  # 梯度裁剪，用于防止梯度爆炸和梯度消失
        optimizer.step()
        # 分析打印损失值
        acc_dict = metric(output,batch_data['label'])
        #logger.info(acc_dict)
        acc = acc_dict['n_correct'] / cur_batch_size
        norm_edit_dis = 1 - acc_dict['norm_edit_dis'] / cur_batch_size
        if (i+1) % train_options['print_interval'] == 0:
            interval_batch_time = time.time() - start
            logger.info(f"[{epoch}/{train_options['epochs']}] - "
                        f"[{i+1}/{all_step}] - "
                        f"lr:{current_lr} - "
                        f"loss:{loss_dict['loss'].item():.4f} - "
                        f"acc:{acc:.4f} - "
                        f"norm_edit_dis:{norm_edit_dis:.4f} - "
                        f"time:{interval_batch_time:.4f}") 
            start = time.time()
        global_step += 1
    scheduler.step()

