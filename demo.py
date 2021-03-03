import os
from PIL import Image

import sys
import pathlib

# 将 torchocr路径加到python路径里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from torchocr.networks import build_model
from torchocr.datasets.det_modules import ResizeShortSize,ResizeFixedSize
from torchocr.postprocess import build_post_process

import cv2
from matplotlib import pyplot as plt
from torchocr.utils import draw_ocr_box_txt, draw_bbox
import argparse


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.resize = ResizeFixedSize(736, False)
        self.post_proess = build_post_process(cfg['post_process'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])
        ])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        box_list, score_list = self.post_proess(out, data['shape'], is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    parser.add_argument('--model_path', type=str, help='rec model path',default='/home/elimen/Data/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth')
    parser.add_argument('--img_path', type=str, help='img path for predict',default='/home/elimen/Data/dbnet_pytorch/test_images/mt03.png')
    args = parser.parse_args()
    img = cv2.imread(args.img_path)
    model = DetInfer(args.model_path)
    box_list, score_list = model.predict(img, is_output_polygon=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_bbox(img, box_list)
    cv2.imwrite('/home/elimen/Data/dbnet_pytorch/test_images/mt03_result.jpg',img)

    '''
    to do:
    1)写一个crop boxes函数
    2)
    '''
    model = RecInfer(args.model_path)
    out = model.predict(img)
