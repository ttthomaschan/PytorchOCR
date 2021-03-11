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

from torchocr.datasets.RecDataSet import RecDataProcess
from torchocr.utils import CTCLabelConverter

import cv2
from matplotlib import pyplot as plt
from torchocr.utils import draw_ocr_box_txt, draw_bbox
import argparse
import time


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


class RecInfer:
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

        self.process = RecDataProcess(cfg['dataset']['train']['dataset'])
        self.converter = CTCLabelConverter(cfg['dataset']['alphabet'])

    def predict(self, img):
        # 预处理根据训练来
        img = self.process.resize_with_specific_height(img)
        # img = self.process.width_pad_img(img, 120)
        img = self.process.normalize_img(img)
        tensor = torch.from_numpy(img.transpose([2, 0, 1])).float()
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        out = self.model(tensor)
        txt = self.converter.decode(out.softmax(dim=2).detach().cpu().numpy())
        return txt

class TabRecognition:
    def __init__(self):

    def lineDetection(self):
    
    def cellRecognition(self):

def generateExcelFile():
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PytorchOCR infer')
    # parser.add_argument('--modeldet_path', type=str, help='rec model path',default='/home/junlin/Git/github/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth')
    # parser.add_argument('--modelrec_path', type=str, help='rec model path',default='/home/junlin/Git/github/dbnet_pytorch/checkpoints/ch_rec_server_crnn_res34.pth')
    # parser.add_argument('--img_path', type=str, help='img path for predict',default='/home/junlin/Git/github/dbnet_pytorch/test_images/mt04.png')
    parser.add_argument('--modeldet_path', type=str, help='rec model path',default='/home/elimen/Data/dbnet_pytorch/checkpoints/ch_det_server_db_res18.pth')
    parser.add_argument('--modelrec_path', type=str, help='rec model path',default='/home/elimen/Data/dbnet_pytorch/checkpoints/ch_rec_server_crnn_res34.pth')
    parser.add_argument('--img_path', type=str, help='img path for predict',default='/home/elimen/Data/dbnet_pytorch/test_images/mt03.png')
    args = parser.parse_args()
    
    start = time.time()
    img = cv2.imread(args.img_path)
    img_bak = img.copy()
    modeldet = DetInfer(args.modeldet_path)
    box_list, score_list = modeldet.predict(img, is_output_polygon=False)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_bbox(img, box_list)

    #imageres_path = '/home/junlin/Git/github/dbnet_pytorch/test_results/'
    imageres_path = '/home/elimen/Data/dbnet_pytorch/test_results/'
    imageres_name = 'mt03_result.jpg'
    cv2.imwrite(imageres_path+imageres_name,img)

    txt_file = os.path.join(imageres_path, imageres_name.split('.')[0]+'.txt')
    txt_f = open(txt_file, 'w')
    

    '''
    output the text recognition result
    '''
    box_file = os.path.join(imageres_path, imageres_name.split('.')[0]+'_bbox.txt')
    box_f = open(box_file, 'w')
    imgcroplist = []
    bbox_cornerlist = []
    for i, box in enumerate(box_list):
        pt0=box[0]
        pt1=box[1]
        pt2=box[2]
        pt3=box[3]
        imgout = img_bak[int(min(pt0[1],pt1[1]))-4 :int(max(pt2[1],pt3[1])) +4,int(min(pt0[0],pt3[0]))-4:int(max(pt1[0],pt2[0]))+4]
        box_corner = [int(pt0[0]),int(pt0[1]),int(pt2[0]),int(pt2[1])]
        imgcroplist.append(imgout)
        bbox_cornerlist.append(box_corner)
        ######
        box_f.write(str(pt0))
        box_f.write(str(pt2))
        box_f.write('\n')
        cv2.imwrite(imageres_path+imageres_name.split('.')[0]+'_'+str(i)+'.jpg',imgout)
    
    modelrec = RecInfer(args.modelrec_path)

    for i in range(len(imgcroplist)-1,-1,-1):
        out = modelrec.predict(imgcroplist[i])

        txt_f.write(str(bbox_cornerlist[i]))
        txt_f.write(out[0][0]+ '\n')
        
    txt_f.close()


    print("Mission complete, it took {:.3f}s".format(time.time() - start))
    print(len(imgcroplist))
    print(out[0][0])
    print('Well done!')

