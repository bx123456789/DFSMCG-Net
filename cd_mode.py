import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


from utils.utils import cvtColor, preprocess_input, resize_image, show_config

import albumentations as A
#-----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
#-----------------------------------------------------------------------------------#
class CD_Model(object):
    #---------------------------------------------------#
    #   初始化Deeplab
    #---------------------------------------------------#
    def __init__(self, model_path=None, num_classes=None, input_shape=None, cuda=None):
        # 参数初始化
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.cuda = cuda
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        # 归一化转为tensor
        self.val_transform = A.Compose([
            A.Normalize(),
        ], additional_targets={'image_2': 'image'}, is_check_shapes=False)
                    
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #     加载的具体模型
        #-------------------------------#
        from nets.BIT import BASE_Transformer
        self.net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                                 with_pos='learned', enc_depth=1, dec_depth=8)
        #-------------------------------#
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
    def get_miou_png(self, image_a,image_b):
        image_a       = cvtColor(image_a)
        image_b       = cvtColor(image_b)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_A = np.array(image_a)
        image_B   = np.array(image_b)
        transformed_data = self.val_transform(image=image_A, image_2=image_B)
        image_data_A, image_data_B = transformed_data['image'], transformed_data['image_2']
        image_data_A = np.expand_dims(np.transpose(image_data_A, (2, 0, 1)), 0)
        image_data_B = np.expand_dims(np.transpose(image_data_B, (2, 0, 1)), 0)

        with torch.no_grad():
            images_a = torch.from_numpy(image_data_A)
            images_b = torch.from_numpy(image_data_B)
            if self.cuda:
                images_a = images_a.cuda()
                images_b = images_b.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images_a,images_b)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
