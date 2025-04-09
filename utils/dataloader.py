import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import albumentations as A

class CD_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path,image_format,label_format):
        super(CD_Dataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.vocdevkit_path       = dataset_path
        self.image_format = image_format
        self.label_format = label_format
        self.train_transform = A.Compose([
            A.HueSaturationValue(p=0.5),  # 模拟季节性变化和光照差异
            A.RandomBrightnessContrast(p=0.5),  # 增强光照和对比度差异的适应性
            A.RGBShift(p=0.5),  # 模拟传感器差异
            A.RandomGamma(p=0.5),  # 处理不同曝光情况
            A.RandomResizedCrop(height=self.input_shape[0], width=self.input_shape[1], scale=(0.5, 1.0), p=0.5),
            # 提升对小面积变化的敏感性
            A.Flip(p=0.5),  # 水平翻转，增强对对称变化的处理
            A.Rotate(30, p=0.5),  # 随机旋转，增加视角变化的多样性
            A.GaussianBlur(p=0.3),  # 模拟模糊图像
            A.Normalize(),
        ], additional_targets={'image_2': 'image'})

        self.val_transform = A.Compose([
            A.Normalize(),
        ], additional_targets={'image_2': 'image'})
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg_A = Image.open(os.path.join(self.vocdevkit_path, f"VOC2007/JPEGImages/A/{name}.{self.image_format}"))
        jpg_B = Image.open(os.path.join(self.vocdevkit_path, f"VOC2007/JPEGImages/B/{name}.{self.image_format}"))
        png = Image.open(os.path.join(self.vocdevkit_path, f"VOC2007/SegmentationClass/{name}.{self.label_format}"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        transformed_data = self.get_random_data(image_A=jpg_A, image_B=jpg_B,label=png, random=self.train)
        jpg_A, jpg_B, png = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
        jpg_A         = np.transpose(jpg_A, [2,0,1])
        jpg_B         = np.transpose(jpg_B, [2,0,1])
        #-------------------------------------------------------#
        #   去除标签中的异常值
        #-------------------------------------------------------#
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes+1)[png]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes))

        return jpg_A,jpg_B, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_A,image_B, label, random=False):
        image_A = np.array(image_A)
        image_B   = np.array(image_B)
        label   = np.array(label)
        if random:
             return self.train_transform(image=image_A, image_2=image_B, mask=label)
        else:
            return self.val_transform(image=image_A, image_2=image_B, mask=label)


# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images_A      = []
    images_B      = []
    pngs        = []
    seg_labels  = []
    for img_a,img_b, png, labels in batch:
        images_A.append(img_a)
        images_B.append(img_b)
        pngs.append(png)
        seg_labels.append(labels)
    images_A      = torch.from_numpy(np.array(images_A)).type(torch.FloatTensor)
    images_B      = torch.from_numpy(np.array(images_B)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images_A,images_B, pngs, seg_labels
