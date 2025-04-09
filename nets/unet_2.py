import torch
import torch.nn as nn

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import timm
from nets.pspmodel import PSPModule
from nets.TxyA import CA_Block
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.skip_conv = nn.Sequential(
#             nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_size),
#             nn.ReLU(inplace=True)
#         )
#         self.drop_path = DropPath(0.2)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv=nn.Conv2d(in_size,out_size,kernel_size=1)
#     def forward(self, inputs1, inputs2):
#         skip = self.skip_conv(inputs1)
#         x = self.up(inputs2)
#         x = self.drop_path(x) + skip
#         return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        )
        self.drop_path = nn.Identity()

        self.drop_path = DropPath(0.8)
    def forward(self, inputs1, inputs2):
        skip=self.skip_conv(inputs1)
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        skip=self.drop_path(skip)
        outputs=skip+outputs
        return outputs


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=not use_batchnorm
        )
        relu = nn.GELU()
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class MergeBlock(nn.Module):
    def __init__(self, policy,target_channels):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))

        self.policy = policy
        self.target_channels = target_channels
        self.conv_layers = nn.ModuleList()
    def forward(self, x):
        if self.policy == 'add':
            processed_tensors = []
            for tensor in x:
                device = tensor.device
                if tensor.shape[1] != self.target_channels:
                    conv = nn.Conv2d(tensor.shape[1], self.target_channels, kernel_size=1).to(device)
                    tensor = conv(tensor)
                processed_tensors.append(tensor)
            return sum(processed_tensors)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, align_corners=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='resnet50'):
        super(Unet, self).__init__()
        if backbone == 'convx':
            self.resnet = timm.create_model('convnext_tiny', features_only=True, pretrained=False, in_chans=3)
            in_filters = [96, 192, 384, 768]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use convx.'.format(backbone))

        self.tCda_4 = CA_Block(in_filters[3], reduction=16)
        self.tCda_3 = CA_Block(in_filters[2], reduction=16)
        self.tCda_2 = CA_Block(in_filters[1], reduction=16)
        self.tCda_1 = CA_Block(in_filters[0], reduction=16)
        self.psp = PSPModule(in_channels=in_filters[3], out_channels=in_filters[3])
        self.psp_last_conv = Conv2dReLU(in_channels=3840, out_channels=768, kernel_size=1, use_batchnorm=True)
        self.segmentation_head = SegmentationHead(in_channels=96, out_channels=2, kernel_size=1, upsampling=4, align_corners=False)

        self.up_concat3 = unetUp(in_filters[3] + in_filters[2], 384)
        self.up_concat2 = unetUp(in_filters[2] + in_filters[1], 192)
        self.up_concat1 = unetUp(in_filters[1] + in_filters[0], 96)
        self.merge = MergeBlock("add",target_channels=96)

    def forward(self, t1_inputs, t2_inputs):
        [t1_feat1, t1_feat2, t1_feat3, t1_feat4] = self.resnet.forward(t1_inputs)
        [t2_feat1, t2_feat2, t2_feat3, t2_feat4] = self.resnet.forward(t2_inputs)

        # 融合
        feat4 = self.tCda_4(t1_feat4, t2_feat4)
        feat3 = self.tCda_3(t1_feat3, t2_feat3)
        feat2 = self.tCda_2(t1_feat2, t2_feat2)
        feat1 = self.tCda_1(t1_feat1, t2_feat1)

        # feat4 = self.psp(feat4)
        # feat4 = self.psp_last_conv(feat4)
        # decoder


        up3 = self.up_concat3(feat3, feat4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)




        output_size = up1.size()[2:]
        feature_pyramid = [nn.functional.interpolate(p, output_size, mode='bilinear', align_corners=False) for p in [feat4, up3, up2, up1]]

        seg_out = self.merge(feature_pyramid)
        seg_out_f = self.segmentation_head(seg_out)

        return seg_out_f
if __name__ == "__main__":


    # 实例化UNet模型
    model = Unet(num_classes=2,pretrained=False, backbone='convx')

    # 前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor1 = torch.randn(2, 3, 512, 512,device=device)
    input_tensor2 = torch.randn(2, 3, 512, 512,device=device)
    # 执行前向传播
    outputs = model(input_tensor1,input_tensor2)
    print(outputs.shape)
    # 获取GPU上分配的内存
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Allocated memory on GPU: {allocated:.2f} MB")

    # 可选:清除分配的临时内存
    torch.cuda.empty_cache()