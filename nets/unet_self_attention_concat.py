# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# import timm
# from nets.TxyA import CA_Block
# # from nets.pspmodel import PSPModule
# # from TxyA import CA_Block
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=0.0):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#         random_tensor.floor_()
#         return x.div(keep_prob) * random_tensor
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.up = nn.ConvTranspose2d(in_size-out_size, in_size-out_size, kernel_size=2, stride=2)
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         outputs = self.up(inputs2)
#         outputs = torch.cat([inputs1, outputs], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs
#
#
#
# class MergeBlock(nn.Module):
#     def __init__(self, policy,target_channels):
#         super().__init__()
#         if policy not in ["add", "cat"]:
#             raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
#
#         self.policy = policy
#         self.target_channels = target_channels
#         self.conv_layers = nn.ModuleList()
#     def forward(self, x):
#         if self.policy == 'add':
#             processed_tensors = []
#             for tensor in x:
#                 device = tensor.device
#                 if tensor.shape[1] != self.target_channels:
#                     conv = nn.Conv2d(tensor.shape[1], self.target_channels, kernel_size=1).to(device)
#                     tensor = conv(tensor)
#                 processed_tensors.append(tensor)
#             return sum(processed_tensors)
#         elif self.policy == 'cat':
#             return torch.cat(x, dim=1)
#         else:
#             raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))
#
# class SegmentationHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1, align_corners=True):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=align_corners) if upsampling > 1 else nn.Identity()
#         super().__init__(conv2d, upsampling)
# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# class Change_Guide_Mul_Window_attention(nn.Module):
#     def __init__(self, dim, map_dim,num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         # 确保dim可以被num_heads整除
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         # 每个head的维度
#         head_dim = int(dim / num_heads)
#         self.dim = dim
#
#         # 设置head数量
#         self.h_heads = num_heads
#         self.h_dim = self.h_heads * head_dim
#
#         self.window_sizes = [2, 4, 8, 16,32]
#
#         # 缩放因子
#         self.scale = head_dim ** -0.5
#
#         # 使用1x1卷积代替线性层，生成qkv
#         # 使用1x1卷积代替线性层，生成qkv
#         self.h_qkv = nn.Conv2d(self.dim, self.h_dim * 2, kernel_size=1, bias=qkv_bias)
#         self.v = nn.Conv2d(self.h_dim, self.h_dim, kernel_size=1, bias=qkv_bias)
#
#         # 使用1x1卷积代替线性层，生成投影输出
#         self.h_proj = nn.Conv2d(self.h_dim, self.h_dim, kernel_size=1)
#
#
#         self.conv_temp = nn.Conv2d(map_dim, self.h_dim, kernel_size=1)
#     def mul_window(self, x, temp):
#         B, C, H, W = x.shape  # 获取输入张量的形状
#
#         # 计算q, k, v，并调整形状
#         qkv = self.h_qkv(x).reshape(B, 2, self.h_heads, self.h_dim // self.h_heads, H, W).permute(1, 0, 2, 4, 5, 3)
#         k, v = qkv[0], qkv[1]  # 分别获取q, k, v的值，形状为 (B, n_head, H, W, head_dim)
#         q = self.v(temp).reshape(B, self.h_heads, self.h_dim // self.h_heads, H, W).permute(0, 1, 3, 4, 2)
#         attn_outputs = []
#         remaining_heads = self.h_heads
#         for i, ws in enumerate(self.window_sizes):  # 对于每个窗口大小
#             if H < ws: break
#             h_group, w_group = H // ws, W // ws  # 计算分组数
#             total_groups = h_group * w_group
#
#             # 调整q, k, v的形状以适应分组
#             q_i = q[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
#             k_i = k[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
#             v_i = v[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
#
#             # 调整形状以进行矩阵乘法
#             q_i = q_i.reshape(B, total_groups, ws * ws, -1)
#             k_i = k_i.reshape(B, total_groups, ws * ws, -1)
#             v_i = v_i.reshape(B, total_groups, ws * ws, -1)
#
#             # 计算注意力得分并进行归一化
#             attn = (q_i @ k_i.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = (attn @ v_i).reshape(B, h_group, w_group, ws, ws, -1).transpose(2, 3)
#
#             # 将输出重新调整形状并添加到输出列表
#             attn_outputs.append(attn.reshape(B, H, W, -1))
#             remaining_heads -= 1
#         if remaining_heads > 0:
#             q_remain = q[:, -remaining_heads:].reshape(B, remaining_heads, H, W, -1)
#             k_remain = k[:, -remaining_heads:].reshape(B, remaining_heads, H, W, -1)
#             v_remain = v[:, -remaining_heads:].reshape(B, remaining_heads, H, W, -1)
#             attn_remain = (q_remain @ k_remain.transpose(-2, -1)) * self.scale
#             attn_remain = attn_remain.softmax(dim=-1)
#             attn_remain = (attn_remain @ v_remain).reshape(B, remaining_heads, H, W, -1)
#             attn_outputs.append(attn_remain.reshape(B, H, W, -1))
#
#         # 将所有head的输出拼接
#         x = torch.cat(attn_outputs, dim=-1).permute(0, 3, 1, 2)
#         # 投影到原始维度
#         x = self.h_proj(x)
#         return x
#
#     def forward(self, x, temp):
#         output_size = x.size()[2:]
#         temp = nn.functional.interpolate(temp, output_size, mode='bilinear', align_corners=False)
#         temp=self.conv_temp(temp)
#         # 调用hifi方法计算输出
#         mul_window__out = self.mul_window(x, temp)+x
#
#         return mul_window__out  # 返回最终输出
# class cSE(nn.Module):
#     def __init__(self, out_channels,rate=16):
#         super(cSE, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=out_channels, out_channels=int(out_channels / 16), kernel_size=1, padding=0),
#         )
#         self.relu=nn.ReLU(inplace=True)
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=int(out_channels / 16), out_channels=out_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.sigmoid=nn.Sigmoid()
#     def forward(self, x):
#         x = nn.AvgPool2d(x.size()[2:])(x)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.sigmoid(x)
#         return x
# class diff(nn.Module):
#     def __init__(self, in_channels):
#         super(diff, self).__init__()
#         self.catten=cSE(in_channels)
#     def forward(self, x1, x2):
#         diff_abs= torch.abs(x1 - x2)
#         diff=self.catten(diff_abs)*diff_abs+diff_abs
#         return diff
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, pretrained=False, backbone='resnet50'):
#         super(Unet, self).__init__()
#         if backbone == 'convx':
#             self.resnet = timm.create_model('convnext_tiny', features_only=True, pretrained=True, in_chans=3)
#             in_filters = [96, 192, 384, 768]
#         elif backbone == 'resnet':
#             self.resnet = timm.create_model('resnet18', features_only=True, pretrained=True, in_chans=3)
#             in_filters = [64, 128, 256, 512]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use convx.'.format(backbone))
#         self.tCda_5 = CA_Block(in_filters[3], reduction=16)
#         self.tCda_4 = CA_Block(in_filters[2], reduction=16)
#         self.tCda_3 = CA_Block(in_filters[1], reduction=16)
#         self.tCda_2 = CA_Block(in_filters[0], reduction=16)
#         self.tCda_1 = CA_Block(in_filters[0], reduction=16)
#
#         self.segmentation_head = SegmentationHead(in_channels=512, out_channels=2, kernel_size=1, upsampling=2, align_corners=False)
#         #
#         self.up_concat3 = unetUp(in_filters[2]+in_filters[3] , 256)
#         self.up_concat2 = unetUp(in_filters[1]+in_filters[2] , 128)
#         self.up_concat1 = unetUp(in_filters[0]+in_filters[1] , 64)
#         self.up_concat0 = unetUp(in_filters[0]+in_filters[0] , 64)
#         self.merge = MergeBlock("cat",target_channels=96)
#         # 解码曾
#         self.Upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
#
#         self.diff5=diff(512)
#         self.diff4=diff(256)
#         self.diff3=diff(128)
#         self.diff2=diff(64)
#         self.diff1=diff(64)
#         # 指导模块
#         self.guide4=Change_Guide_Mul_Window_attention(dim=256,map_dim=512)
#         self.guide3=Change_Guide_Mul_Window_attention(dim=128,map_dim=256)
#         self.guide2=Change_Guide_Mul_Window_attention(dim=64,map_dim=128)
#         self.guide1=Change_Guide_Mul_Window_attention(dim=64,map_dim=64)
#
#         #
#         self.deocde = nn.Sequential(BasicConv2d(512, 256, 3, 1, 1),
#                                     BasicConv2d(256, 128, 3, 1, 1),
#                                     BasicConv2d(128, 64, 3, 1, 1),
#                                     nn.Conv2d(64, 1, 3, 1, 1))
#
#     def forward(self, t1_inputs, t2_inputs):
#         [t1_feat1, t1_feat2, t1_feat3, t1_feat4,t1_feat5] = self.resnet.forward(t1_inputs)
#         [t2_feat1, t2_feat2, t2_feat3, t2_feat4,t2_feat5] = self.resnet.forward(t2_inputs)
#         #
#         feat5 = self.tCda_5(t1_feat5, t2_feat5)
#         feat4 = self.tCda_4(t1_feat4, t2_feat4)
#         feat3 = self.tCda_3(t1_feat3, t2_feat3)
#         feat2 = self.tCda_2(t1_feat2, t2_feat2)
#         feat1 = self.tCda_1(t1_feat1, t2_feat1)
#         # 差异特征
#         feat5_map=self.diff5(t1_feat5,t2_feat5)
#         feat4_map=self.diff4(t1_feat4,t2_feat4)
#         feat3_map=self.diff3(t1_feat3,t2_feat3)
#         feat2_map=self.diff2(t1_feat2,t2_feat2)
#         # decoder
#         up4 = self.up_concat3(feat4,feat5)
#         up4=self.guide4(up4,feat5_map)
#
#         up3 = self.up_concat2(feat3,up4)
#         up3 = self.guide3(up3, feat4_map)
#
#         up2 = self.up_concat1(feat2,up3)
#         up2 = self.guide2(up2, feat3_map)
#
#         up1 = self.up_concat0(feat1,up2)
#         up1 = self.guide1(up1, feat2_map)
#
#         output_size = up1.size()[2:]
#         feature_pyramid = [nn.functional.interpolate(p, output_size, mode='bilinear', align_corners=False) for p in [up4,up3, up2, up1]]
#
#         seg_out = self.merge(feature_pyramid)
#         # seg_out_f = self.deocde(seg_out)
#
#         seg_out_f = self.segmentation_head(seg_out)
#
#         return seg_out_f
# if __name__ == "__main__":
#
#
#     # 实例化UNet模型
#     model = Unet(num_classes=2,pretrained=False, backbone='resnet')
#
#     # 前向传播
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     input_tensor1 = torch.randn(16, 3, 256, 256,device=device)
#     input_tensor2 = torch.randn(16, 3, 256, 256,device=device)
#     # 执行前向传播
#     outputs = model(input_tensor1,input_tensor2)
#     print(outputs.shape)
#     # 获取GPU上分配的内存
#     allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
#     print(f"Allocated memory on GPU: {allocated:.2f} MB")
#
#     # 可选:清除分配的临时内存
#     torch.cuda.empty_cache()
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm
from nets.TxyA import CA_Block
from nets.pspmodel import PSPModule


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


class Mul_Window_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 确保dim可以被num_heads整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 每个head的维度
        head_dim = int(dim / num_heads)
        self.dim = dim

        # 设置head数量
        self.h_heads = num_heads
        self.h_dim = self.h_heads * head_dim

        self.all_window_sizes = [2, 4, 8, 16]

        # 缩放因子
        self.scale = qk_scale or head_dim ** -0.5

        # 使用1x1卷积代替线性层，生成qkv
        self.h_qkv = nn.Conv2d(self.dim, self.h_dim * 3, kernel_size=1, bias=qkv_bias)
        # 使用1x1卷积代替线性层，生成投影输出
        self.h_proj = nn.Conv2d(self.h_dim, self.h_dim, kernel_size=1)

    def mul_window(self, x):
        B, C, H, W = x.shape  # 获取输入张量的形状
        # 动态选择可用的窗口大小
        self.window_sizes = [ws for ws in self.all_window_sizes if ws <= min(H, W)]
        # 计算q, k, v，并调整形状
        qkv = self.h_qkv(x).reshape(B, 3, self.h_heads, self.h_dim // self.h_heads, H, W).permute(1, 0, 2, 4, 5, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取q, k, v的值，形状为 (B, n_head, H, W, head_dim)

        x = []
        for i, ws in enumerate(self.window_sizes):  # 对于每个窗口大小
            h_group, w_group = H // ws, W // ws  # 计算分组数
            total_groups = h_group * w_group

            # 调整q, k, v的形状以适应分组
            q_i = q[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
            k_i = k[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
            v_i = v[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)

            # 调整形状以进行矩阵乘法
            q_i = q_i.reshape(B, total_groups, ws * ws, -1)
            k_i = k_i.reshape(B, total_groups, ws * ws, -1)
            v_i = v_i.reshape(B, total_groups, ws * ws, -1)

            # 计算注意力得分并进行归一化
            attn = (q_i @ k_i.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = (attn @ v_i).reshape(B, h_group, w_group, ws, ws, -1).transpose(2, 3)

            # 将输出重新调整形状并添加到输出列表
            x.append(attn.reshape(B, H, W, -1))

        # 将所有head的输出拼接
        x = torch.cat(x, dim=-1).permute(0, 3, 1, 2)
        # 投影到原始维度
        x = self.h_proj(x)
        return x

    def forward(self, x):
        # 调用hifi方法计算输出
        mul_window__out = self.mul_window(x) + x

        return mul_window__out  # 返回最终输出


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.GELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.GELU()
        )
        self.drop_path = nn.Identity()

        self.drop_path = DropPath(0.2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = self.drop_path(outputs)
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
    def __init__(self, policy, target_channels):
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
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear',
                                 align_corners=align_corners) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Change_Guide_Mul_Window_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        # 确保dim可以被num_heads整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 每个head的维度
        head_dim = int(dim / num_heads)
        self.dim = dim

        # 设置head数量
        self.h_heads = num_heads
        self.h_dim = self.h_heads * head_dim

        self.window_sizes = [2, 4, 8, 16, 32]

        # 缩放因子
        self.scale = head_dim ** -0.5

        # 使用1x1卷积代替线性层，生成qkv
        self.h_qkv = nn.Conv2d(self.dim, self.h_dim * 2, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(self.h_dim, self.h_dim, kernel_size=1, bias=qkv_bias)

        # 使用1x1卷积代替线性层，生成投影输出
        self.h_proj = nn.Conv2d(self.h_dim, self.h_dim, kernel_size=1)

        self.conv_temp = nn.Conv2d(512, self.h_dim, kernel_size=1)

    def mul_window(self, x, temp):
        B, C, H, W = x.shape  # 获取输入张量的形状

        # 计算q, k, v，并调整形状
        qkv = self.h_qkv(x).reshape(B, 2, self.h_heads, self.h_dim // self.h_heads, H, W).permute(1, 0, 2, 4, 5, 3)
        k, v = qkv[0], qkv[1]  # 分别获取q, k, v的值，形状为 (B, n_head, H, W, head_dim)
        q = self.v(temp).reshape(B, self.h_heads, self.h_dim // self.h_heads, H, W).permute(0, 1, 3, 4, 2)
        attn_outputs = []
        remaining_heads = self.h_heads
        for i, ws in enumerate(self.window_sizes):  # 对于每个窗口大小
            if H < ws: break
            h_group, w_group = H // ws, W // ws  # 计算分组数
            total_groups = h_group * w_group

            # 调整q, k, v的形状以适应分组
            q_i = q[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
            k_i = k[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)
            v_i = v[:, i].reshape(B, h_group, ws, w_group, ws, -1).transpose(2, 3)

            # 调整形状以进行矩阵乘法
            q_i = q_i.reshape(B, total_groups, ws * ws, -1)
            k_i = k_i.reshape(B, total_groups, ws * ws, -1)
            v_i = v_i.reshape(B, total_groups, ws * ws, -1)

            # 计算注意力得分并进行归一化
            attn = (q_i @ k_i.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = (attn @ v_i).reshape(B, h_group, w_group, ws, ws, -1).transpose(2, 3)

            # 将输出重新调整形状并添加到输出列表
            attn_outputs.append(attn.reshape(B, H, W, -1))
            remaining_heads -= 1
        if remaining_heads > 0:
            q_remain = q[:, -remaining_heads:]
            k_remain = k[:, -remaining_heads:]
            v_remain = v[:, -remaining_heads:]
            attn_remain = (q_remain @ k_remain.transpose(-2, -1)) * self.scale
            attn_remain = attn_remain.softmax(dim=-1)
            attn_remain = (attn_remain @ v_remain).reshape(B, remaining_heads, H, W, -1)
            attn_outputs.append(attn_remain.reshape(B, H, W, -1))

        # 将所有head的输出拼接
        x = torch.cat(attn_outputs, dim=-1).permute(0, 3, 1, 2)
        # 投影到原始维度
        x = self.h_proj(x)
        return x

    def forward(self, x, temp):
        output_size = x.size()[2:]
        temp = nn.functional.interpolate(temp, output_size, mode='bilinear', align_corners=False)
        temp = self.conv_temp(temp)
        # 调用hifi方法计算输出
        mul_window__out = self.mul_window(x, temp) + x

        return mul_window__out  # 返回最终输出


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W).permute(1, 0, 2, 4, 5, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='resnet50'):
        super(Unet, self).__init__()
        if backbone == 'convx':
            self.resnet = timm.create_model('convnext_tiny', features_only=True, pretrained=True, in_chans=3)
            in_filters = [96, 192, 384, 768]
        elif backbone == 'resnet':
            self.resnet = timm.create_model('resnet18', features_only=True, pretrained=False, in_chans=3)
            in_filters = [64, 128, 256, 512]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use convx.'.format(backbone))
        self.tCda_5 = CA_Block(in_filters[3], reduction=16)
        self.tCda_4 = CA_Block(in_filters[2], reduction=16)
        self.tCda_3 = CA_Block(in_filters[1], reduction=16)
        self.tCda_2 = CA_Block(in_filters[0], reduction=16)
        self.tCda_1 = CA_Block(in_filters[0], reduction=16)

        self.segmentation_head = SegmentationHead(in_channels=512, out_channels=2, kernel_size=1, upsampling=2,
                                                  align_corners=False)
        #
        self.up_concat3 = unetUp(in_filters[2] + in_filters[3], 256)
        self.up_concat2 = unetUp(in_filters[1] + in_filters[2], 128)
        self.up_concat1 = unetUp(in_filters[0] + in_filters[1], 64)
        self.up_concat0 = unetUp(in_filters[0] + in_filters[0], 64)

        self.merge = MergeBlock("cat", target_channels=96)
        # 解码曾

        self.psp = PSPModule(in_channels=512, out_channels=512)

        # 使用多头自注意力替换guide模块
        self.guide4 = MultiHeadSelfAttention(dim=256, num_heads=8)
        self.guide3 = MultiHeadSelfAttention(dim=128, num_heads=8)
        self.guide2 = MultiHeadSelfAttention(dim=64, num_heads=8)
        self.guide1 = MultiHeadSelfAttention(dim=64, num_heads=8)
        # 同各国

        self.conv_5 =  nn.Sequential(
            nn.Conv2d(in_filters[3]*2, in_filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_filters[3]),
            nn.ReLU(inplace=True)
        )
        self.conv_4 =  nn.Sequential(
            nn.Conv2d(in_filters[2]*2, in_filters[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.conv_3 =  nn.Sequential(
            nn.Conv2d(in_filters[1]*2, in_filters[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_filters[1]),
            nn.ReLU(inplace=True)
        )
        self.conv_2 =  nn.Sequential(
            nn.Conv2d(in_filters[0]*2, in_filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_filters[0]),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_filters[0]*2, in_filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_filters[0]),
            nn.ReLU(inplace=True)
        )


    def forward(self, t1_inputs, t2_inputs):
        [t1_feat1, t1_feat2, t1_feat3, t1_feat4, t1_feat5] = self.resnet.forward(t1_inputs)
        [t2_feat1, t2_feat2, t2_feat3, t2_feat4, t2_feat5] = self.resnet.forward(t2_inputs)

        feat5 = torch.cat((t1_feat5,t2_feat5),dim=1)
        feat5=self.conv_5(feat5)

        feat4 = torch.cat((t1_feat4,t2_feat4),dim=1)
        feat4 = self.conv_4(feat4)

        feat3 = torch.cat((t1_feat3,t2_feat3),dim=1)
        feat3 = self.conv_3(feat3)

        feat2 = torch.cat((t1_feat2,t2_feat2),dim=1)
        feat2 = self.conv_2(feat2)

        feat1 = torch.cat((t1_feat1,t2_feat1),dim=1)
        feat1 = self.conv_1(feat1)

        feat5 = self.psp(feat5)

        # decoder
        up4 = self.up_concat3(feat4, feat5)
        up4 = self.guide4(up4)

        up3 = self.up_concat2(feat3, up4)
        up3 = self.guide3(up3)

        up2 = self.up_concat1(feat2, up3)
        up2 = self.guide2(up2)

        up1 = self.up_concat0(feat1, up2)
        up1 = self.guide1(up1)

        output_size = up1.size()[2:]
        feature_pyramid = [nn.functional.interpolate(p, output_size, mode='bilinear', align_corners=False) for p in
                           [up4, up3, up2, up1]]

        seg_out = self.merge(feature_pyramid)
        seg_out_f = self.segmentation_head(seg_out)

        return seg_out_f

from thop import profile
from thop import clever_format
import time
if __name__ == "__main__":
    # # 实例化UNet模型
    # model = Unet(num_classes=2, pretrained=False, backbone='resnet')
    # 实例化UNet模型
    model = Unet(num_classes=2, pretrained=True, backbone='resnet')
    # 前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # # num_params = count_parameters(model)
    # # num_params = count_parameters(model)
    input_tensor1 = torch.randn(1, 3, 256, 256,device=device)
    input_tensor2 = torch.randn(1, 3, 256, 256,device=device)
    #
    from thop import profile
    from thop import clever_format
    macs, params = profile(model, inputs=(input_tensor1,input_tensor2))

    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Parameters: {params}")
    # torch.cuda.empty_cache()
    #
    # # 获取模型在开始时的内存占用
    # torch.cuda.reset_peak_memory_stats()
    # model(input_tensor1,input_tensor2)
    #
    # # 获取内存占用，单位为字节
    # allocated_memory = torch.cuda.memory_allocated()  # 已分配的内存
    #
    # # 转换为MB（兆字节）
    # allocated_memory_mb = allocated_memory / (1024 ** 2)
    # print(f"Allocated Memory (MB): {allocated_memory_mb:.2f} MB")
    # 清理CUDA缓存（确保没有旧缓存影响）
    # torch.cuda.empty_cache()
    #
    # # 进行一次预热推理（加载权重并初始化模型）
    # _ = model(input_tensor1, input_tensor2)
    #
    # # 清理缓存，确保下一次推理时间更准确
    # torch.cuda.empty_cache()
    #
    # # 记录前向传播开始时间
    # start_time = time.time()
    #
    # # 执行一次推理，计算输出结果
    # output = model(input_tensor1, input_tensor2)
    #
    # # 记录前向传播结束时间
    # end_time = time.time()
    #
    # # 计算推理时间
    # inference_time = end_time - start_time
    #
    # # 打印推理时间
    # print(f"Inference Time (s): {inference_time:.4f} s")