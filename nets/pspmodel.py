import torch
import torch.nn as nn
import torch.nn.functional as F
#
# # 定义 Conv2dReLU 类
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)

# 定义 PSPBlock 类
class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

# 定义 PSPModule 类
class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2,3), use_bathcnorm=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, out_channels, size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])
        self.conv = Conv2dReLU(in_channels + len(sizes) * out_channels, out_channels, kernel_size=1,
                               use_batchnorm=use_bathcnorm)

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        x = self.conv(x)
        return x

# 测试 PSPModule
def test_psp_module():
    # 输入张量，形状为 (batch_size, in_channels, height, width)
    x = torch.randn(2, 768, 16, 16)

    # 创建 PSPModule 实例
    psp = PSPModule(in_channels=768, out_channels=768)

    # 前向传播
    output = psp(x)

    # 打印输出形状
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
#
# # # 运行测试
# # test_psp_module()
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#         # -----------------------------------------#
#         #   一共五个分支
#         # -----------------------------------------#
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)
#         conv3x3_2 = self.branch3(x)
#         conv3x3_3 = self.branch4(x)
#         # -----------------------------------------#
#         #   第五个分支，全局平均池化+卷积
#         # -----------------------------------------#
#         global_feature = torch.mean(x, 2, True)
#         global_feature = torch.mean(global_feature, 3, True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         # -----------------------------------------#
#         #   将五个分支的内容堆叠起来
#         #   然后1x1卷积整合特征。
#         # -----------------------------------------#
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result
if __name__ == "__main__":


    # 实例化UNet模型
    model =ASPP(dim_in=512, dim_out=512, rate=16 // 16)

    # 前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # input_tensor1 = torch.randn(4, 3, 256, 256,device=device)
    input_tensor2 = torch.randn(4, 512, 16, 16,device=device)
    # 执行前向传播
    outputs = model(input_tensor2)
    print(outputs.shape)
    # 获取GPU上分配的内存
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Allocated memory on GPU: {allocated:.2f} MB")

    # 可选:清除分配的临时内存
    torch.cuda.empty_cache()
