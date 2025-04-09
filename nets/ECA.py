import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return y

class AdaptiveChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AdaptiveChannelAttention, self).__init__()
        self.channels = channels

        # MaxPool and AvgPool layers
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Convolution layer
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # ECA block for adaptive weights
        self.eca = ECA_block(channels)

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        # Apply MaxPool and AvgPool
        maxpool1 = self.maxpool(x1)
        avgpool1 = self.avgpool(x1)
        maxpool2 = self.maxpool(x2)
        avgpool2 = self.avgpool(x2)

        # Concatenate MaxPool and AvgPool results
        cat1 = torch.cat((maxpool1, avgpool1), dim=1)
        cat2 = torch.cat((maxpool2, avgpool2), dim=1)

        # Apply Conv and ReLU
        conv_out1 = self.relu(self.conv(cat1))
        conv_out2 = self.relu(self.conv(cat2))

        # Apply ECA block to get adaptive weights
        weight1 = self.eca(conv_out1)
        weight2 = self.eca(conv_out2)

        # Apply Softmax to get final weights
        softmax_out = self.softmax(torch.stack((weight1, weight2), dim=1))

        # Separate weights for x1 and x2
        weight1, weight2 = softmax_out[:, 0, :, :, :], softmax_out[:, 1, :, :, :]

        # Apply weights to the inputs
        out1 = weight1 * x1
        out2 = weight2 * x2

        # Sum the weighted inputs
        out = out1 + out2

        return out

# 测试模块
# x1 = torch.randn(1, 64, 32, 32)  # 假设 x1 的尺寸为 (batch_size, channels, height, width)
# x2 = torch.randn(1, 64, 32, 32)  # 假设 x2 的尺寸相同
#
# channel_attention = AdaptiveChannelAttention(channels=64)
# output = channel_attention(x1, x2)
#
# print(output.shape)  # 形状应该是 (batch_size, channels, height, width)
