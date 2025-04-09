import torch
import torch.nn as nn
from torchvision.models import vgg16

from VGG16 import VGG16_layer1, VGG16_layer2, VGG16_layer3, VGG16_layer4, VGG16_layer5
from Decoder import decoder_layer5, decoder_layer4, decoder_layer3, decoder_layer2, decoder_layer1


class MBSSCA(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(MBSSCA, self).__init__()
        self.channel_attention = ChannelAttention(in_channels*3, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x_t1, x_td, x_t2):
        # Concatenate features from T1, TD, and T2 branches
        combined = torch.cat((x_t1, x_td, x_t2), dim=1)

        # Apply channel attention
        combined = self.channel_attention(combined)

        # Apply spatial attention
        combined = self.spatial_attention(combined)

        return combined


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max-pooling and Average-pooling along the channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)

        # Apply convolution and sigmoid
        spatial_attention = self.sigmoid(self.conv1(pool))
        return x * spatial_attention


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP layers
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.conv= nn.Conv2d(in_channels, in_channels//3, 1, bias=False)
    def forward(self, x):
        # Average Pooling
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max Pooling
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        # Add and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        out=self.conv(x * channel_attention)
        return out
class ChannelAttention_1(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP layers
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # Average Pooling
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max Pooling
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        # Add and apply sigmoid
        channel_attention = self.sigmoid(avg_out + max_out)
        return x * channel_attention
class CustomVGG16(torch.nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        # Load VGG16 model and use the 'features' part for convolutional layers
        self.backbone = vgg16(pretrained=False).features

    def forward(self, x_t1):
        # Pass input through successive layers and save outputs
        t1_stream_layer1 = self.backbone[:4](x_t1)    # Up to Conv2 layer
        t1_stream_layer2 = self.backbone[:9](x_t1)    # Up to Conv3 layer
        t1_stream_layer3 = self.backbone[:16](x_t1)   # Up to Conv4 layer
        t1_stream_layer4 = self.backbone[:23](x_t1)   # Up to Conv5 layer
        t1_stream_layer5 = self.backbone[:30](x_t1)   # Up to Conv6 layer

        return t1_stream_layer1, t1_stream_layer2, t1_stream_layer3, t1_stream_layer4, t1_stream_layer5

class TripleNet(nn.Module):
    def __init__(self):
        super(TripleNet, self).__init__()

        self.backbone = CustomVGG16()
        self.main_stream_layer1 = VGG16_layer1()  # 3-64
        self.main_stream_layer2 = VGG16_layer2()  # 64-128
        self.main_stream_layer3 = VGG16_layer3()  # 128-256
        self.main_stream_layer4 = VGG16_layer4()  # 256-512
        self.main_stream_layer5 = VGG16_layer5()  # 512-512

        self.mbssca1 = MBSSCA(64)
        self.mbssca2 = MBSSCA(128)
        self.mbssca3 = MBSSCA(256)
        self.mbssca4 = MBSSCA(512)
        self.mbssca5 = MBSSCA(512)

        self.decoder_layer5 = decoder_layer5(512, 512)
        self.sa5 = SpatialAttention()
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_trans5 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

        self.ca4 = ChannelAttention_1(in_channels=512 * 2, ratio=8)
        self.decoder_layer4 = decoder_layer4(512, 256)
        self.sa4 = SpatialAttention()
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_trans4 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

        self.ca3 = ChannelAttention_1(in_channels=256 * 2, ratio=8)
        self.decoder_layer3 = decoder_layer3(256, 128)
        self.sa3 = SpatialAttention()
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_trans3 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

        self.ca2 = ChannelAttention_1(in_channels=128 * 2, ratio=8)
        self.decoder_layer2 = decoder_layer2(128, 64)
        self.sa2 = SpatialAttention()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

        self.ca1 = ChannelAttention_1(in_channels=64 * 2, ratio=8)
        self.decoder_layer1 = decoder_layer1(64, 3)
        self.sa1 = SpatialAttention()
        self.bn1 = nn.BatchNorm2d(3)

        self.output = nn.Conv2d(3, 2, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t1, x_t2):
        t1_stream_layer1, t1_stream_layer2, t1_stream_layer3, t1_stream_layer4, t1_stream_layer5 = self.backbone(x_t1)
        t2_stream_layer1, t2_stream_layer2, t2_stream_layer3, t2_stream_layer4, t2_stream_layer5 = self.backbone(x_t2)

        main_stream_input = torch.abs(x_t1 - x_t2)
        main_stream_layer1 = self.main_stream_layer1(main_stream_input)
        main_stream_layer1 = self.mbssca1(main_stream_layer1, t1_stream_layer1, t2_stream_layer1)
        main_stream_layer2 = self.main_stream_layer2(main_stream_layer1)
        main_stream_layer2 = self.mbssca2(main_stream_layer2, t1_stream_layer2, t2_stream_layer2)
        main_stream_layer3 = self.main_stream_layer3(main_stream_layer2)
        main_stream_layer3 = self.mbssca3(main_stream_layer3, t1_stream_layer3, t2_stream_layer3)
        main_stream_layer4 = self.main_stream_layer4(main_stream_layer3)
        main_stream_layer4 = self.mbssca4(main_stream_layer4, t1_stream_layer4, t2_stream_layer4)
        main_stream_layer5 = self.main_stream_layer5(main_stream_layer4)
        main_stream_layer5 = self.mbssca5(main_stream_layer5, t1_stream_layer5, t2_stream_layer5)

        decoder_layer5 = self.decoder_layer5(main_stream_layer5)
        decoder_layer5 = self.bn5(self.sa5(decoder_layer5) * decoder_layer5)
        decoder_layer5 = self.conv_trans5(decoder_layer5)

        decoder_cat4 = torch.cat([decoder_layer5, main_stream_layer4], dim=1)
        decoder_layer4 = self.ca4(decoder_cat4) * decoder_cat4
        decoder_layer4 = self.decoder_layer4(decoder_layer4)
        decoder_layer4 = self.bn4(self.sa4(decoder_layer4) * decoder_layer4)
        decoder_layer4 = self.conv_trans4(decoder_layer4)

        decoder_cat3 = torch.cat([decoder_layer4, main_stream_layer3], dim=1)
        decoder_layer3 = self.ca3(decoder_cat3) * decoder_cat3
        decoder_layer3 = self.decoder_layer3(decoder_layer3)
        decoder_layer3 = self.bn3(self.sa3(decoder_layer3) * decoder_layer3)
        decoder_layer3 = self.conv_trans3(decoder_layer3)

        decoder_cat2 = torch.cat([decoder_layer3, main_stream_layer2], dim=1)
        decoder_layer2 = self.ca2(decoder_cat2) * decoder_cat2
        decoder_layer2 = self.decoder_layer2(decoder_layer2)
        decoder_layer2 = self.bn2(self.sa2(decoder_layer2) * decoder_layer2)
        decoder_layer2 = self.conv_trans2(decoder_layer2)

        decoder_cat1 = torch.cat([decoder_layer2, main_stream_layer1], dim=1)
        decoder_layer1 = self.ca1(decoder_cat1) * decoder_cat1
        decoder_layer1 = self.decoder_layer1(decoder_layer1)
        decoder_layer1 = self.bn1(self.sa1(decoder_layer1) * decoder_layer1)

        output = self.output(decoder_layer1)

        return output


if __name__ == "__main__":
    # 实例化UNet模型
    model = TripleNet()
    # 前向传播
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # # num_params = count_parameters(model)
    input_tensor1 = torch.randn(1, 3, 256, 256,device=device)
    input_tensor2 = torch.randn(1, 3, 256, 256,device=device)

    out=model(input_tensor1,input_tensor2)
    print(out)