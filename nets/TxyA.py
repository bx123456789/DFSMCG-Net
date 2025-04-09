import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialAttentionModule(nn.Module):
    def __init__(self,channel, reduction=16):
        super(SpatialAttentionModule,self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        return x_cat_conv_split_h,x_cat_conv_split_w
class CA_Block(nn.Module):
    def __init__(self, channel, reduction):
        super(CA_Block, self).__init__()
        self.sa_input_1=SpatialAttentionModule(channel, reduction)
        self.conv = nn.Conv2d(channel * 4, channel, kernel_size=1, bias=False)

        self.F_h = nn.Conv2d(in_channels=channel // (reduction), out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // (reduction), out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.conv_out = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.sigmoid_w=nn.Sigmoid()
        self.sigmoid_h=nn.Sigmoid()
    def forward(self, x1,x2):
        x1_cat_conv_split_h, x1_cat_conv_split_w = self.sa_input_1(x1)
        x2_cat_conv_split_h, x2_cat_conv_split_w = self.sa_input_1(x2)
        x1_stack_w = torch.cat([x1_cat_conv_split_w, x2_cat_conv_split_w], dim=3)
        x2_stack_h = torch.cat([x1_cat_conv_split_h, x2_cat_conv_split_h], dim=3)
        x1_stack_w=x1_stack_w.permute(0, 1, 3, 2)
        s_h = self.F_h(x1_stack_w)
        s_w = self.F_w(x2_stack_h)
        s_h=self.sigmoid_h(s_h)
        s_w=self.sigmoid_h(s_w)
        batch_size, channels, height, width = s_h.shape
        s_h_ = s_h.view(batch_size, channels, 2, height // 2, width)
        batch_size, channels, height, width = s_w.shape
        s_w_ = s_w.view(batch_size, channels, height, 2, width // 2)
        # softmax_w = F.softmax(s_w_, dim=3)
        combined_w0_x1 = s_w_[:, :, :, 0, :]  # 左半部分
        combined_w1_x2 = s_w_[:, :, :, 1, :]  # 右半部分
        # softmax_h = F.softmax(s_h_, dim=2)
        combined_h0_x1 = s_h_[:, :, 0, :, :]  # 上半部分
        combined_h1_x2 = s_h_[:, :, 1, :, :]  # 下半部分

        # 将w方向和h方向的增强特征进行组合
        enhanced_x1 = combined_w0_x1 *combined_h0_x1*x1
        enhanced_x2 = combined_w1_x2 *combined_h1_x2*x2


        # 拼接通道和空间
        out=torch.cat((enhanced_x1,enhanced_x2),dim=1)
        # 同各国
        out=self.conv_out(out)
        return out

if __name__ == "__main__":
    x1=torch.randn(2,256,512,512)
    x2=torch.randn(2,256,512,512)
    model=CA_Block(channel=256,reduction=16)
    x1=model(x1,x2)
    print(x1.shape)
