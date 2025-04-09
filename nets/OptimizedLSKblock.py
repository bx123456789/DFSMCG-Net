import torch.nn as nn
import torch
class OptimizedLSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_small = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_medium = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_large = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

        self.drop_path = nn.Dropout(0.5)

    def forward(self, x):
        attn_small = self.conv_small(x)
        attn_medium = self.conv_medium(x)
        attn_large = self.conv_large(x)

        attn1 = self.conv1(attn_small + attn_medium)
        attn2 = self.conv2(attn_large)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        attn=self.drop_path(attn)
        return attn