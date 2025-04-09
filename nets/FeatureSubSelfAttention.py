



import torch.nn as nn
import torch
class FeatureSubSelfAttention(nn.Module):
    """
    Feature Sub-Self-Attention Module with external Q input
    dim: number of channels of input
    heads: number of attention heads
    """

    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.kvl = nn.Conv2d(dim, dim * (self.heads - 1), kernel_size=1, padding=0)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim* 2 , dim, 1, stride=1, padding=0, bias=True)
        self.act = nn.GELU()
    def forward(self, x, q_input):
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
            q_input: External Q tensor of shape [B, C', H', W']
        Returns:
            Aggregated tensor of shape [B, C*2, H, W]
        """
        B, C, H, W = x.shape

        # Linear projection to obtain K, V, L
        kvl = self.act(self.kvl(x))  # Shape: [B, dim * heads, H, W]
        kvl = kvl.view(B, (self.heads - 1), C, H, W)  # Split into heads: [B, heads, dim, H, W]

        k = kvl[:, -2, :, :, :]  # Key: second-to-last head
        v = kvl[:, -1, :, :, :].flatten(2)  # Value: second head flattened
        lfeat = kvl[:, -0, :, :, :]  # Last feature for residual addition

        # Use external Q input as Query
        q = self.q(q_input)
        q = q.flatten(2)  # Flatten external Q: [B, C', H*W]

        # Flatten and compute attention
        k_flat = k.flatten(2)  # [B, dim, H*W]
        qk = torch.matmul(q, k_flat.transpose(1, 2))  # [B, H*W, H*W]
        qk = torch.softmax(qk, dim=-1)

        # Compute attention-weighted features
        x2 = torch.matmul(qk, v).reshape(B, C, H, W)  # [B, dim, H, W]

        # Combine features
        x = torch.cat([lfeat, x2], dim=1)  # Combine local feature and attention output
        x=self.proj(x)
        x=self.act(x)
        return x