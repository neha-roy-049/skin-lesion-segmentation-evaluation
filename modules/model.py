import torch
import torch.nn as nn
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class ViTBottleneck(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.norm2(tokens))
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)

class SkinLesionSegNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        enc_channels = self.encoder.feature_info.channels()

        self.vit_bottleneck = ViTBottleneck(enc_channels[-1], num_heads=8)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.dec4 = ConvBlock(enc_channels[-1] + enc_channels[-2], 256)
        self.dec3 = ConvBlock(256 + enc_channels[-3], 128)
        self.dec2 = ConvBlock(128 + enc_channels[-4], 64)
        self.dec1 = ConvBlock(64, 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        features = self.encoder(x)               # Extract 4 levels of features
        e1, e2, e3, e4 = features

        b = self.vit_bottleneck(e4)              # Global attention on deepest features
        b = self.dropout(b)

        d4 = self.dec4(torch.cat([self.up(b), e3], dim=1))
        d4 = self.dropout(d4)
        d3 = self.dec3(torch.cat([self.up(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        d1 = self.dec1(self.up(d2))
        out = self.up(self.head(d1))
        return out