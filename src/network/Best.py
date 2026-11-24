import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWaveletTransform(nn.Module):
    """Extract four frequency subbands (LL, LH, HL, HH)."""
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        ll = x1 + x2 + x3 + x4
        lh = -x1 - x2 + x3 + x4
        hl = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4
        return torch.cat([ll, lh, hl, hh], dim=1)

class SAFE_Mixer_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Spatial branch: large-kernel depthwise convolution.
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # Frequency branch: wavelet-based gating.
        self.hwdm = HaarWaveletTransform(dim)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        # Spatial gating branch.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )

        # Channel fusion.
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x):
        identity = x

        s_feat = self.spatial_dw(x)

        f_gate = self.freq_gate(self.hwdm(x))
        f_gate_up = F.interpolate(f_gate, size=s_feat.shape[2:], mode='bilinear', align_corners=False)

        s_gate = self.spatial_gate(x)

        mutual_gate = f_gate_up * s_gate

        refined_feat = s_feat * (1 + mutual_gate)

        return self.pw(refined_feat) + identity

class Stacked_SAFE_Mixer(nn.Module):
    """Stack SAFE-Mixer blocks with configurable depth."""
    def __init__(self, dim=1024, depth=7):
        super().__init__()
        self.layers = nn.Sequential(
            *[SAFE_Mixer_Block(dim) for _ in range(depth)]
        )

    def forward(self, x):
        return self.layers(x)

class Encoder_SRA_Block(nn.Module):
    def __init__(self, channels, sr_ratio, num_heads=4):
        super(Encoder_SRA_Block, self).__init__()
        self.sr_ratio = sr_ratio
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        if sr_ratio > 1:
            self.sr = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.BatchNorm2d(channels)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        nn.init.constant_(self.proj[1].weight, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_conv(x).flatten(2).transpose(1, 2).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x_sr = self.sr_norm(self.sr(x)) if self.sr_ratio > 1 else x
        k = self.k_conv(x_sr).flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_conv(x_sr).flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.proj(attn) + x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.up(x)

class UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder ---
        self.Conv1 = conv_block(img_ch, 64)
        self.Conv2 = conv_block(64, 128)
        self.sra2 = Encoder_SRA_Block(128, sr_ratio=8)
        self.Conv3 = conv_block(128, 256)
        self.sra3 = Encoder_SRA_Block(256, sr_ratio=4)
        self.Conv4 = conv_block(256, 512)
        self.sra4 = Encoder_SRA_Block(512, sr_ratio=2)

        # --- Bottleneck: 7 SAFE-Mixer blocks ---
        self.Conv5 = conv_block(512, 1024)
        self.safe_mixer_bottleneck = Stacked_SAFE_Mixer(dim=1024, depth=7)

        # --- Decoder D4: 7 SAFE-Mixer blocks ---
        self.Up5 = up_conv(1024, 512)
        self.safe_mixer_d4 = Stacked_SAFE_Mixer(dim=1024, depth=7)
        self.Up_conv5 = conv_block(1024, 512)

        # --- Decoder D3: convolution only ---
        self.Up4 = up_conv(512, 256)
        self.Up_conv4 = conv_block(512, 256)

        # --- Decoder D2: 2 SAFE-Mixer blocks ---
        self.Up3 = up_conv(256, 128)
        self.safe_mixer_d2 = Stacked_SAFE_Mixer(dim=256, depth=2)
        self.Up_conv3 = conv_block(256, 128)

        # --- Decoder D1: 1 SAFE-Mixer block ---
        self.Up2 = up_conv(128, 64)
        self.safe_mixer_d1 = Stacked_SAFE_Mixer(dim=128, depth=1)
        self.Up_conv2 = conv_block(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2_pre = self.Conv2(self.Maxpool(x1))
        x2 = self.sra2(x2_pre)

        x3 = self.sra3(self.Conv3(self.Maxpool(x2)))
        x4 = self.sra4(self.Conv4(self.Maxpool(x3)))

        x5 = self.Conv5(self.Maxpool(x4))
        x5_refined = self.safe_mixer_bottleneck(x5)

        up_5 = self.Up5(x5_refined)
        merge5 = torch.cat((x4, up_5), dim=1)
        refined_5 = self.safe_mixer_d4(merge5)
        d5 = self.Up_conv5(refined_5)

        up_4 = self.Up4(d5)
        merge4 = torch.cat((x3, up_4), dim=1)
        d4 = self.Up_conv4(merge4)

        up_3 = self.Up3(d4)
        merge3 = torch.cat((x2_pre, up_3), dim=1)
        refined_3 = self.safe_mixer_d2(merge3)
        d3 = self.Up_conv3(refined_3)

        up_2 = self.Up2(d3)
        merge2 = torch.cat((x1, up_2), dim=1)
        refined_2 = self.safe_mixer_d1(merge2)
        d2 = self.Up_conv2(refined_2)

        return self.Conv_1x1(d2)
