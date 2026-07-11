import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

class Generator(nn.Module):
    def __init__(self, n_channels=3, n_residual_blocks=16):
        super(Generator, self).__init__()
        
        self.conv1 = nn.Conv2d(n_channels, 64, 9, padding=4)
        self.prelu = nn.PReLU()
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(n_residual_blocks)])
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        
        self.conv3 = nn.Conv2d(64, n_channels, 9, padding=4)
        
    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1
        out = self.upsample(out)
        out = self.conv3(out)
        return torch.tanh(out)