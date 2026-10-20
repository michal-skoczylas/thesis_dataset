import torch
import torch.nn as nn
from .common import ConvBlock


class GestureCNN_v1(nn.Module):
    """
    Autorski, lekki klasyfikator gestów (Wersja bazowa v1).
    - 4 bloki konwolucyjne z BatchNorm
    - Global Average Pooling (chroni przed overfittingiem przy małej ilości danych)
    - Kompaktowa głowica z Dropoutem
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_filters),            # 128x128 -> 64x64
            ConvBlock(base_filters, base_filters * 2),       # 64x64   -> 32x32
            ConvBlock(base_filters * 2, base_filters * 4),   # 32x32   -> 16x16
            #ConvBlock(base_filters * 4, base_filters * 8),   # 16x16   -> 8x8
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(p=0.35),
            nn.Linear(base_filters * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class GestureCNN_v2(nn.Module):
    """
    Autorski, lekki klasyfikator gestów (Wersja bazowa v1).
    - 4 bloki konwolucyjne z BatchNorm
    - Global Average Pooling (chroni przed overfittingiem przy małej ilości danych)
    - Kompaktowa głowica z Dropoutem
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_filters),            # 128x128 -> 64x64
            ConvBlock(base_filters, base_filters * 2),       # 64x64   -> 32x32
            ConvBlock(base_filters * 2, base_filters * 4),   # 32x32   -> 16x16
            ConvBlock(base_filters * 4, base_filters * 8),   # 16x16   -> 8x8
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(base_filters * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x





class ResidualBlock(nn.Module):
    """Blok rezydualny z połączeniem skrótowym (Skip Connection)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class GestureCNN_Residual(nn.Module):
    """
    Wersja v2: Customowa sieć z połączeniami rezydualnymi (ResNet-like).
    Ułatwia przepływ gradientu i pozwala na lepszą separację drobnych detali gestów.
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(base_filters),
            nn.MaxPool2d(2, 2),  # 128 -> 64
        )
        self.downsample1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=1)

        self.layer2 = nn.Sequential(
            ResidualBlock(base_filters * 2),
            nn.MaxPool2d(2, 2),  # 64 -> 32
        )
        self.downsample2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=1)

        self.layer3 = nn.Sequential(
            ResidualBlock(base_filters * 4),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return self.classifier(x)
