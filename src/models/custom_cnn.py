import torch
import torch.nn as nn


class SimpleBaselineCNN(nn.Module):
    """
    ====================================================================
    KROK 1: Absolutnie najprostsza sieć bazowa (Czysty Baseline)
    ====================================================================
    Tylko podstawowe bloki: Conv2d -> ReLU -> MaxPool2d.
    Brak BatchNorm, brak Dropoutu.
    Filtry: 3 -> 32 -> 64 -> 128
    """
    def __init__(self, num_classes: int = 8, **kwargs):
        super().__init__()

        # 1. EKSTRAKCJA CECH
        self.features = nn.Sequential(
            # Blok 1: 3 kanały wejściowe -> 32 filtry (128x128 -> 64x64)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2: 32 -> 64 filtry (64x64 -> 32x32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 3: 64 -> 128 filtrów (32x32 -> 16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 2. GŁOWICA KLASYFIKACYJNA
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class GestureCNN_v1(nn.Module):
    """
    ====================================================================
    KROK 2: Dodanie Normalizacji Wsadowej (BatchNorm)
    ====================================================================
    Konstrukcja splotowa z Batch Normalization po każdym splocie.
    Filtry: 3 -> 32 -> 64 -> 128 -> 256
    """
    def __init__(self, num_classes: int = 8, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 3 -> 32 filtry (128x128 -> 64x64)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2: 32 -> 64 filtry (64x64 -> 32x32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 3: 64 -> 128 filtrów (32x32 -> 16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 4: 128 -> 256 filtrów (16x16 -> 8x8)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class GestureCNN_v2(nn.Module):
    """
    ====================================================================
    KROK 3: Regularyzacja (Dropout) zapobiegająca przeuczeniu
    ====================================================================
    Architektura z warstwami Dropout w celu ograniczenia luki generalizacyjnej.
    Filtry: 3 -> 32 -> 64 -> 128 -> 256
    """
    def __init__(self, num_classes: int = 8, dropout_rate: float = 0.4, **kwargs):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 3 -> 32 filtry (128x128 -> 64x64)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2: 32 -> 64 filtry (64x64 -> 32x32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 3: 64 -> 128 filtrów (32x32 -> 16x16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 4: 128 -> 256 filtrów (16x16 -> 8x8)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Blok rezydualny ze stałą liczbą kanałów i połączeniem skrótowym (Skip Connection)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class GestureCNN_Residual(nn.Module):
    """
    ====================================================================
    KROK 4: Połączenia rezydualne (Skip Connections)
    ====================================================================
    Ułatwiają przepływ gradientu i chronią przed degradacją głębokiej sieci.
    Filtry: 3 -> 32 -> 64 -> 128
    """
    def __init__(self, num_classes: int = 8, **kwargs):
        super().__init__()

        # Warstwa wejściowa: 3 -> 32 filtry
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Etap 1: 32 kanały (128x128 -> 64x64)
        self.stage1 = nn.Sequential(
            ResidualBlock(channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Projekcja przejścia: 32 -> 64
        self.trans1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        # Etap 2: 64 kanały (64x64 -> 32x32)
        self.stage2 = nn.Sequential(
            ResidualBlock(channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Projekcja przejścia: 64 -> 128
        self.trans2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

        # Etap 3: 128 kanałów (32x32 -> 16x16)
        self.stage3 = nn.Sequential(
            ResidualBlock(channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Głowica
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        x = self.trans2(x)
        x = self.stage3(x)
        x = self.classifier(x)
        return x
