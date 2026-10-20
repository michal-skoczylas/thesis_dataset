import torch
import torch.nn as nn


def print_model_summary(model: nn.Module, input_size=(1, 3, 128, 128)):
    """Wyświetla podsumowanie liczby parametrów i szacowany rozmiar wag."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)
    print("=" * 60)
    print(f"🧠 Model: {model.__class__.__name__}")
    print(f"📊 Wszystkie parametry       : {total_params:,}")
    print(f"🔥 Parametry trenowalne      : {trainable_params:,}")
    print(f"💾 Rozmiar wag w pamięci     : {size_mb:.2f} MB")
    print("=" * 60)


class ConvBlock(nn.Module):
    """
    Podstawowy podwójny blok konwolucyjny:
    [Conv 3x3 -> BN -> ReLU] x 2 -> MaxPool 2x2 -> Spatial Dropout
    """
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
