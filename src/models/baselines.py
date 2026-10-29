import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3Baseline(nn.Module):
    """
    Model bazowy (Transfer Learning) do porównania w pracy dyplomowej:
    MobileNetV3-Small z pretrenowanymi wagami ImageNet i podmienioną głowicą.
    """
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet18Baseline(nn.Module):
    """ResNet-18 z pretrenowanymi wagami ImageNet jako baseline / model do transfer learningu."""
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ResNet34Baseline(nn.Module):
    """ResNet-34 z pretrenowanymi wagami ImageNet do głębszego transfer learningu."""
    def __init__(self, num_classes: int = 8, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        in_features = self.backbone.fc.in_features
        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

