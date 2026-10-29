from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from .models import (
    SimpleBaselineCNN,
    GestureCNN_v1,
    GestureCNN_v2,
    GestureCNN_Residual,
    MobileNetV3Baseline,
    ResNet18Baseline,
    ResNet34Baseline,
)

# Rejestr modeli do szybkiego wywoływania po nazwie
MODEL_REGISTRY = {
    "simple_baseline": SimpleBaselineCNN,
    "baseline": SimpleBaselineCNN,
    "simplebaselinecnn": SimpleBaselineCNN,
    "v1": GestureCNN_v1,
    "gesturecnn_v1": GestureCNN_v1,
    "v2": GestureCNN_v2,
    "gesturecnn_v2": GestureCNN_v2,
    "residual": GestureCNN_Residual,
    "res": GestureCNN_Residual,
    "gesturecnn_residual": GestureCNN_Residual,
    "mobilenet": MobileNetV3Baseline,
    "mobilenet_v3": MobileNetV3Baseline,
    "mobilenetv3baseline": MobileNetV3Baseline,
    "resnet18": ResNet18Baseline,
    "resnet18baseline": ResNet18Baseline,
    "resnet34": ResNet34Baseline,
    "resnet34baseline": ResNet34Baseline,
}


def build_model(
    model: Optional[Union[nn.Module, str]] = None,
    num_classes: int = 8,
    **kwargs: Any,
) -> nn.Module:
    """Tworzy instancję modelu PyTorch na podstawie nazwy lub zwraca przekazany obiekt."""
    if isinstance(model, nn.Module):
        return model

    if model is None:
        model_name = "gesturecnn_v1"
    else:
        model_name = str(model).lower().strip()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Nieznany model: '{model}'. Dostępne opcje: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(num_classes=num_classes, **kwargs)


class GestureLightningModule(pl.LightningModule):
    """
    Uniwersalny moduł PyTorch Lightning dla klasyfikacji gestów dłoni.

    Zalety:
    - Obsługuje dowolną architekturę z `src/models/` (wystarczy podać nazwę np. 'v1' lub instancję nn.Module)
    - Wbudowane metryki torchmetrics (Accuracy) dla train, val i test
    - Automatyczne logowanie do TensorBoarda (loss, accuracy, learning rate)
    - Konfigurowalny optymalizator (AdamW, Adam, SGD) i harmonogram (CosineAnnealing, Plateau)
    - Bezpośrednia kompatybilność z modułami `src.export` oraz `src.metrics`
    """

    def __init__(
        self,
        model: Optional[Union[nn.Module, str]] = "GestureCNN_v1",
        num_classes: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        t_max: Optional[int] = None,
        eta_min: Optional[float] = None,
        class_names: Optional[List[str]] = None,
        **model_kwargs: Any,
    ):
        super().__init__()
        # Zapisujemy hiperparametry (z wyłączeniem instancji obiektu nn.Module)
        self.save_hyperparameters(ignore=["model"])

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer_name.lower()
        self.scheduler_name = scheduler_name.lower()
        self.t_max = t_max
        self.eta_min = eta_min if eta_min is not None else (lr * 0.1)
        self.class_names = class_names or []

        # Inicjalizacja sieci bazowej
        self.model = build_model(model, num_classes=num_classes, **model_kwargs)
        self.model_name = self.model.__class__.__name__

        # Funkcja straty z wygładzaniem etykiet (label smoothing)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Metryki TorchMetrics (zoptymalizowane pod Lightning)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        self.train_acc(logits, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        self.val_acc(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)

        self.test_acc(logits, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        inputs, targets = batch if isinstance(batch, (tuple, list)) else (batch, None)
        logits = self(inputs)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        res = {"preds": preds, "probs": probs}
        if targets is not None:
            res["targets"] = targets
        return res

    def configure_optimizers(self) -> Any:
        # 1. Dobór optymalizatora
        if self.optimizer_name == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Nieobsługiwany optymalizator: {self.optimizer_name}")

        # 2. Dobór harmonogramu uczenia (scheduler)
        if self.scheduler_name == "cosine":
            t_max = self.t_max or (self.trainer.max_epochs if self.trainer else 40)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif self.scheduler_name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_acc",
                    "interval": "epoch",
                },
            }
        elif self.scheduler_name == "none":
            return optimizer
        else:
            raise ValueError(f"Nieobsługiwany harmonogram: {self.scheduler_name}")

    def get_underlying_model(self) -> nn.Module:
        """Zwraca wewnętrzny obiekt nn.Module (np. do eksportu ONNX lub TorchScript)."""
        return self.model

    def save_torch_weights(self, path: Union[str, Path]):
        """Zapisuje czyste wagi state_dict wewnętrznego modelu PyTorch."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(out))
        print(f"💾 Zapisano wagi PyTorch: {out}")
