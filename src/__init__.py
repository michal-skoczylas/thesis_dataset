"""
Pakiet src - moduły potoku klasyfikacji gestów dłoni dla pracy inżynierskiej.
"""

from .dataset import get_device, find_dataset_dir, get_dataloaders, denormalize
from .trainer import train_model, sanity_check_overfit
from .metrics import evaluate_model, show_misclassified_samples
from .export import export_model
from .lightning_data import GestureDataModule
from .lightning_module import GestureLightningModule, build_model, MODEL_REGISTRY
from .lightning_trainer import create_lightning_trainer, train_lightning

__all__ = [
    # Dane i urządzenia
    "get_device",
    "find_dataset_dir",
    "get_dataloaders",
    "denormalize",
    # Trener klasyczny PyTorch
    "train_model",
    "sanity_check_overfit",
    # Metryki i ewaluacja
    "evaluate_model",
    "show_misclassified_samples",
    # Eksport
    "export_model",
    # PyTorch Lightning
    "GestureDataModule",
    "GestureLightningModule",
    "build_model",
    "MODEL_REGISTRY",
    "create_lightning_trainer",
    "train_lightning",
]
