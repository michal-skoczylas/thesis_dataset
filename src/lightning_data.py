from pathlib import Path
from typing import Optional, List, Tuple
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .dataset import find_dataset_dir, get_device


class GestureDataModule(pl.LightningDataModule):
    """
    LightningDataModule dla zbioru gestów dłoni z widoku drona.
    
    Automatycznie:
    - Lokalizuje zbiór danych w `data/cnn_dataset`
    - Aplikuje augmentacje specyficzne dla pracy drona (rotacja, przesunięcie, jitter światła)
    - Pomiata RandomHorizontalFlip (kluczowe, by odróżniać gesty 'left' i 'right')
    - Konfiguruje DataLoadery dla train, val i test
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        img_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: Optional[bool] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        if pin_memory is None:
            dev = get_device()
            self.pin_memory = dev.type == "cuda"
        else:
            self.pin_memory = pin_memory

        self.train_dataset: Optional[datasets.ImageFolder] = None
        self.val_dataset: Optional[datasets.ImageFolder] = None
        self.test_dataset: Optional[datasets.ImageFolder] = None
        self.class_names: List[str] = []
        self.num_classes: int = 0

        # Transformacje
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: Optional[str] = None):
        """Wczytanie zbiorów danych w zależności od fazy (fit, validate, test)."""
        data_path = find_dataset_dir(self.data_dir)
        train_path = data_path / "train"
        val_path = data_path / "val"
        test_path = data_path / "test"

        if stage in (None, "fit", "validate"):
            if train_path.exists():
                self.train_dataset = datasets.ImageFolder(str(train_path), transform=self.train_transform)
                self.class_names = self.train_dataset.classes
                self.num_classes = len(self.class_names)
            if val_path.exists():
                self.val_dataset = datasets.ImageFolder(str(val_path), transform=self.val_transform)
                if not self.class_names:
                    self.class_names = self.val_dataset.classes
                    self.num_classes = len(self.class_names)

        if stage in (None, "test"):
            # Jeśli istnieje dedykowany folder test/, użyj go; w przeciwnym razie użyj val/
            target_test = test_path if test_path.exists() else val_path
            if target_test.exists():
                self.test_dataset = datasets.ImageFolder(str(target_test), transform=self.val_transform)
                if not self.class_names:
                    self.class_names = self.test_dataset.classes
                    self.num_classes = len(self.class_names)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("validate")
        use_persistent = self.num_workers > 0
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_persistent,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
