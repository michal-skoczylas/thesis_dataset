from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device() -> torch.device:
    """Wykrywa optymalne urządzenie obliczeniowe: Apple Silicon (MPS), CUDA lub CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """Automatycznie lokalizuje katalog cnn_dataset niezależnie od katalogu roboczego."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "data" / "cnn_dataset",
        Path.cwd().parent / "data" / "cnn_dataset",
        Path(__file__).resolve().parent.parent / "data" / "cnn_dataset",
    ]
    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c
    raise FileNotFoundError(
        f"Nie znaleziono katalogu 'cnn_dataset'. Szukano w: {[str(c) for c in candidates]}"
    )


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Odwraca normalizację ImageNet dla celów wyświetlania w matplotlib."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = tensor.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    return np.clip(inp, 0, 1)


def get_dataloaders(
    dataset_dir: Optional[str] = None,
    img_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Przygotowuje DataLoadery dla zbioru treningowego i walidacyjnego.

    UWAGA: Celowo nie stosujemy transforms.RandomHorizontalFlip(),
    ponieważ zbiór zawiera gesty 'left' oraz 'right'!
    """
    data_path = find_dataset_dir(dataset_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"

    # Transformacje pod specyfikę widoku z drona (lot, rotacja, światło)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(train_path), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_path), transform=val_transform)

    device = get_device()
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names
