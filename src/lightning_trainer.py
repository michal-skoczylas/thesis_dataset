from pathlib import Path
from typing import Optional, Union, Tuple, Any, List
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from .lightning_data import GestureDataModule
from .lightning_module import GestureLightningModule


class SaveBestTorchWeightsCallback(pl.Callback):
    """
    Callback zapisujący czyste wagi PyTorch (.pt) przy każdym rekordzie val_acc.
    Zapewnia to natychmiastową kompatybilność z modułami `src.export` i notatnikami,
    bez konieczności ręcznego wyciągania wag z pliku .ckpt Lightninga.
    """

    def __init__(self, save_path: Union[str, Path], monitor: str = "val_acc", mode: str = "max"):
        super().__init__()
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor].item()
        is_better = (current > self.best_score) if self.mode == "max" else (current < self.best_score)

        if is_better:
            self.best_score = current
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(pl_module, "model"):
                torch.save(pl_module.model.state_dict(), str(self.save_path))
                trainer.print(f"⭐ [Lightning] Nowy rekord {self.monitor}={current:.4f}! Zapisano wagi PyTorch: {self.save_path}")


def create_lightning_trainer(
    experiment_name: str = "lightning_gesture_cnn",
    max_epochs: int = 40,
    save_dir: str = "weights",
    log_dir: str = "runs/lightning",
    early_stopping_patience: Optional[int] = None,
    accelerator: str = "auto",
    devices: Union[int, str] = 1,
    precision: Union[int, str] = "32-true",
    enable_progress_bar: bool = True,
    **trainer_kwargs: Any,
) -> pl.Trainer:
    """
    Tworzy skonfigurowaną instancję pl.Trainer z:
    - Logerem TensorBoard (runs/lightning/<experiment_name>)
    - ModelCheckpoint (zapisuje najlepszy pełny checkpoint .ckpt według val_acc)
    - SaveBestTorchWeightsCallback (zapisuje weights/<experiment_name>_best.pt dla kompatybilności)
    - LearningRateMonitor
    - Opcjonalnym EarlyStopping
    - Automatycznym wykrywaniem akceleratora (MPS / CUDA / CPU)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Logger TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False,
    )

    # 2. Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_path),
        filename=f"{experiment_name}_best-{{epoch:02d}}-{{val_acc:.3f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # 3. Zapis czystych wag .pt (dla src.export i notatnika)
    raw_weights_path = save_path / f"{experiment_name}_best.pt"
    raw_weights_callback = SaveBestTorchWeightsCallback(save_path=raw_weights_path, monitor="val_acc")

    # 4. Monitor LR
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks: List[pl.Callback] = [checkpoint_callback, raw_weights_callback, lr_monitor]

    # 5. Opcjonalne wczesne zatrzymanie (Early Stopping)
    if early_stopping_patience is not None and early_stopping_patience > 0:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stop)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=tb_logger,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=trainer_kwargs.pop("log_every_n_steps", 10),
        **trainer_kwargs,
    )
    return trainer


def train_lightning(
    model: Optional[Union[nn.Module, str]] = "GestureCNN_v1",
    data_dir: Optional[str] = None,
    datamodule: Optional[GestureDataModule] = None,
    num_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    img_size: int = 64,
    experiment_name: str = "lightning_gesture_cnn",
    early_stopping_patience: Optional[int] = None,
    save_dir: str = "weights",
    log_dir: str = "runs/lightning",
    num_workers: int = 2,
    accelerator: str = "auto",
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
    **trainer_kwargs: Any,
) -> Tuple[GestureLightningModule, pl.Trainer, GestureDataModule]:
    """
    Główna, ultra-prosta funkcja do uruchomienia treningu PyTorch Lightning w jednej linii.

    Przykład użycia:
        >>> from src.lightning_trainer import train_lightning
        >>> pl_model, trainer, dm = train_lightning(
        ...     model="GestureCNN_v1",
        ...     num_epochs=30,
        ...     experiment_name="exp_lightning_v1"
        ... )
    """
    # 1. Przygotowanie DataModule
    if datamodule is None:
        datamodule = GestureDataModule(
            data_dir=data_dir,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    datamodule.setup("fit")

    # 2. Przygotowanie modułu Lightning
    if isinstance(model, GestureLightningModule):
        pl_module = model
    else:
        pl_module = GestureLightningModule(
            model=model,
            num_classes=datamodule.num_classes,
            lr=lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            t_max=num_epochs,
            class_names=datamodule.class_names,
        )

    # 3. Utworzenie Trainera
    trainer = create_lightning_trainer(
        experiment_name=experiment_name,
        max_epochs=num_epochs,
        save_dir=save_dir,
        log_dir=log_dir,
        early_stopping_patience=early_stopping_patience,
        accelerator=accelerator,
        **trainer_kwargs,
    )

    trainer.print(f"🚀 [PyTorch Lightning] Start treningu: {experiment_name}")
    trainer.print(f"📊 Model: {pl_module.model_name} | Klasy ({datamodule.num_classes}): {datamodule.class_names}")

    # 4. Trening
    trainer.fit(pl_module, datamodule=datamodule)

    # 5. Załadowanie najlepszych wag do wewnętrznego modelu (jeśli istnieją)
    best_pt = Path(save_dir) / f"{experiment_name}_best.pt"
    if best_pt.exists():
        try:
            pl_module.model.load_state_dict(torch.load(best_pt, map_location="cpu"))
            trainer.print(f"🏆 Wczytano najlepsze wagi do pl_module.model z {best_pt}")
        except Exception:
            pass

    return pl_module, trainer, datamodule
