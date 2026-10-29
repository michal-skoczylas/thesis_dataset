#!/usr/bin/env python3
"""
scripts/run_resnet_experiments.py

Zautomatyzowane eksperymenty trenowania i optymalizacji ResNet (ResNet-18, ResNet-34)
w PyTorch Lightning z akceleracją Apple Metal (MPS).

Główne zadanie:
- Eksploracja różnych wariantów pod kątem detekcji gestów dłoni, ze szczególnym naciskiem na 'closehand' i 'openhand'
- Porównanie rozdzielczości (64x64 vs 112x112), strategii transfer learningu (warmup vs full fine-tuning) oraz regularyzacji (Dropout)
- Generowanie kompletu wykresów (krzywe uczenia, macierze pomyłek, krzywe ROC/PR, siatki predykcji)
- Zestawienie wyników w postaci wykresu słupkowego i tabeli podsumowującej (CSV & Markdown)
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import pytorch_lightning as pl

# Upewnij się, że katalog główny jest w sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightning_data import GestureDataModule
from src.lightning_module import GestureLightningModule, build_model
from src.lightning_trainer import create_lightning_trainer
from src.dataset import denormalize, get_device

REPORTS_DIR = PROJECT_ROOT / "reports" / "resnet_experiments"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


class BackboneWarmupCallback(pl.Callback):
    """Callback zamrażający backbone na N epok i automatycznie go odmrażający."""
    def __init__(self, unfreeze_epoch: int = 0):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.unfreeze_epoch <= 0:
            return

        if trainer.current_epoch == 0:
            if hasattr(pl_module.model, "backbone"):
                for p in pl_module.model.backbone.parameters():
                    p.requires_grad = False
                if hasattr(pl_module.model.backbone, "fc"):
                    for p in pl_module.model.backbone.fc.parameters():
                        p.requires_grad = True
                trainer.print(f"🧊 [Warmup] Zamrożono backbone na {self.unfreeze_epoch} pierwszych epok.")
        elif trainer.current_epoch == self.unfreeze_epoch:
            if hasattr(pl_module.model, "backbone"):
                for p in pl_module.model.backbone.parameters():
                    p.requires_grad = True
                trainer.print(f"🔥 [Fine-tuning] Odmrożono backbone do pełnego dotrenowywania!")


def plot_metrics_history(history: Optional[Dict[str, list]], trainer: pl.Trainer, exp_name: str, out_path: Path):
    """Rysuje krzywe Train/Val Loss, Train/Val Accuracy oraz LR."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Przebieg Treningu: {exp_name} (Apple Metal MPS)", fontsize=14, fontweight="bold")
    ax1, ax2, ax3 = axes

    # 1. Próba odczytu z MetricsHistoryCallback
    if history and history.get("epoch"):
        valid_train = [(e + 1, l, a) for e, l, a in zip(history["epoch"], history.get("train_loss", []), history.get("train_acc", [])) if l is not None and a is not None]
        valid_val = [(e + 1, l, a) for e, l, a in zip(history["epoch"], history.get("val_loss", []), history.get("val_acc", [])) if l is not None and a is not None]
        valid_lr = [(e + 1, lr) for e, lr in zip(history["epoch"], history.get("lr", [])) if lr is not None]

        if valid_train:
            t_ep, t_loss, t_acc = zip(*valid_train)
            t_acc = [a * 100 for a in t_acc]
            ax1.plot(t_ep, t_loss, label="Train Loss", color="#1f77b4", lw=2.2)
            ax2.plot(t_ep, t_acc, label="Train Acc", color="#2ca02c", lw=2.2)

        if valid_val:
            v_ep, v_loss, v_acc = zip(*valid_val)
            v_acc = [a * 100 for a in v_acc]
            ax1.plot(v_ep, v_loss, label="Val Loss", color="#ff7f0e", lw=2.2, linestyle="--")
            ax2.plot(v_ep, v_acc, label="Val Acc", color="#d62728", lw=2.5)
            best_val = max(v_acc)
            best_idx = v_acc.index(best_val)
            best_ep = v_ep[best_idx]
            ax2.scatter([best_ep], [best_val], color="#d62728", s=100, zorder=5)
            ax2.annotate(f"Best: {best_val:.2f}%", (best_ep, best_val),
                         textcoords="offset points", xytext=(-25, 10), fontweight="bold")

        if valid_lr:
            lr_ep, lrs = zip(*valid_lr)
            ax3.plot(lr_ep, lrs, label="Learning Rate", color="#9467bd", lw=2.0)
    else:
        # Fallback z pliku CSV
        log_dir = Path(trainer.logger.log_dir) if hasattr(trainer.logger, "log_dir") else Path("runs/lightning") / exp_name
        csv_file = log_dir / "metrics.csv"
        if not csv_file.exists():
            return None
        df = pd.read_csv(csv_file)
        val_df = df.dropna(subset=["val_loss", "val_acc"])
        train_df = df.dropna(subset=["train_loss"])
        if not train_df.empty:
            t_loss = train_df.groupby("epoch")["train_loss"].mean()
            ax1.plot(t_loss.index + 1, t_loss.values, label="Train Loss", color="#1f77b4", lw=2.2)
            if "train_acc" in train_df.columns:
                t_acc = train_df.groupby("epoch")["train_acc"].mean() * 100
                ax2.plot(t_acc.index + 1, t_acc.values, label="Train Acc", color="#2ca02c", lw=2.2)
        if not val_df.empty:
            v_ep = val_df["epoch"].values + 1
            v_loss = val_df["val_loss"].values
            v_acc = val_df["val_acc"].values * 100
            ax1.plot(v_ep, v_loss, label="Val Loss", color="#ff7f0e", lw=2.2, linestyle="--")
            ax2.plot(v_ep, v_acc, label="Val Acc", color="#d62728", lw=2.5)

    ax1.set_title("Funkcja Straty (Cross-Entropy)", fontweight="bold")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_title("Dokładność Klasyfikacji (Accuracy)", fontweight="bold")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("%")
    ax2.set_ylim(-2, 103)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    ax3.set_title("Harmonogram LR (Cosine Annealing)", fontweight="bold")
    ax3.set_xlabel("Epoka")
    ax3.set_ylabel("LR")
    ax3.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def evaluate_and_plot_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    exp_name: str,
    out_path: Path,
    device: torch.device
) -> Dict[str, Any]:
    """Generuje macierz pomyłek i wylicza kompletne metryki ewaluacji."""
    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    fig_size = (7, 6) if len(class_names) <= 3 else (9, 7.5)
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Prawdopodobieństwo (znormalizowane)"},
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(f"Macierz Pomyłek: {exp_name}\nVal Acc: {acc*100:.2f}% | Macro F1: {f1_macro*100:.2f}%", fontsize=12, pad=12)
    plt.ylabel("Rzeczywisty gest (Ground Truth)", fontsize=11)
    plt.xlabel("Rozpoznany gest (Model ResNet)", fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "report_dict": report_dict,
        "cm": cm,
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_probs": all_probs,
    }


def plot_binary_roc_pr(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    class_names: List[str],
    exp_name: str,
    out_path: Path,
):
    """Krzywe ROC oraz Precision-Recall dla wariantów binarnych."""
    if len(class_names) != 2:
        return None

    pos_idx = 1
    pos_label = class_names[pos_idx]
    y_scores = all_probs[:, pos_idx]
    y_true = (all_targets == pos_idx).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Krzywe Jakości: {exp_name} ({class_names[0]} vs {class_names[1]})", fontsize=13, fontweight="bold")

    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("Krzywa ROC", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"PR (AP = {pr_auc:.4f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.05])
    ax2.set_xlabel("Recall (Czułość)")
    ax2.set_ylabel("Precision (Precyzja)")
    ax2.set_title(f"Krzywa Precision-Recall (Klasa: {pos_label})", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def visualize_sample_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    exp_name: str,
    out_path: Path,
    device: torch.device,
    num_samples: int = 16
):
    """Generuje siatkę losowych próbek walidacyjnych z rzeczywistą i przewidzianą etykietą."""
    model.eval()
    model = model.to(device)

    collected_images = []
    collected_targets = []
    collected_preds = []
    collected_confs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(len(targets)):
                collected_images.append(inputs[i].cpu())
                collected_targets.append(targets[i].item())
                collected_preds.append(preds[i].item())
                collected_confs.append(confs[i].item())

                if len(collected_images) >= num_samples * 2:
                    break
            if len(collected_images) >= num_samples * 2:
                break

    indices = list(range(len(collected_images)))
    np.random.seed(42)
    selected_idx = np.random.choice(indices, min(num_samples, len(indices)), replace=False)

    cols = 4
    rows = (len(selected_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
    axes = np.array(axes).reshape(-1)

    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        img_tensor = collected_images[idx]
        t_cls = class_names[collected_targets[idx]]
        p_cls = class_names[collected_preds[idx]]
        conf = collected_confs[idx]

        is_correct = (t_cls == p_cls)
        color = "#2ca02c" if is_correct else "#d62728"

        # Odwróć normalizację ImageNet
        img_np = denormalize(img_tensor)

        ax.imshow(img_np)
        title_text = f"Prawda: {t_cls}\nPred: {p_cls} ({conf*100:.1f}%)"
        ax.set_title(title_text, color=color, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    for j in range(len(selected_idx), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Próbki predykcji ResNet: {exp_name}\n(Zielona ramka = Zgodność, Czerwona = Błąd)",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def run_single_experiment(
    exp_id: str,
    model_name: str,
    allowed_classes: Optional[List[str]],
    img_size: int = 112,
    batch_size: int = 32,
    epochs: int = 25,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    patience: int = 10,
    freeze_backbone_epochs: int = 0,
) -> Dict[str, Any]:
    """Wykonuje kompletny trening i ewaluację pojedynczego wariantu."""
    print("\n" + "=" * 76)
    print(f"🚀 [EKSPERYMENT: {exp_id}] Model: {model_name.upper()} | Img: {img_size}x{img_size} | Klasy: {allowed_classes or 'Wszystkie 8'}")
    print(f"   LR: {lr} | Dropout: {dropout} | Freeze warmup: {freeze_backbone_epochs} epok")
    print("=" * 76)

    # 1. DataModule
    dm = GestureDataModule(
        img_size=img_size,
        batch_size=batch_size,
        num_workers=2,
        allowed_classes=allowed_classes,
    )
    dm.setup("fit")
    num_classes = dm.num_classes
    class_names = dm.class_names
    print(f"📦 Dane: {len(dm.train_dataset)} train, {len(dm.val_dataset)} val. Klasy: {class_names}")

    # 2. Urządzenie MPS
    device = get_device()
    print(f"🍎 Akcelerator sprzętowy: {device.type.upper()} (Apple Metal)")

    # 3. Model PyTorch z Dropoutem
    base_model = build_model(model_name, num_classes=num_classes, dropout=dropout)

    pl_module = GestureLightningModule(
        model=base_model,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=0.05,
        optimizer_name="adamw",
        scheduler_name="cosine",
        class_names=class_names,
    )

    # 4. Trener Lightning
    exp_name = f"exp_{exp_id}"
    trainer = create_lightning_trainer(
        experiment_name=exp_name,
        max_epochs=epochs,
        save_dir=str(WEIGHTS_DIR),
        log_dir="runs/lightning",
        early_stopping_patience=patience,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
    )

    if freeze_backbone_epochs > 0:
        trainer.callbacks.append(BackboneWarmupCallback(unfreeze_epoch=freeze_backbone_epochs))

    # 5. Fit
    t0 = time.time()
    trainer.fit(pl_module, datamodule=dm)
    train_duration = time.time() - t0
    print(f"⏱️  Trening trwał: {train_duration:.1f} s ({train_duration/60:.2f} min)")

    # 6. Odczyt historii
    history = None
    for cb in trainer.callbacks:
        if hasattr(cb, "history"):
            history = cb.history
            break

    # 7. Krzywe uczenia
    curves_path = REPORTS_DIR / f"curves_{exp_name}.png"
    plot_metrics_history(history, trainer, exp_name, curves_path)

    # 8. Ewaluacja z najlepszymi wagami
    best_weights_path = WEIGHTS_DIR / f"{exp_name}_best.pt"
    eval_model = build_model(model_name, num_classes=num_classes, dropout=dropout)
    if best_weights_path.exists():
        eval_model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"🏆 Wczytano najlepsze wagi do ewaluacji: {best_weights_path}")
    else:
        eval_model.load_state_dict(pl_module.model.state_dict())

    val_loader = dm.val_dataloader()
    cm_path = REPORTS_DIR / f"cm_{exp_name}.png"
    eval_results = evaluate_and_plot_confusion_matrix(eval_model, val_loader, class_names, exp_name, cm_path, device)

    # ROC / PR dla klasyfikacji binarnej
    roc_res = None
    if num_classes == 2:
        roc_path = REPORTS_DIR / f"roc_pr_{exp_name}.png"
        roc_res = plot_binary_roc_pr(eval_results["all_targets"], eval_results["all_probs"], class_names, exp_name, roc_path)

    # Podgląd próbek
    samples_path = REPORTS_DIR / f"samples_{exp_name}.png"
    visualize_sample_predictions(eval_model, val_loader, class_names, exp_name, samples_path, device)

    param_count = sum(p.numel() for p in eval_model.parameters())

    # Wyciągnij F1 dla kluczowych klas (closehand i openhand)
    close_f1 = eval_results["report_dict"].get("closehand", {}).get("f1-score", 0.0)
    open_f1 = eval_results["report_dict"].get("openhand", {}).get("f1-score", 0.0)

    res = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "model_name": model_name,
        "classes": class_names,
        "num_classes": num_classes,
        "img_size": img_size,
        "dropout": dropout,
        "lr": lr,
        "accuracy": eval_results["accuracy"],
        "f1_macro": eval_results["f1_macro"],
        "close_f1": close_f1,
        "open_f1": open_f1,
        "roc_auc": roc_res["roc_auc"] if roc_res else None,
        "duration_sec": train_duration,
        "param_count": param_count,
        "best_weights": best_weights_path,
        "curves_path": curves_path,
        "cm_path": cm_path,
        "samples_path": samples_path,
        "report_dict": eval_results["report_dict"],
    }

    print(f"\n📈 WYNIKI DLA {exp_name}:")
    print(f"   • Dokładność (Val Acc) : {res['accuracy']*100:.2f}%")
    print(f"   • Macro F1-Score       : {res['f1_macro']*100:.2f}%")
    print(f"   • 'closehand' F1       : {close_f1*100:.2f}%")
    print(f"   • 'openhand'  F1       : {open_f1*100:.2f}%")
    if res["roc_auc"]:
        print(f"   • ROC AUC              : {res['roc_auc']:.4f}")

    return res


def run_all_experiments() -> List[Dict[str, Any]]:
    """Uruchamia zaplanowany pakiet eksperymentów badawczych."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments_to_run = [
        # Wariant 1: ResNet-18 Binary (64x64) - Standard Baseline
        {
            "exp_id": "v1_resnet18_binary_base64",
            "model_name": "resnet18",
            "allowed_classes": ["closehand", "openhand"],
            "img_size": 64,
            "batch_size": 32,
            "epochs": 25,
            "lr": 1e-3,
            "dropout": 0.0,
            "patience": 10,
            "freeze_backbone_epochs": 0,
        },
        # Wariant 2: ResNet-18 Binary (112x112) - Wyższa rozdzielczość + Dropout (0.3)
        {
            "exp_id": "v2_resnet18_binary_hires_drop",
            "model_name": "resnet18",
            "allowed_classes": ["closehand", "openhand"],
            "img_size": 112,
            "batch_size": 32,
            "epochs": 25,
            "lr": 5e-4,
            "dropout": 0.3,
            "patience": 10,
            "freeze_backbone_epochs": 0,
        },
        # Wariant 3: ResNet-18 Binary (112x112) - Warmup głowicy przez pierwsze 3 epoki
        {
            "exp_id": "v3_resnet18_binary_hires_warmup",
            "model_name": "resnet18",
            "allowed_classes": ["closehand", "openhand"],
            "img_size": 112,
            "batch_size": 32,
            "epochs": 25,
            "lr": 3e-4,
            "dropout": 0.2,
            "patience": 10,
            "freeze_backbone_epochs": 3,
        },
        # Wariant 4: ResNet-34 Binary (112x112) - Głębsza sieć residualna
        {
            "exp_id": "v4_resnet34_binary_hires",
            "model_name": "resnet34",
            "allowed_classes": ["closehand", "openhand"],
            "img_size": 112,
            "batch_size": 32,
            "epochs": 25,
            "lr": 3e-4,
            "dropout": 0.2,
            "patience": 10,
            "freeze_backbone_epochs": 0,
        },
        # Wariant 5: ResNet-18 Multiclass (Wszystkie 8 klas, analiza close & open)
        {
            "exp_id": "v5_resnet18_multiclass_all8",
            "model_name": "resnet18",
            "allowed_classes": None,
            "img_size": 112,
            "batch_size": 32,
            "epochs": 30,
            "lr": 5e-4,
            "dropout": 0.2,
            "patience": 12,
            "freeze_backbone_epochs": 0,
        },
    ]

    all_results = []
    for cfg in experiments_to_run:
        res = run_single_experiment(**cfg)
        all_results.append(res)

    # Wykres porównawczy
    plot_comparison_chart(all_results, REPORTS_DIR / "experiments_comparison.png")

    # Tabela podsumowująca CSV & Markdown
    save_summary_tables(all_results, REPORTS_DIR)

    return all_results


def plot_comparison_chart(results: List[Dict[str, Any]], out_path: Path):
    """Rysuje wykres słupkowy porównujący dokładność, F1 i czas wszystkich wariantów."""
    exp_labels = [r["exp_id"].replace("v", "V").replace("_", "\n") for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    f1s = [r["f1_macro"] * 100 for r in results]
    close_f1s = [r["close_f1"] * 100 for r in results]
    open_f1s = [r["open_f1"] * 100 for r in results]

    x = np.arange(len(exp_labels))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(13, 6.5))

    rects1 = ax1.bar(x - 1.5 * width, accs, width, label='Val Accuracy (%)', color='#1f77b4')
    rects2 = ax1.bar(x - 0.5 * width, f1s, width, label='Macro F1 (%)', color='#2ca02c')
    rects3 = ax1.bar(x + 0.5 * width, close_f1s, width, label='CloseHand F1 (%)', color='#ff7f0e')
    rects4 = ax1.bar(x + 1.5 * width, open_f1s, width, label='OpenHand F1 (%)', color='#9467bd')

    ax1.set_ylabel('Wynik (%)', fontsize=12)
    ax1.set_title('Zestawienie Wariantów Treningowych ResNet (Apple Silicon Metal MPS)', fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', framealpha=0.95)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax1.annotate(f'{h:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n📊 Zapisano wykres porównania wariantów: {out_path}")


def save_summary_tables(results: List[Dict[str, Any]], out_dir: Path):
    """Zapisuje podsumowanie w plikach CSV i Markdown."""
    rows = []
    for r in results:
        rows.append({
            "Eksperyment": r["exp_id"],
            "Model": r["model_name"],
            "Tryb": "Binary (Close/Open)" if r["num_classes"] == 2 else "Multiclass (8)",
            "Rozdzielczość": f"{r['img_size']}x{r['img_size']}",
            "Val Acc (%)": f"{r['accuracy']*100:.2f}%",
            "Macro F1 (%)": f"{r['f1_macro']*100:.2f}%",
            "CloseHand F1 (%)": f"{r['close_f1']*100:.2f}%",
            "OpenHand F1 (%)": f"{r['open_f1']*100:.2f}%",
            "ROC AUC": f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
            "Czas (s)": f"{r['duration_sec']:.1f}",
            "Parametry": f"{r['param_count']:,}",
        })

    df = pd.DataFrame(rows)
    csv_file = out_dir / "summary_table.csv"
    md_file = out_dir / "summary_table.md"

    df.to_csv(csv_file, index=False)
    with open(md_file, "w") as f:
        f.write("# Podsumowanie Eksperymentów ResNet (PyTorch Lightning MPS)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    print(f"💾 Zapisano tabelę podsumowującą:\n   • {csv_file}\n   • {md_file}")


if __name__ == "__main__":
    run_all_experiments()
