#!/usr/bin/env python3
"""
scripts/train_resnet.py

Dedykowany skrypt CLI do dotrenowywania (fine-tuning) architektury ResNet (ResNet-18, ResNet-34)
w PyTorch Lightning z pełnym wsparciem Apple Silicon Metal (MPS).

Główne cechy:
- Domyślny fokus na etykietach 'closehand' i 'openhand' (flaga --binary lub --classes)
- Wykorzystanie pretrenowanych wag ImageNet
- Elastyczny transfer learning (zamrożenie backbone na N epok z automatycznym odmrożeniem)
- Regularyzacja Dropout w głowicy klasyfikacyjnej
- Generowanie zestawu wizualizacji (krzywe uczenia, macierz pomyłek, krzywe ROC/PR, siatka predykcji)
- Eksport wag do .pt, .ckpt oraz opcjonalnie ONNX / TorchScript

Przykłady użycia:
    # 1. Szybki trening ResNet-18 na closehand vs openhand (64x64):
    python scripts/train_resnet.py --binary --epochs 30 --batch-size 32

    # 2. Trening z wyższą rozdzielczością (112x112), dropoutem i 4 epokami warmup:
    python scripts/train_resnet.py --binary --img-size 112 --dropout 0.3 --freeze-epochs 4 --lr 5e-4

    # 3. Trening na wszystkich 8 klasach gestów:
    python scripts/train_resnet.py --all-classes --epochs 35 --img-size 112
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

# Upewniamy się, że katalog projektu jest w sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightning_data import GestureDataModule
from src.lightning_module import GestureLightningModule, build_model
from src.lightning_trainer import create_lightning_trainer
from src.dataset import denormalize, get_device
from src.export import export_model


class BackboneWarmupCallback(pl.Callback):
    """Callback zamrażający wagi backbone na okres wstępnego dopasowania głowicy."""
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
                trainer.print(f"🧊 [Warmup] Zamrożono wagi backbone na pierwsze {self.unfreeze_epoch} epok.")
        elif trainer.current_epoch == self.unfreeze_epoch:
            if hasattr(pl_module.model, "backbone"):
                for p in pl_module.model.backbone.parameters():
                    p.requires_grad = True
                trainer.print(f"🔥 [Fine-tuning] Odmrożono wszystkie wagi backbone do pełnego dotrenowywania!")


def plot_learning_curves(history: Dict[str, list], exp_name: str, out_path: Path):
    """Rysuje estetyczne krzywe Loss, Accuracy oraz Learning Rate."""
    if not history or not history.get("epoch"):
        return

    valid_train = [(e + 1, l, a) for e, l, a in zip(history["epoch"], history.get("train_loss", []), history.get("train_acc", [])) if l is not None and a is not None]
    valid_val = [(e + 1, l, a) for e, l, a in zip(history["epoch"], history.get("val_loss", []), history.get("val_acc", [])) if l is not None and a is not None]
    valid_lr = [(e + 1, lr) for e, lr in zip(history["epoch"], history.get("lr", [])) if lr is not None]

    if not valid_train and not valid_val:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Historia Treningu ResNet: {exp_name} (Apple Silicon MPS)", fontsize=14, fontweight="bold")
    ax1, ax2, ax3 = axes

    # 1. Loss
    if valid_train:
        t_ep, t_loss, t_acc = zip(*valid_train)
        ax1.plot(t_ep, t_loss, label="Train Loss", color="#1f77b4", lw=2.2)
        ax2.plot(t_ep, [a * 100 for a in t_acc], label="Train Acc", color="#2ca02c", lw=2.2)

    if valid_val:
        v_ep, v_loss, v_acc = zip(*valid_val)
        val_accs = [a * 100 for a in v_acc]
        ax1.plot(v_ep, v_loss, label="Val Loss", color="#ff7f0e", lw=2.2, linestyle="--")
        ax2.plot(v_ep, val_accs, label="Val Acc", color="#d62728", lw=2.5)
        best_val = max(val_accs)
        best_idx = val_accs.index(best_val)
        best_epoch = v_ep[best_idx]
        ax2.scatter([best_epoch], [best_val], color="#d62728", s=100, zorder=5)
        ax2.annotate(f"Best: {best_val:.2f}% (ep {best_epoch})", (best_epoch, best_val),
                     textcoords="offset points", xytext=(-25, 10), fontweight="bold")

    ax1.set_title("Funkcja Straty (Cross-Entropy)", fontweight="bold")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_title("Dokładność (Accuracy %)", fontweight="bold")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("%")
    ax2.set_ylim(-2, 103)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # 3. Learning Rate
    if valid_lr:
        lr_ep, lrs = zip(*valid_lr)
        ax3.plot(lr_ep, lrs, label="Learning Rate", color="#9467bd", lw=2.0)
    ax3.set_title("Harmonogram LR (Cosine Annealing)", fontweight="bold")
    ax3.set_xlabel("Epoka")
    ax3.set_ylabel("Wartość LR")
    ax3.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix_and_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    exp_name: str,
    out_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Generuje macierz pomyłek z podwójną adnotacją (liczność + procent)."""
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
    plt.xlabel("Rozpoznany gest (Predykcja ResNet)", fontsize=11)
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
    """Dla klasyfikacji binarnej (open vs close) rysuje krzywe ROC oraz Precision-Recall."""
    if len(class_names) != 2:
        return

    # Przyjmujemy drugą klasę jako pozytywną (zwykle openhand)
    pos_idx = 1
    pos_label = class_names[pos_idx]
    y_scores = all_probs[:, pos_idx]
    y_true = (all_targets == pos_idx).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Krzywe Oceny Jakości: {exp_name} ({class_names[0]} vs {class_names[1]})", fontsize=13, fontweight="bold")

    # ROC
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("Krzywa ROC", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # PR
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


def visualize_predictions_grid(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    exp_name: str,
    out_path: Path,
    device: torch.device,
    num_samples: int = 16,
):
    """Generuje estetyczną siatkę predykcji na klatkach z drona."""
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
        border_color = "#2ca02c" if is_correct else "#d62728"

        img_np = denormalize(img_tensor)

        ax.imshow(img_np)
        title_text = f"Prawda: {t_cls}\nPred: {p_cls} ({conf*100:.1f}%)"
        ax.set_title(title_text, color=border_color, fontsize=10, fontweight="bold", pad=4)
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    for j in range(len(selected_idx), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Próbki predykcji ResNet: {exp_name}\n(Zielona ramka = Zgodność, Czerwona = Błąd)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning modeli ResNet na gestach dłoni w PyTorch Lightning")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet", "residual"],
                        help="Architektura modelu do dotrenowywania (domyślnie resnet18)")
    parser.add_argument("--binary", action="store_true", default=False,
                        help="Trening binarny skupiony na 'closehand' i 'openhand'")
    parser.add_argument("--all-classes", action="store_true", default=False,
                        help="Trening na wszystkich 8 klasach gestów")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Opcjonalna lista klas (np. --classes closehand openhand)")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu (domyślnie 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Wielkość batcha (domyślnie 32)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate dla AdamW (domyślnie 5e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Waga regularyzacji L2 (domyślnie 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Współczynnik Dropout w głowicy (domyślnie 0.2)")
    parser.add_argument("--img-size", type=int, default=112, help="Rozmiar klatki wejściowej NxN (domyślnie 112)")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Liczba epok zamrożenia wag backbone na warmup (domyślnie 0 = brak)")
    parser.add_argument("--patience", type=int, default=12, help="Cierpliwość EarlyStopping (domyślnie 12)")
    parser.add_argument("--name", type=str, default=None, help="Własna nazwa eksperymentu")
    parser.add_argument("--save-dir", type=str, default="weights", help="Folder zapisu wag")
    parser.add_argument("--reports-dir", type=str, default="reports/resnet_experiments", help="Folder zapisu wizualizacji")
    parser.add_argument("--export", action="store_true", help="Eksport najlepszego modelu do ONNX i TorchScript")
    return parser.parse_args()


def main():
    args = parse_args()

    # Wybór klas: domyślnie priorytet dla closehand i openhand
    if args.all_classes:
        allowed_classes = None
        mode_desc = "Pełny zbiór (8 klas)"
    elif args.classes is not None:
        allowed_classes = args.classes
        mode_desc = f"Wybrane klasy ({allowed_classes})"
    else:
        # Domyślnie lub z flagą --binary: closehand i openhand
        allowed_classes = ["closehand", "openhand"]
        mode_desc = "Fokus Binarny: 'closehand' vs 'openhand'"

    exp_name = args.name or f"{args.model}_{('binary' if allowed_classes else 'all8')}_{args.img_size}px"
    reports_dir = Path(args.reports_dir)
    save_dir = Path(args.save_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 76)
    print("🚁 [PyTorch Lightning ResNet Training Pipeline]")
    print(f"   Model          : {args.model.upper()}")
    print(f"   Tryb Klas      : {mode_desc}")
    print(f"   Rozdzielczość  : {args.img_size}x{args.img_size}")
    print(f"   Epoki / Batch  : {args.epochs} / {args.batch_size}")
    print(f"   LR / Decay     : {args.lr} / {args.weight_decay}")
    print(f"   Dropout / Warm : {args.dropout} / {args.freeze_epochs} epok")
    print(f"   Akcelerator    : {device.type.upper()} (Apple Metal)")
    print(f"   Eksperyment    : {exp_name}")
    print("=" * 76)

    # 1. DataModule
    dm = GestureDataModule(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=2,
        allowed_classes=allowed_classes,
    )
    dm.setup("fit")
    num_classes = dm.num_classes
    class_names = dm.class_names
    print(f"📦 Dane: {len(dm.train_dataset)} train, {len(dm.val_dataset)} val. Klasy: {class_names}")

    # 2. Budowa modelu bazowego z dropoutem
    base_model = build_model(args.model, num_classes=num_classes, dropout=args.dropout)

    # 3. LightningModule
    pl_module = GestureLightningModule(
        model=base_model,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=0.05,
        optimizer_name="adamw",
        scheduler_name="cosine",
        class_names=class_names,
    )

    # 4. Trener Lightning z callbackiem warmup
    trainer = create_lightning_trainer(
        experiment_name=exp_name,
        max_epochs=args.epochs,
        save_dir=str(save_dir),
        log_dir="runs/lightning",
        early_stopping_patience=args.patience,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=True,
    )

    if args.freeze_epochs > 0:
        trainer.callbacks.append(BackboneWarmupCallback(unfreeze_epoch=args.freeze_epochs))

    # 5. Start Treningu
    t0 = time.time()
    trainer.fit(pl_module, datamodule=dm)
    duration = time.time() - t0
    print(f"\n⏱️  Trening trwał: {duration:.1f} s ({duration/60:.2f} min)")

    # 6. Odczyt historii metryk z MetricsHistoryCallback
    history = None
    for cb in trainer.callbacks:
        if hasattr(cb, "history"):
            history = cb.history
            break

    # 7. Generowanie wykresów zbieżności
    curves_path = reports_dir / f"curves_{exp_name}.png"
    if history:
        plot_learning_curves(history, exp_name, curves_path)
        print(f"📈 Zapisano wykresy krzywych uczenia: {curves_path}")

    # 8. Ewaluacja na najlepszych wagach
    best_weights_path = save_dir / f"{exp_name}_best.pt"
    eval_model = build_model(args.model, num_classes=num_classes, dropout=args.dropout)
    if best_weights_path.exists():
        eval_model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"🏆 Wczytano najlepsze wagi do ewaluacji: {best_weights_path}")
    else:
        eval_model.load_state_dict(pl_module.model.state_dict())

    val_loader = dm.val_dataloader()

    # Macierz pomyłek
    cm_path = reports_dir / f"cm_{exp_name}.png"
    eval_res = plot_confusion_matrix_and_metrics(eval_model, val_loader, class_names, exp_name, cm_path, device)
    print(f"📊 Zapisano macierz pomyłek: {cm_path}")

    # Krzywe ROC & PR dla klasyfikacji binarnej
    if num_classes == 2:
        roc_path = reports_dir / f"roc_pr_{exp_name}.png"
        plot_binary_roc_pr(eval_res["all_targets"], eval_res["all_probs"], class_names, exp_name, roc_path)
        print(f"📉 Zapisano krzywe ROC & PR: {roc_path}")

    # Podgląd predykcji na klatkach z drona
    samples_path = reports_dir / f"samples_{exp_name}.png"
    visualize_predictions_grid(eval_model, val_loader, class_names, exp_name, samples_path, device)
    print(f"🖼️  Zapisano siatkę predykcji: {samples_path}")

    # Opcjonalny eksport do formatów wdrożeniowych (ONNX, TorchScript)
    if args.export:
        print("\n📦 Eksportowanie do TorchScript i ONNX...")
        export_model(
            model=eval_model,
            output_dir=str(save_dir),
            model_name=f"{exp_name}_exported",
            input_size=(1, 3, args.img_size, args.img_size),
        )

    # Raport końcowy w konsoli
    print("\n" + "=" * 70)
    print(f"🏁 PODSUMOWANIE EKSPERYMENTU: {exp_name}")
    print(f"   • Dokładność (Val Accuracy) : {eval_res['accuracy']*100:.2f}%")
    print(f"   • Macro F1-Score            : {eval_res['f1_macro']*100:.2f}%")
    print(f"   • Macro Precision           : {eval_res['precision_macro']*100:.2f}%")
    print(f"   • Macro Recall              : {eval_res['recall_macro']*100:.2f}%")
    for c in class_names:
        c_p = eval_res["report_dict"].get(c, {}).get("precision", 0)
        c_r = eval_res["report_dict"].get(c, {}).get("recall", 0)
        c_f1 = eval_res["report_dict"].get(c, {}).get("f1-score", 0)
        print(f"   • Gest '{c:<10}': Precision={c_p*100:5.1f}%, Recall={c_r*100:5.1f}%, F1={c_f1*100:5.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
