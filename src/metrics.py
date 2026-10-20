from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import get_device, denormalize


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje raport klasyfikacji (Precision, Recall, F1) oraz wykres Macierzy Pomyłek (Confusion Matrix).
    """
    dev = device or get_device()
    model.eval()
    model = model.to(dev)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(dev)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Raport klasyfikacji w konsoli
    print("=" * 65)
    print(f"📋 RAPORT KLASYFIKACJI ({title})")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # 2. Rysowanie macierzy pomyłek
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Prawdopodobieństwo (znormalizowane)"},
    )
    plt.title(f"{title} (zbiór walidacyjny)", fontsize=13, pad=12)
    plt.ylabel("Rzeczywista klasa (Ground Truth)", fontsize=11)
    plt.xlabel("Predykcja modelu", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300)
        print(f"💾 Zapisano wykres macierzy: {out}")

    plt.show()
    return all_preds, all_targets


def show_misclassified_samples(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    max_images: int = 8,
    device: Optional[torch.device] = None,
):
    """Wyświetla siatkę wycinków dłoni, na których model popełnił błąd (Error Analysis)."""
    dev = device or get_device()
    model.eval()
    model = model.to(dev)

    errors = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs.to(dev))
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            for i in range(len(targets)):
                pred_cls = preds[i].cpu().item()
                true_cls = targets[i].item()
                if pred_cls != true_cls:
                    errors.append({
                        "image": inputs[i],
                        "true_label": class_names[true_cls],
                        "pred_label": class_names[pred_cls],
                        "confidence": confidences[i].item() * 100,
                    })
                if len(errors) >= max_images:
                    break
            if len(errors) >= max_images:
                break

    if not errors:
        print("🎉 Model nie popełnił żadnego błędu na wybranym podzbiorze!")
        return

    cols = min(4, len(errors))
    rows = (len(errors) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    for idx, err in enumerate(errors):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(denormalize(err["image"]))
        plt.title(
            f"True: {err['true_label']}\nPred: {err['pred_label']} ({err['confidence']:.1f}%)",
            color="darkred",
            fontweight="bold",
            fontsize=11,
        )
        plt.axis("off")

    plt.suptitle("Błędnie sklasyfikowane próbki walidacyjne (Error Analysis)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
