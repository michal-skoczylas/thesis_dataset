#!/usr/bin/env python3
"""
scripts/visualize_yolo_results.py

Generuje zestawienie wyników treningu YOLO:
1. Wykresy metryk (Loss, mAP50, mAP50-95, Precision, Recall) z zaznaczonym najlepszym punktem.
2. Wizualną siatkę predykcji na zbiorze walidacyjnym (Ground Truth vs Predykcja YOLO z pewnością i IoU).
3. Eksport podglądu do katalogu artefaktów.
"""

import os
import sys
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "yolo_dataset"
RUNS_DIR = BASE_DIR / "runs" / "detect" / "hand_detector"
WEIGHTS_PATH = BASE_DIR / "weights" / "yolo_hand_detector.pt"


def calculate_iou(box1, box2):
    """Oblicza IoU pomiędzy dwoma boxami w formacie [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def plot_training_curves(results_csv_path: Path, output_img: Path):
    """Rysuje szczegółowe wykresy przebiegu uczenia YOLO."""
    if not results_csv_path.exists():
        print(f"⚠️  Brak pliku {results_csv_path}, pomijam wykresy krzywych uczenia.")
        return None

    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]

    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Raport Treningu Detektora Dłoni YOLO (Apple Metal MPS)", fontsize=16, fontweight="bold")

    # 1. Losses
    ax1 = axes[0, 0]
    if "train/box_loss" in df.columns:
        ax1.plot(epochs, df["train/box_loss"], label="Train Box Loss", color="#1f77b4", lw=2)
    if "val/box_loss" in df.columns:
        ax1.plot(epochs, df["val/box_loss"], label="Val Box Loss", color="#ff7f0e", lw=2, linestyle="--")
    if "train/dfl_loss" in df.columns:
        ax1.plot(epochs, df["train/dfl_loss"], label="Train DFL Loss", color="#2ca02c", lw=1.5, alpha=0.7)
    ax1.set_title("Funkcje Straty (Losses)", fontweight="bold")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Strata")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. mAP Metrics
    ax2 = axes[0, 1]
    best_map50 = 0.0
    best_epoch = 1
    if "metrics/mAP50(B)" in df.columns:
        ax2.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.50", color="#d62728", lw=2.5)
        best_idx = df["metrics/mAP50(B)"].idxmax()
        best_map50 = df["metrics/mAP50(B)"][best_idx]
        best_epoch = epochs.iloc[best_idx] if hasattr(epochs, "iloc") else epochs[best_idx]
        ax2.scatter([best_epoch], [best_map50], color="#d62728", s=100, zorder=5)
        ax2.annotate(f"Best: {best_map50*100:.2f}%", (best_epoch, best_map50),
                     textcoords="offset points", xytext=(-20, 10), fontweight="bold")

    if "metrics/mAP50-95(B)" in df.columns:
        ax2.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.50:0.95", color="#9467bd", lw=2)

    ax2.set_title("Średnia Precyzja (mAP)", fontweight="bold")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("mAP")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Precision & Recall
    ax3 = axes[1, 0]
    if "metrics/precision(B)" in df.columns:
        ax3.plot(epochs, df["metrics/precision(B)"], label="Precyzja (Precision)", color="#2ca02c", lw=2)
    if "metrics/recall(B)" in df.columns:
        ax3.plot(epochs, df["metrics/recall(B)"], label="Czułość (Recall)", color="#17becf", lw=2)
    ax3.set_title("Precyzja i Czułość", fontweight="bold")
    ax3.set_xlabel("Epoka")
    ax3.set_ylabel("Wartość")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Learning Rate & Box Loss
    ax4 = axes[1, 1]
    if "lr/pg0" in df.columns:
        ax4.plot(epochs, df["lr/pg0"], label="Learning Rate (lr)", color="#e377c2", lw=2)
        ax4.set_ylabel("Learning Rate", color="#e377c2")
        ax4.tick_params(axis='y', labelcolor="#e377c2")
    ax4.set_title("Harmonogram Learning Rate", fontweight="bold")
    ax4.set_xlabel("Epoka")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img, dpi=200)
    plt.close()
    print(f"📊 Zapisano wykres krzywych uczenia: {output_img}")
    return output_img


def visualize_validation_predictions(
    model_path: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    output_grid_path: Path,
    num_samples: int = 12,
    conf_thresh: float = 0.25
):
    """Generuje siatkę obrazów walidacyjnych z nałożonymi ramkami GT (zielona) i Predykcją (czerwona/błękitna)."""
    if not model_path.exists():
        print(f"❌ Brak wag modelu: {model_path}")
        return None

    model = YOLO(str(model_path))
    val_images = sorted(list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")))
    if not val_images:
        print("❌ Brak obrazów w katalogu walidacyjnym!")
        return None

    # Dobierz reprezentatywne próbki (losowo, ale ze stałym ziarnem)
    random.seed(42)
    selected_images = random.sample(val_images, min(num_samples, len(val_images)))

    cols = 4
    rows = (len(selected_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.array(axes).reshape(-1)

    for idx, img_path in enumerate(selected_images):
        ax = axes[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Wczytaj Ground Truth (etykiety z pliku .txt)
        label_file = val_labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, bw, bh = map(float, parts)
                        x1 = (cx - bw / 2.0) * w_img
                        y1 = (cy - bh / 2.0) * h_img
                        x2 = (cx + bw / 2.0) * w_img
                        y2 = (cy + bh / 2.0) * h_img
                        gt_boxes.append([x1, y1, x2, y2])

        # 2. Wykonaj inferencję wytrenowanego modelu
        preds = model(img_bgr, conf=conf_thresh, verbose=False)[0]
        pred_boxes = []
        confs = []
        if len(preds.boxes) > 0:
            pred_boxes = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()

        # 3. Rysuj Ground Truth (Zielona ramka)
        for box in gt_boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 fill=False, edgecolor='#00FF00', linewidth=2.5, linestyle='-')
            ax.add_patch(rect)

        # 4. Rysuj Predykcje YOLO (Czerwona/Niebieska ramka)
        best_iou_str = "Brak detekcji"
        for p_box, conf in zip(pred_boxes, confs):
            rect = plt.Rectangle((p_box[0], p_box[1]), p_box[2] - p_box[0], p_box[3] - p_box[1],
                                 fill=False, edgecolor='#FF3333', linewidth=2, linestyle='--')
            ax.add_patch(rect)
            ax.text(p_box[0], max(10, p_box[1] - 4), f"hand {conf:.2f}",
                    bbox=dict(facecolor='#FF3333', alpha=0.75, edgecolor='none', pad=1),
                    fontsize=8, color='white', weight='bold')

            if gt_boxes:
                ious = [calculate_iou(p_box, g) for g in gt_boxes]
                best_iou = max(ious)
                best_iou_str = f"IoU: {best_iou*100:.1f}%"

        ax.imshow(img_rgb)
        ax.set_title(f"{img_path.name[:25]}...\n[GT: {len(gt_boxes)} | Pred: {len(pred_boxes)} | {best_iou_str}]",
                     fontsize=9)
        ax.axis('off')

    # Wyłącz puste podwykresy
    for idx in range(len(selected_images), len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Wizualizacja Walidacji Detektora Dłoni (Zielony: Ground Truth, Czerwony przerywany: Predykcja YOLO)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    output_grid_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🖼️  Zapisano siatkę predykcji walidacyjnych: {output_grid_path}")
    return output_grid_path


if __name__ == "__main__":
    csv_file = RUNS_DIR / "results.csv"
    curves_out = RUNS_DIR / "thesis_training_metrics.png"
    grid_out = RUNS_DIR / "val_predictions_preview.jpg"

    val_imgs = DATA_DIR / "images" / "val"
    val_lbls = DATA_DIR / "labels" / "val"

    plot_training_curves(csv_file, curves_out)
    visualize_validation_predictions(WEIGHTS_PATH, val_imgs, val_lbls, grid_out)
