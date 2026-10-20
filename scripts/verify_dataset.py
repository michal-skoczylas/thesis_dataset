#!/usr/bin/env python3
"""
scripts/verify_dataset.py
Weryfikacja spójności zbioru danych YOLO (Etap 1) oraz wycinków gestów Custom CNN (Etap 2).
Generuje również poglądową mozaikę zdjęć (data/crops_preview.jpg) z podpisanymi klasami gestów.
"""

import sys
import csv
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

try:
    from config import DATA_DIR
except ImportError:
    DATA_DIR = BASE_DIR / "data"

YOLO_DIR = DATA_DIR / "yolo_dataset"
CNN_DIR = DATA_DIR / "cnn_dataset"
CSV_PATH = CNN_DIR / "crops_metadata.csv"
PREVIEW_IMG_PATH = DATA_DIR / "crops_preview.jpg"


def verify_datasets():
    print("=" * 68)
    print("🔍 AUDYT I WERYFIKACJA SPÓJNOŚCI ZBIORU DANYCH (YOLO & CNN)")
    print("=" * 68)

    # 1. Weryfikacja YOLO
    print("\n1️⃣  ETAP 1: YOLO DATASET")
    if not YOLO_DIR.exists():
        print(f"   ⚠️  Katalog {YOLO_DIR} jeszcze nie istnieje. Uruchom 'make prepare-yolo-zip'.")
    else:
        yolo_train_imgs = list((YOLO_DIR / "images" / "train").glob("*.*")) if (YOLO_DIR / "images" / "train").exists() else []
        yolo_val_imgs = list((YOLO_DIR / "images" / "val").glob("*.*")) if (YOLO_DIR / "images" / "val").exists() else []
        yolo_train_lbls = list((YOLO_DIR / "labels" / "train").glob("*.txt")) if (YOLO_DIR / "labels" / "train").exists() else []
        yolo_val_lbls = list((YOLO_DIR / "labels" / "val").glob("*.txt")) if (YOLO_DIR / "labels" / "val").exists() else []

        print(f"   📁 Ścieżka        : {YOLO_DIR}")
        print(f"   📸 Klatki (TRAIN) : {len(yolo_train_imgs)} zdjęć, {len(yolo_train_lbls)} etykiet")
        print(f"   📸 Klatki (VAL)   : {len(yolo_val_imgs)} zdjęć, {len(yolo_val_lbls)} etykiet")

        # Sprawdzenie poprawności współrzędnych
        corrupted_boxes = 0
        for lbl in yolo_train_lbls + yolo_val_lbls:
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        vals = [float(p) for p in parts[1:5]]
                        if any(v < 0.0 or v > 1.0 for v in vals):
                            corrupted_boxes += 1
        if corrupted_boxes == 0:
            print("   ✅ Wszystkie bounding boxy YOLO są poprawnie znormalizowane [0.0 - 1.0].")
        else:
            print(f"   ⚠️  Znaleziono {corrupted_boxes} ramek z nieprawidłowymi współrzędnymi!")

    # 2. Weryfikacja CNN
    print("\n2️⃣  ETAP 2: CUSTOM CNN DATASET (Klasyfikacja gestów)")
    if not CNN_DIR.exists():
        print(f"   ⚠️  Katalog {CNN_DIR} jeszcze nie istnieje. Uruchom 'make prepare-yolo-zip'.")
        return

    train_classes = [d.name for d in (CNN_DIR / "train").iterdir() if d.is_dir()] if (CNN_DIR / "train").exists() else []
    val_classes = [d.name for d in (CNN_DIR / "val").iterdir() if d.is_dir()] if (CNN_DIR / "val").exists() else []

    all_classes = sorted(list(set(train_classes + val_classes)))
    print(f"   📁 Ścieżka        : {CNN_DIR}")
    print(f"   🏷️  Wykryte klasy  : {', '.join(all_classes) if all_classes else 'Brak'}")

    total_train_crops = 0
    total_val_crops = 0

    print("\n   📊 Szczegółowy rozkład wycinków per klasa:")
    for cls_name in all_classes:
        tr_c = len(list((CNN_DIR / "train" / cls_name).glob("*.*"))) if (CNN_DIR / "train" / cls_name).exists() else 0
        va_c = len(list((CNN_DIR / "val" / cls_name).glob("*.*"))) if (CNN_DIR / "val" / cls_name).exists() else 0
        total_train_crops += tr_c
        total_val_crops += va_c
        print(f"      • {cls_name:<12} : {tr_c:>3} train | {va_c:>3} val (łącznie: {tr_c + va_c})")

    print(f"\n   📈 Sumarycznie     : {total_train_crops} train / {total_val_crops} val (Łącznie: {total_train_crops + total_val_crops} wycinków dłoni)")

    # 3. Weryfikacja spójności tabeli CSV
    if CSV_PATH.exists():
        print(f"\n3️⃣  SPÓJNOŚĆ TABELI METADANYCH ({CSV_PATH.name}):")
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"   📄 Wpisów w tabeli: {len(rows)}")

        unknown_gestures = [r for r in rows if r.get("gesture") == "unknown"]
        if unknown_gestures:
            print(f"   ⚠️  Uwaga: {len(unknown_gestures)} wycinków ma nieznaną klasę ('unknown')!")
        else:
            print("   ✅ Każdy wycinek dłoni ma precyzyjnie przypisaną kanoniczną klasę gestu.")

    # 4. Generowanie mozaiki podglądowej
    generate_preview_grid(rows if CSV_PATH.exists() else [])


def generate_preview_grid(rows, max_samples=12):
    """Tworzy mozaikę podglądową wycinków z podpisanymi klasami gestów."""
    if not rows:
        return

    sample_rows = rows[:max_samples]
    cells = []
    cell_size = 120

    for r in sample_rows:
        split = r.get("split", "train")
        gesture = r.get("gesture", "unknown")
        fname = r.get("crop_filename", "")
        img_path = CNN_DIR / split / gesture / fname

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Przeskaluj do kwadratu cell_size z zachowaniem proporcji
        h, w = img.shape[:2]
        scale = (cell_size - 30) / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        # Ramka z tłem
        cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        cell[:] = (35, 35, 35)  # ciemne tło

        y_off = (cell_size - 30 - nh) // 2
        x_off = (cell_size - nw) // 2
        cell[y_off:y_off+nh, x_off:x_off+nw] = resized

        # Podpis klasy
        cv2.putText(cell, gesture, (5, cell_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
        cells.append(cell)

    if cells:
        # Ułóż w siatkę np. do 6 w rzędzie
        cols = min(6, len(cells))
        rows_count = (len(cells) + cols - 1) // cols
        grid_w = cols * cell_size
        grid_h = rows_count * cell_size

        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        grid[:] = (20, 20, 20)

        for idx, c in enumerate(cells):
            r_idx = idx // cols
            c_idx = idx % cols
            y = r_idx * cell_size
            x = c_idx * cell_size
            grid[y:y+cell_size, x:x+cell_size] = c

        cv2.imwrite(str(PREVIEW_IMG_PATH), grid)
        print(f"\n🖼️  Wygenerowano mozaikę weryfikacyjną: {PREVIEW_IMG_PATH}")
        print("    Możesz otworzyć ten plik, aby zobaczyć wycinki dłoni z podpisanymi klasami gestów!")
    print("=" * 68)


if __name__ == "__main__":
    verify_datasets()
