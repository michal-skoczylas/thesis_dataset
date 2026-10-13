import os
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch

from config import RAW_DIR, LABELED_DIR


def configure_device():
    """Wykrywa najlepsze dostępne urządzenie: CUDA -> Apple MPS (Metal) -> CPU."""
    if torch.cuda.is_available():
        device_name = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    try:
        import autodistill_grounding_dino.helpers as h
        h.DEVICE = torch.device(device_name)
    except Exception as e:
        print(f"[Warning] Nie udało się ustawić h.DEVICE: {e}")

    print(f"[Hardware] Wykryte urządzenie akceleracji: {device_name.upper()}")
    return torch.device(device_name)


def filter_top_detections(detections, max_detections=2):
    """Zwraca co najwyżej max_detections najbardziej pewnych detekcji."""
    if len(detections) <= max_detections:
        return detections
    best_indices = np.argsort(detections.confidence)[-max_detections:][::-1]
    return detections[best_indices]


def run_detection(
    input_dir: Path = RAW_DIR,
    output_dir: Path = LABELED_DIR,
    max_hands: int = 2,
    target_class: str = "hand",
    force_all: bool = False
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Znajdź wszystkie zdjęcia w katalogu wejściowym
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_raw_images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in valid_exts]

    # 2. Sprawdź, które obrazy już zostały przetworzone (mają etykietę .txt)
    existing_stems = {p.stem for p in output_dir.rglob("*.txt") if p.name != "classes.txt"}

    if force_all:
        images_to_process = all_raw_images
        print(f"[Tryb: Wymuszony] Przetwarzanie wszystkich {len(images_to_process)} obrazów...")
    else:
        images_to_process = [p for p in all_raw_images if p.stem not in existing_stems]

    print(f"[Dataset] Łącznie w raw: {len(all_raw_images)}, już oznaczonych: {len(existing_stems)}, nowych do przetworzenia: {len(images_to_process)}")

    if not images_to_process:
        print("✅ Wszystkie zdjęcia są już przetworzone! Brak nowych klatek do analizy.")
        return

    # 3. Konfiguracja sprzętu (Apple Metal / CUDA)
    configure_device()

    # 4. Ładowanie modelu
    from autodistill_grounding_dino import GroundingDINO
    from autodistill.detection import CaptionOntology

    print(f"Ładowanie modelu Grounding DINO dla klasy '{target_class}'...")
    ontology = CaptionOntology({target_class: target_class})
    model = GroundingDINO(ontology=ontology)

    # 5. Przetwarzanie tylko nowych obrazów
    for idx, img_path in enumerate(images_to_process, 1):
        print(f"[{idx}/{len(images_to_process)}] Analiza klatki: {img_path.name}...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [Błąd] Nie można wczytać obrazu {img_path.name}, pomijam.")
            continue

        h_img, w_img = img.shape[:2]

        # Predykcja
        detections = model.predict(str(img_path))
        detections = filter_top_detections(detections, max_detections=max_hands)

        # Konwersja ramek na format YOLO: <class_id> <cx> <cy> <w> <h>
        yolo_lines = []
        if len(detections) > 0:
            for box in detections.xyxy:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / (2.0 * w_img)
                cy = (y1 + y2) / (2.0 * h_img)
                bw = (x2 - x1) / float(w_img)
                bh = (y2 - y1) / float(h_img)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # Zapis pliku etykiety
        out_txt_path = train_labels_dir / f"{img_path.stem}.txt"
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        # Kopiowanie/zapis obrazu do train/images
        out_img_path = train_images_dir / f"{img_path.stem}.jpg"
        cv2.imwrite(str(out_img_path), img)

    # 6. Zapis klas i metadanych YOLO
    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write(f"{target_class}\n")

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"names:\n- {target_class}\nnc: 1\ntrain: {train_images_dir}\n")

    print(f"🎉 Zakończono! Zaktualizowano zbiór w: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detekcja rąk za pomocą Grounding DINO (tryb przyrostowy)")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DIR), help="Katalog wejściowy ze zdjęciami")
    parser.add_argument("--output-dir", type=str, default=str(LABELED_DIR), help="Katalog wyjściowy na adnotacje YOLO")
    parser.add_argument("--max-hands", type=int, default=2, help="Maksymalna liczba rąk na obraz")
    parser.add_argument("--extension", type=str, default=".png", help="Rozszerzenie (opcjonalne)")
    parser.add_argument("--class-name", type=str, default="hand", help="Nazwa klasy")
    parser.add_argument("--force", action="store_true", help="Wymuszenie ponownej analizy wszystkich klatek od zera")
    args = parser.parse_args()

    run_detection(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_hands=args.max_hands,
        target_class=args.class_name,
        force_all=args.force
    )