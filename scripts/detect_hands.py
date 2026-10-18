import os
import sys
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from config import RAW_DIR, LABELED_DIR, WEIGHTS_DIR


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
    except Exception:
        pass

    print(f"[Hardware] Wykryte urządzenie akceleracji: {device_name.upper()}")
    return device_name


def filter_top_detections(detections, max_detections=2):
    """Zwraca co najwyżej max_detections najbardziej pewnych detekcji."""
    if len(detections) <= max_detections:
        return detections
    best_indices = np.argsort(detections.confidence)[-max_detections:][::-1]
    return detections[best_indices]


def get_unique_stem(img_path: Path, raw_dir: Path) -> str:
    """
    Tworzy unikalną nazwę pliku uwzględniającą strukturę podkatalogów (np. odległość, gest, data).
    Przykład:
      data/raw/2m/OpenHand/2026-10-15_143000/DJI_0001.jpg
      -> 2m_OpenHand_2026-10-15_143000_DJI_0001
      data/raw/00000.png -> 00000
    """
    try:
        rel = img_path.relative_to(raw_dir)
        if len(rel.parts) > 1:
            clean_subdirs = [p.replace(" ", "_") for p in rel.parts[:-1]]
            return f"{'_'.join(clean_subdirs)}_{img_path.stem}"
    except ValueError:
        pass
    return img_path.stem


def run_detection(
    input_dir: Path = RAW_DIR,
    output_dir: Path = LABELED_DIR,
    max_hands: int = 2,
    target_class: str = "hand",
    force_all: bool = False,
    engine: str = "auto",
    weights_path: Path = None,
    conf: float = 0.25
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rekurencyjne wyszukiwanie wszystkich zdjęć w katalogu wejściowym i podkatalogach
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_raw_images = [
        p for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    # 2. Sprawdź, które obrazy już zostały przetworzone (mają etykietę .txt)
    existing_stems = {p.stem for p in output_dir.rglob("*.txt") if p.name != "classes.txt"}

    if force_all:
        images_to_process = all_raw_images
        print(f"[Tryb: Wymuszony] Przetwarzanie wszystkich {len(images_to_process)} obrazów...")
    else:
        images_to_process = [
            p for p in all_raw_images
            if get_unique_stem(p, input_dir) not in existing_stems
        ]

    print(f"[Dataset] Łącznie w raw: {len(all_raw_images)}, już oznaczonych: {len(existing_stems)}, nowych do przetworzenia: {len(images_to_process)}")

    if not images_to_process:
        print("✅ Wszystkie zdjęcia są już przetworzone! Brak nowych klatek do analizy.")
        return

    # 3. Konfiguracja sprzętu (Apple Metal / CUDA)
    device_name = configure_device()

    # 4. Wybór silnika detekcji (Active Learning: Twój YOLO vs Grounding DINO)
    if weights_path is None:
        weights_path = WEIGHTS_DIR / "yolo_hand_detector.pt"
    weights_path = Path(weights_path)

    selected_engine = engine.lower()
    if selected_engine == "auto":
        if weights_path.exists():
            selected_engine = "yolo"
            print(f"🚀 [Active Learning] Wykryto wytrenowany model YOLO ({weights_path}).")
            print("   Używam szybkiego modelu YOLO zamiast Grounding DINO!")
        else:
            selected_engine = "dino"
            print("🦖 [Bootstrap] Brak wytrenowanych wag YOLO. Używam modelu bazowego Grounding DINO.")

    yolo_model = None
    dino_model = None

    if selected_engine == "yolo":
        if not weights_path.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku wag YOLO: {weights_path}")
        from ultralytics import YOLO
        device_arg = "0" if device_name == "cuda" else device_name
        print(f"Ładowanie modelu YOLO ({weights_path}) na urządzeniu {device_arg.upper()}...")
        yolo_model = YOLO(str(weights_path))
    else:
        from autodistill_grounding_dino import GroundingDINO
        from autodistill.detection import CaptionOntology
        print(f"Ładowanie modelu Grounding DINO dla klasy '{target_class}'...")
        ontology = CaptionOntology({target_class: target_class})
        dino_model = GroundingDINO(ontology=ontology)

    # 5. Przetwarzanie tylko nowych obrazów
    print(f"\nRozpoczynanie detekcji dla {len(images_to_process)} nowych klatek...")
    for idx, img_path in enumerate(images_to_process, 1):
        unique_stem = get_unique_stem(img_path, input_dir)
        if idx % 25 == 0 or idx == len(images_to_process):
            print(f"[{idx}/{len(images_to_process)}] Analiza: {img_path.name}...")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [Błąd] Nie można wczytać obrazu {img_path}, pomijam.")
            continue

        h_img, w_img = img.shape[:2]
        yolo_lines = []

        if selected_engine == "yolo":
            device_arg = "0" if device_name == "cuda" else device_name
            results = yolo_model(img, conf=conf, device=device_arg, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                top_indices = np.argsort(confs)[::-1][:max_hands]
                xywhn = boxes.xywhn.cpu().numpy()[top_indices]
                for box in xywhn:
                    cx, cy, bw, bh = box
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        else:
            detections = dino_model.predict(str(img_path))
            detections = filter_top_detections(detections, max_detections=max_hands)
            if len(detections) > 0:
                for box in detections.xyxy:
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / (2.0 * w_img)
                    cy = (y1 + y2) / (2.0 * h_img)
                    bw = (x2 - x1) / float(w_img)
                    bh = (y2 - y1) / float(h_img)
                    yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        # Zapis pliku etykiety
        out_txt_path = train_labels_dir / f"{unique_stem}.txt"
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        # Kopiowanie/zapis obrazu do train/images
        out_img_path = train_images_dir / f"{unique_stem}.jpg"
        cv2.imwrite(str(out_img_path), img)

    # 6. Zapis klas i metadanych YOLO
    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write(f"{target_class}\n")

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"names:\n- {target_class}\nnc: 1\ntrain: {train_images_dir}\n")

    print(f"\n🎉 Zakończono detekcję! Zaktualizowano zbiór w: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detekcja rąk: Grounding DINO lub wytrenowany model YOLO (Active Learning)")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DIR), help="Katalog wejściowy ze zdjęciami")
    parser.add_argument("--output-dir", type=str, default=str(LABELED_DIR), help="Katalog wyjściowy na adnotacje YOLO")
    parser.add_argument("--max-hands", type=int, default=2, help="Maksymalna liczba rąk na obraz")
    parser.add_argument("--class-name", type=str, default="hand", help="Nazwa klasy")
    parser.add_argument("--force", action="store_true", help="Wymuszenie ponownej analizy wszystkich klatek od zera")
    parser.add_argument("--engine", type=str, choices=["auto", "yolo", "dino"], default="auto", help="Silnik detekcji: 'auto' (wybiera YOLO jeśli są wagi, inaczej DINO), 'yolo' lub 'dino'")
    parser.add_argument("--weights", type=str, default=None, help="Ścieżka do wytrenowanych wag YOLO (np. weights/yolo_hand_detector.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Próg pewności detekcji dla YOLO")
    args = parser.parse_args()

    run_detection(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_hands=args.max_hands,
        target_class=args.class_name,
        force_all=args.force,
        engine=args.engine,
        weights_path=Path(args.weights) if args.weights else None,
        conf=args.conf
    )