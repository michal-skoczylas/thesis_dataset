import os
import sys
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

try:
    from config import WEIGHTS_DIR, DATA_DIR
except ImportError:
    WEIGHTS_DIR = BASE_DIR / "weights"
    DATA_DIR = BASE_DIR / "data"

DEFAULT_DATA_YAML = DATA_DIR / "yolo_dataset" / "data.yaml"


def detect_best_device(requested_device: str = "auto") -> str:
    """Wykrywa optymalne urządzenie obliczeniowe dla YOLO."""
    if requested_device and requested_device != "auto":
        return requested_device

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 [Hardware] Wykryto NVIDIA GPU (CUDA): {gpu_name}")
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🍎 [Hardware] Wykryto Apple Silicon GPU (MPS/Metal)")
            return "mps"
    except Exception as e:
        print(f"⚠️  [Hardware] Błąd detekcji akceleratora: {e}")

    print("🐢 [Hardware] Brak akceleracji GPU – używam procesora (CPU)")
    return "cpu"


def train_yolo(
    data_yaml: Path = DEFAULT_DATA_YAML,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "auto",
    patience: int = 25,
    lr0: float = 0.01,
    workers: int = 4,
    project_dir: Path = None,
    run_name: str = "hand_detector",
    save_best_to_weights: bool = True
):
    """
    Uruchamia trening modelu YOLO dla Etapu 1 (detekcja rąk).
    Kompatybilny z PC (NVIDIA CUDA), Mac (MPS) oraz Google Colab.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ BŁĄD: Pakiet 'ultralytics' nie jest zainstalowany.")
        print("   Zainstaluj go poleceniem: pip install ultralytics")
        sys.exit(1)

    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.exists():
        print(f"\n❌ BŁĄD: Nie znaleziono pliku konfiguracji datasetu: {data_yaml}")
        print("   Wygeneruj zbiór danych uruchamiając najpierw:")
        print("   python3 scripts/prepare_yolo_dataset.py")
        sys.exit(1)

    # Ustal katalog projektów
    if project_dir is None:
        project_dir = BASE_DIR / "runs" / "detect"
    project_dir = Path(project_dir).resolve()

    selected_device = detect_best_device(device)

    print("\n" + "=" * 65)
    print("🚁 THESIS: ROZPOCZYNANIE TRENINGU YOLO (ETAP 1: DETEKTOR DŁONI)")
    print("=" * 65)
    print(f"📦 Model bazowy       : {model_name}")
    print(f"📄 Konfiguracja danych: {data_yaml}")
    print(f"⏱️  Liczba epok        : {epochs}")
    print(f"📦 Rozmiar batcha     : {batch_size}")
    print(f"📐 Rozdzielczość (px) : {imgsz}x{imgsz}")
    print(f"⚡ Urządzenie         : {selected_device}")
    print(f"🛑 Wczesne zatrzymanie: patience={patience} epok")
    print(f"📁 Wyniki lądują w    : {project_dir / run_name}")
    print("=" * 65 + "\n")

    # Inicjalizacja modelu
    model = YOLO(model_name)

    # Uruchomienie treningu
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=selected_device,
        patience=patience,
        lr0=lr0,
        workers=workers,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("🎉 TRENING ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 65)

    # Ścieżka do najlepszych wag
    best_weights_path = project_dir / run_name / "weights" / "best.pt"
    if best_weights_path.exists():
        print(f"🏆 Najlepsze wagi modelu (best.pt): {best_weights_path}")

        if save_best_to_weights:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            target_weights = WEIGHTS_DIR / "yolo_hand_detector.pt"
            shutil.copy2(best_weights_path, target_weights)
            print(f"💾 Skopiowano wagi produkcyjne do: {target_weights}")

    # Ewaluacja na zbiorze walidacyjnym
    print("\n📊 Przeprowadzanie końcowej ewaluacji na zbiorze walidacyjnym (val)...")
    val_metrics = model.val(data=str(data_yaml), device=selected_device)

    print("\n📈 Podsumowanie metryk detekcji dłoni (mAP):")
    try:
        map50 = val_metrics.box.map50
        map50_95 = val_metrics.box.map
        precision = val_metrics.box.mp
        recall = val_metrics.box.mr

        print(f"   • mAP@0.50      : {map50:.4f} ({map50*100:.2f}%)")
        print(f"   • mAP@0.50:0.95 : {map50_95:.4f} ({map50_95*100:.2f}%)")
        print(f"   • Precyzja (P)  : {precision:.4f}")
        print(f"   • Czułość (R)   : {recall:.4f}")
    except Exception as e:
        print(f"   (Szczegółowe metryki zapisane w katalogu eksperymentu: {project_dir / run_name})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening YOLO Nano (Etap 1: Detektor dłoni)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Ścieżka do pliku data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Model bazowy YOLO (np. yolo11n.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Liczba epok treningu")
    parser.add_argument("--batch", type=int, default=16, help="Rozmiar batcha")
    parser.add_argument("--imgsz", type=int, default=640, help="Rozmiar obrazu (np. 640)")
    parser.add_argument("--device", type=str, default="auto", help="Urządzenie: 'auto', '0', 'cuda:0', 'mps', 'cpu'")
    parser.add_argument("--patience", type=int, default=25, help="Wczesne zatrzymanie (patience)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Początkowy learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Liczba workerów DataLoader")
    parser.add_argument("--project", type=str, default=None, help="Katalog wyników treningu")
    parser.add_argument("--name", type=str, default="hand_detector", help="Nazwa przebiegu")

    args = parser.parse_args()

    train_yolo(
        data_yaml=Path(args.data),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        project_dir=Path(args.project) if args.project else None,
        run_name=args.name
    )
