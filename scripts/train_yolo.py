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


def load_params() -> dict:
    """Ładuje konfigurację z params.yaml."""
    params_path = BASE_DIR / "params.yaml"
    if params_path.exists():
        try:
            import yaml
            with open(params_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def detect_native_imgsz(data_yaml_path: Path) -> int:
    """Wykrywa natywny rozmiar zdjęć w zbiorze (np. 320x320 z kamery drona)."""
    try:
        train_dir = data_yaml_path.parent / "images" / "train"
        if train_dir.exists():
            import cv2
            first_img = next(train_dir.glob("*.*"), None)
            if first_img:
                im = cv2.imread(str(first_img))
                if im is not None:
                    max_dim = max(im.shape[:2])
                    # Zaokrąglenie w górę do wielokrotności 32 (wymóg architektury YOLO)
                    return ((max_dim + 31) // 32) * 32
    except Exception:
        pass
    return 320


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
    model_name: str = "yolo26n.pt",
    epochs: int = None,
    batch_size: int = None,
    imgsz: int = None,
    device: str = "auto",
    patience: int = None,
    lr0: float = None,
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

    # Upewnij się, że pole 'path' w data.yaml wskazuje na rzeczywisty katalog z danymi
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as yf:
            ycfg = yaml.safe_load(yf) or {}
        ycfg["path"] = str(data_yaml.parent.resolve())
        with open(data_yaml, "w", encoding="utf-8") as yf:
            yaml.dump(ycfg, yf, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"⚠️  Nie udało się zaktualizować path w data.yaml: {e}")

    # Wczytaj konfigurację z params.yaml
    params = load_params()
    yolo_params = params.get("train_yolo", {})
    augment_params = yolo_params.get("augment", {})

    # Wartości domyślne (CLI > params.yaml > fallback)
    if epochs is None:
        epochs = yolo_params.get("epochs", 100)
    if batch_size is None:
        batch_size = yolo_params.get("batch_size", 16)
    if patience is None:
        patience = yolo_params.get("patience", 25)
    if lr0 is None:
        lr0 = yolo_params.get("lr0", 0.01)

    # Rozdzielczość: automatycznie z klatek drona (320px) lub z params.yaml/CLI
    if imgsz is None:
        imgsz = yolo_params.get("imgsz", detect_native_imgsz(data_yaml))

    if project_dir is None:
        project_dir = BASE_DIR / "runs" / "detect"
    project_dir = Path(project_dir).resolve()

    selected_device = detect_best_device(device)

    # Parametry augmentacji w locie pod specyfikę widoku z drona:
    aug_cfg = {
        "degrees": augment_params.get("degrees", 15.0),
        "scale": augment_params.get("scale", 0.5),
        "shear": augment_params.get("shear", 2.0),
        "perspective": augment_params.get("perspective", 0.0005),
        "fliplr": augment_params.get("fliplr", 0.5),
        "flipud": augment_params.get("flipud", 0.0),
        "mosaic": augment_params.get("mosaic", 1.0),
        "mixup": augment_params.get("mixup", 0.1),
        "hsv_v": augment_params.get("hsv_v", 0.4),
        "hsv_s": augment_params.get("hsv_s", 0.7),
    }

    print("\n" + "=" * 68)
    print("🚁 THESIS: ROZPOCZYNANIE TRENINGU YOLO (ETAP 1: DETEKTOR DŁONI)")
    print("=" * 68)
    print(f"📦 Model bazowy          : {model_name}")
    print(f"📄 Konfiguracja danych   : {data_yaml}")
    print(f"⏱️  Liczba epok           : {epochs}")
    print(f"📦 Rozmiar batcha        : {batch_size}")
    print(f"📐 Rozdzielczość (px)    : {imgsz}x{imgsz} (dopasowana do klatek z drona)")
    print(f"⚡ Urządzenie            : {selected_device}")
    print(f"🛑 Wczesne zatrzymanie   : patience={patience} epok")
    print(f"🎨 Augmentacje w locie   : rot={aug_cfg['degrees']}°, scale=±{aug_cfg['scale']}, "
          f"mosaic={aug_cfg['mosaic']}, mixup={aug_cfg['mixup']}")
    print(f"📁 Wyniki lądują w       : {project_dir / run_name}")
    print("=" * 68 + "\n")

    # Inicjalizacja modelu
    model = YOLO(model_name)

    # Uruchomienie treningu z pełną augmentacją w locie
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
        verbose=True,
        **aug_cfg
    )

    print("\n" + "=" * 68)
    print("🎉 TRENING ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 68)

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
    val_metrics = model.val(data=str(data_yaml), device=selected_device, imgsz=imgsz)

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
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Model bazowy YOLO (np. yolo26n.pt, yolo11n.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Liczba epok treningu (domyślnie z params.yaml: 100)")
    parser.add_argument("--batch", type=int, default=None, help="Rozmiar batcha (domyślnie z params.yaml: 16)")
    parser.add_argument("--imgsz", type=int, default=None, help="Rozmiar obrazu (domyślnie dopasowany z drona: 320)")
    parser.add_argument("--device", type=str, default="auto", help="Urządzenie: 'auto', '0', 'cuda:0', 'mps', 'cpu'")
    parser.add_argument("--patience", type=int, default=None, help="Wczesne zatrzymanie patience (domyślnie: 25)")
    parser.add_argument("--lr0", type=float, default=None, help="Początkowy learning rate")
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
