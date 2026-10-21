#!/usr/bin/env python3
"""
Skrypt CLI do szybkiego i czytelnego uruchamiania treningu PyTorch Lightning.

Przykłady użycia:
    # 1. Podstawowy trening GestureCNN_v1:
    python scripts/train_lightning.py --model GestureCNN_v1 --epochs 40 --batch-size 32

    # 2. Trening architektury rezydualnej z wczesnym zatrzymaniem i eksportem ONNX:
    python scripts/train_lightning.py --model GestureCNN_Residual --epochs 50 --patience 10 --export --eval

    # 3. Sprawdzenie dostępnych architektur:
    python scripts/train_lightning.py --list-models
"""

import argparse
import sys
from pathlib import Path

# Upewniamy się, że katalog główny projektu jest w sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lightning_module import MODEL_REGISTRY
from src.lightning_trainer import train_lightning
from src.export import export_model
from src.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Trening modeli klasyfikacji gestów dłoni w PyTorch Lightning")
    parser.add_argument(
        "--model",
        type=str,
        default="GestureCNN_v1",
        help=f"Nazwa architektury modelu. Dostępne: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Liczba epok treningu (domyślnie 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha (domyślnie 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Współczynnik uczenia Learning Rate (domyślnie 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Waga regularyzacji L2 (domyślnie 1e-4)")
    parser.add_argument("--img-size", type=int, default=64, help="Rozdzielczość wejściowa obrazu NxN (domyślnie 64)")
    parser.add_argument("--patience", type=int, default=None, help="Liczba epok dla EarlyStopping (opcjonalne)")
    parser.add_argument("--name", type=str, default=None, help="Nazwa eksperymentu (domyślnie generowana z modelu)")
    parser.add_argument("--num-workers", type=int, default=2, help="Liczba wątków DataLoader (domyślnie 2)")
    parser.add_argument("--save-dir", type=str, default="weights", help="Katalog zapisu wag (domyślnie weights/)")
    parser.add_argument("--log-dir", type=str, default="runs/lightning", help="Katalog logów TensorBoard")
    parser.add_argument("--export", action="store_true", help="Automatyczny eksport do TorchScript i ONNX po treningu")
    parser.add_argument("--eval", action="store_true", help="Automatyczna ewaluacja z macierzą pomyłek po treningu")
    parser.add_argument("--list-models", action="store_true", help="Wyświetla listę zarejestrowanych modeli i kończy")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\n📋 Dostępne modele w rejestrze PyTorch Lightning:")
        for key in sorted(set(MODEL_REGISTRY.keys())):
            cls = MODEL_REGISTRY[key]
            print(f"  • {key:22} -> {cls.__name__}")
        print()
        return

    exp_name = args.name or f"pl_{args.model.lower()}"

    print("=" * 70)
    print("⚡ [PyTorch Lightning Pipeline] Rozpoczynanie sesji treningowej")
    print(f"   Model       : {args.model}")
    print(f"   Epoki       : {args.epochs}")
    print(f"   Batch size  : {args.batch_size}")
    print(f"   LR          : {args.lr}")
    print(f"   Img size    : {args.img_size}x{args.img_size}")
    print(f"   Eksperyment : {exp_name}")
    print("=" * 70)

    pl_model, trainer, dm = train_lightning(
        model=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        experiment_name=exp_name,
        early_stopping_patience=args.patience,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        num_workers=args.num_workers,
    )

    print("\n✅ Trening zakończony!")

    # Opcjonalna ewaluacja
    if args.eval:
        print("\n📊 Uruchamianie raportu ewaluacji...")
        val_loader = dm.val_dataloader()
        report_png = Path("reports") / f"cm_{exp_name}.png"
        report_png.parent.mkdir(parents=True, exist_ok=True)
        evaluate_model(
            model=pl_model.model,
            val_loader=val_loader,
            class_names=dm.class_names,
            title=f"Lightning {args.model}",
            save_path=str(report_png),
        )

    # Opcjonalny eksport do formatów drona
    if args.export:
        print("\n📦 Eksportowanie modelu produkcyjnego (TorchScript & ONNX)...")
        export_model(
            model=pl_model.model,
            output_dir=args.save_dir,
            model_name=f"{exp_name}_exported",
            input_size=(1, 3, args.img_size, args.img_size),
        )


if __name__ == "__main__":
    main()
