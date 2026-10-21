#!/usr/bin/env python3
"""
scripts/export_colab.py

Eksporter środowiska treningowego pod Google Colab (lub zewnętrzny serwer GPU).
Pakuje najpotrzebniejsze pliki (dataset, kod z `src/`, modele, loadery, metryki, notatniki)
do pojedynczego archiwum ZIP z wykluczeniem plików tymczasowych i cache.

Użycie:
    python3 scripts/export_colab.py --target cnn
    python3 scripts/export_colab.py --target yolo
    python3 scripts/export_colab.py --target all
"""

import argparse
import os
from pathlib import Path
import zipfile

# Ścieżka bazowa repozytorium (katalog wyżej od scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_NAMES = {
    "__pycache__",
    ".DS_Store",
    ".spyproject",
    ".ipynb_checkpoints",
    "thumbs.db",
}

EXCLUDED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".tmp",
    ".swp",
}


def should_include_file(file_path: Path) -> bool:
    """Sprawdza, czy plik nie jest śmieciem cache / systemowym."""
    for part in file_path.parts:
        if part in EXCLUDED_NAMES:
            return False
    if file_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return False
    return True


def add_dir_to_zip(
    zip_handle: zipfile.ZipFile,
    source_dir: Path,
    arc_base: str
) -> int:
    """Rekurencyjnie dodaje katalog do archiwum ZIP zachowując strukturę."""
    count = 0
    if not source_dir.exists():
        print(f"⚠️  Ostrzeżenie: Katalog {source_dir} nie istnieje. Pomijanie.")
        return 0

    for root, dirs, files in os.walk(source_dir):
        # Modyfikacja listy dirs w miejscu filtruje ignorowane foldery
        dirs[:] = [d for d in dirs if d not in EXCLUDED_NAMES]

        for f in files:
            full_path = Path(root) / f
            if not should_include_file(full_path):
                continue

            rel_path = full_path.relative_to(source_dir)
            arc_name = str(Path(arc_base) / rel_path)
            zip_handle.write(full_path, arcname=arc_name)
            count += 1

    return count


def add_file_to_zip(
    zip_handle: zipfile.ZipFile,
    source_file: Path,
    arc_name: str
) -> bool:
    """Dodaje pojedynczy plik do archiwum ZIP."""
    if not source_file.exists():
        print(f"⚠️  Ostrzeżenie: Plik {source_file} nie istnieje. Pomijanie.")
        return False
    zip_handle.write(source_file, arcname=arc_name)
    return True


def create_colab_readme(target: str) -> str:
    """Generuje instrukcję szybkiego startu dla Google Colab."""
    return f"""# 🚀 Instrukcja uruchomienia w Google Colab ({target.upper()})

Paczka zawiera wszystkie niezbędne pliki do treningu na GPU:
- Kod źródłowy (loadery, modele, trainer, metryki) w katalogu `src/`
- Przygotowany zbiór danych w `data/`
- Notatnik eksperymentalny w `notebooks/`

## Krok 1: Wgraj i rozpakuj paczkę w Colabie
W pierwszej komórce notatnika w Colabie wykonaj:

```python
# Jeśli wgrałeś plik ZIP do /content/ bezpośrednio:
!unzip -q colab_{target}.zip -d /content/

# Upewnij się, że katalog projektu jest w sys.path:
import sys
if "/content" not in sys.path:
    sys.path.append("/content")
```

## Krok 2: Sprawdź GPU
```python
!nvidia-smi
```

## Krok 3: Otwórz notatnik
- Przejdź do katalogu `notebooks/` i uruchamiaj kolejne komórki.
- Modele i wagi możesz zapisywać bezpośrednio do zamontowanego Google Drive.
"""


def create_colab_requirements() -> str:
    """Generuje minimalny plik zależności zoptymalizowany pod Google Colab."""
    return """# Zależności do treningu w Google Colab
# Uwaga: torch, torchvision, numpy, matplotlib, scikit-learn są już w Colabie!
seaborn>=0.12.0
torchmetrics>=1.0.0
ultralytics>=8.3.0
tensorboard>=2.14.0
"""


def main():
    parser = argparse.ArgumentParser(
        description="Eksport plików niezbędnych do treningu w chmurze (Google Colab)"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["cnn", "yolo", "all"],
        default="cnn",
        help="Cel eksportu: 'cnn' (gesty), 'yolo' (detektor dłoni), 'all' (oba zbiory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ścieżka do wyjściowego pliku ZIP (domyślnie: colab_<target>.zip w głównym katalogu)"
    )

    args = parser.parse_args()

    default_names = {
        "cnn": "colab_gesture_cnn.zip",
        "yolo": "colab_yolo_detector.zip",
        "all": "colab_thesis_full.zip",
    }
    output_filename = args.output or default_names[args.target]
    output_zip_path = PROJECT_ROOT / output_filename

    print(f"\n=======================================================")
    print(f"📦 Przygotowywanie paczki Google Colab [Target: {args.target.upper()}]")
    print(f"   Wyjście: {output_zip_path.name}")
    print(f"=======================================================\n")

    files_packed = 0

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Kod źródłowy (src/)
        src_dir = PROJECT_ROOT / "src"
        if src_dir.exists():
            n = add_dir_to_zip(zf, src_dir, "src")
            print(f"  ✅ [src/] Dodano {n} plików Pythona (modele, loadery, trainer, metryki)")
            files_packed += n
        else:
            print("  ❌ Brak katalogu src/!")

        # 2. Datasets
        if args.target in ["cnn", "all"]:
            cnn_dir = PROJECT_ROOT / "data" / "cnn_dataset"
            if cnn_dir.exists():
                n = add_dir_to_zip(zf, cnn_dir, "data/cnn_dataset")
                print(f"  ✅ [data/cnn_dataset] Dodano {n} plików (wycinki gestów + metadata)")
                files_packed += n
            else:
                print(f"  ⚠️  Brak katalogu {cnn_dir}! Uruchom najpierw `make prepare-yolo`")

        if args.target in ["yolo", "all"]:
            yolo_dir = PROJECT_ROOT / "data" / "yolo_dataset"
            if yolo_dir.exists():
                n = add_dir_to_zip(zf, yolo_dir, "data/yolo_dataset")
                print(f"  ✅ [data/yolo_dataset] Dodano {n} plików (obrazy + etykiety YOLO)")
                files_packed += n
            else:
                print(f"  ⚠️  Brak katalogu {yolo_dir}! Uruchom najpierw `make prepare-yolo`")

        # 3. Notatniki
        if args.target in ["cnn", "all"]:
            nb_cnn = PROJECT_ROOT / "notebooks" / "experiment_gesture_cnn.ipynb"
            if add_file_to_zip(zf, nb_cnn, "notebooks/experiment_gesture_cnn.ipynb"):
                print("  ✅ [notebooks] Dodano experiment_gesture_cnn.ipynb")
                files_packed += 1

        if args.target in ["yolo", "all"]:
            nb_yolo = PROJECT_ROOT / "notebooks" / "train_yolo_colab.ipynb"
            if add_file_to_zip(zf, nb_yolo, "notebooks/train_yolo_colab.ipynb"):
                print("  ✅ [notebooks] Dodano train_yolo_colab.ipynb")
                files_packed += 1

        # 4. Instrukcja README i zależności Colab
        zf.writestr("README_COLAB.md", create_colab_readme(args.target))
        zf.writestr("requirements-colab.txt", create_colab_requirements())
        print("  ✅ Dodano README_COLAB.md i requirements-colab.txt")
        files_packed += 2

    size_mb = output_zip_path.stat().st_size / (1024 * 1024)
    print("\n-------------------------------------------------------")
    print(f"🎉 Sukces! Spakowano łącznie {files_packed} plików.")
    print(f"📁 Wynikowy plik: {output_zip_path} ({size_mb:.2f} MB)")
    print("-------------------------------------------------------")
    print("Szybki import w Colabie:")
    print("  1. Wgraj plik na Google Drive lub przeciągnij do panelu Colaba.")
    print(f"  2. W komórce Colaba: !unzip -q {output_zip_path.name} -d /content/")
    print("  3. Gotowe do treningu!")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()
