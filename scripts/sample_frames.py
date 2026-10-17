import os
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2

from config import SOURCE_FRAMES_DIR, RAW_DIR, BASE_DIR
from convert_grayscale import detect_and_convert_image

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def natural_sort_key(path: Path):
    """
    Klucz do naturalnego sortowania (np. 00040.jpg, 00080.jpg, 00120.jpg).
    Chroni przed błędnym sortowaniem leksykalnym.
    """
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_sample_params() -> dict:
    """Odczytuje domyślne parametry próbkowania z params.yaml."""
    params_file = BASE_DIR / "params.yaml"
    defaults = {"stride": 10, "camera": "camera_rgb", "to_gray": False}
    if params_file.exists():
        try:
            import yaml
            with open(params_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data and "sample" in data:
                    defaults["stride"] = int(data["sample"].get("stride", 10))
                    defaults["camera"] = str(data["sample"].get("camera", "camera_rgb"))
                    defaults["to_gray"] = bool(data["sample"].get("to_gray", False))
        except Exception:
            pass
    return defaults


def sample_directory(
    input_dir: Path,
    output_dir: Path,
    stride: int = 10,
    camera: Optional[str] = "camera_rgb",
    strip_camera_dir: bool = True,
    to_gray: bool = False,
    offset: int = 0,
    use_symlink: bool = False,
    clear_output: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Wybiera co N-tą klatkę (stride) z każdej sekwencji w katalogu wejściowym
    z opcjonalnym filtrowaniem konkretnej kamery (np. camera_rgb lub camera_raw).
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Katalog wejściowy nie istnieje: {input_dir}")

    # Normalizacja filtru kamery ('all' lub '' oznacza brak filtra)
    if camera in ("all", "", "none", "None"):
        camera = None

    # Znajdź wszystkie pliki graficzne pogrupowane po folderach (sekwencjach)
    images_by_folder: Dict[Path, List[Path]] = {}
    total_found = 0
    total_skipped_camera = 0

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)

        # Filtrowanie po wybranej kamerze (np. camera_rgb vs camera_raw)
        if camera:
            # Sprawdź czy ten katalog to folder kamery lub czy camera jest w ścieżce
            is_camera_folder = (root_path.name.lower() == camera.lower()) or (camera.lower() in [p.lower() for p in root_path.parts])
            if not is_camera_folder:
                # Jeśli folder ma zdjęcia z innej kamery, zlicz pominięte
                other_cam_files = [f for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS]
                total_skipped_camera += len(other_cam_files)
                continue

        valid_files = [
            root_path / f for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS
        ]
        if valid_files:
            valid_files.sort(key=natural_sort_key)
            images_by_folder[root_path] = valid_files
            total_found += len(valid_files)

    if total_found == 0:
        print(f"[Sampling] Brak plików graficznych pasujących do kryteriów w: {input_dir}")
        if camera:
            print(f"           (Filtr kamery był ustawiony na: '{camera}')")
        return {"total_found": 0, "total_selected": 0}

    print("=" * 70)
    print("  PRÓBKOWANIE KLATEK Z DRONA (FRAME SUBSAMPLING)")
    print("=" * 70)
    print(f"Katalog wejściowy:     {input_dir}")
    print(f"Katalog wyjściowy:     {output_dir}")
    print(f"Wybrana kamera:        {camera.upper() if camera else 'WSZYSTKIE FOLDERY'}")
    print(f"Krok próbkowania:      co {stride}. klatka (początkowy offset: {offset})")
    print(f"Czyszczenie ścieżek:   {'Usuwanie podfolderu kamery z wyjścia' if (camera and strip_camera_dir) else 'Zachowanie pełnej ścieżki'}")
    print(f"Konwersja czarno-biała:{'TAK (usuwanie kropek / RGB -> mono)' if to_gray else 'NIE (kopiowanie bez zmian)'}")
    print(f"Tryb zapisu:           {'Dowiązania symboliczne (symlink)' if use_symlink else 'Kopiowanie plików'}")
    if dry_run:
        print(f"Tryb DRY-RUN:          AKTYWNY (symulacja bez zapisu na dysk)")
    print("-" * 70)

    if clear_output and not dry_run and output_dir.exists():
        print(f"[Sampling] Czyszczenie katalogu wyjściowego: {output_dir}")
        for item in output_dir.iterdir():
            if item.name == ".gitignore":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    total_selected = 0
    sequences_count = len(images_by_folder)

    for seq_idx, (folder_path, files) in enumerate(images_by_folder.items(), start=1):
        rel_folder = folder_path.relative_to(input_dir)

        # Usunięcie zbędnego poziomu podfolderu kamery (np. /camera_rgb) z folderu wyjściowego
        if camera and strip_camera_dir:
            clean_parts = [p for p in rel_folder.parts if p.lower() != camera.lower()]
            target_folder = output_dir / Path(*clean_parts) if clean_parts else output_dir
        else:
            target_folder = output_dir / rel_folder

        selected_files = files[offset::stride]
        total_selected += len(selected_files)

        seq_label = str(rel_folder) if str(rel_folder) != "." else "(główny katalog)"
        print(f"[{seq_idx}/{sequences_count}] Sekwencja: {seq_label}")
        print(f"    Znaleziono: {len(files)} klatek -> Wybrano: {len(selected_files)} klatek")

        if not dry_run:
            target_folder.mkdir(parents=True, exist_ok=True)
            for src_file in selected_files:
                dest_file = target_folder / src_file.name

                if dest_file.exists() or dest_file.is_symlink():
                    dest_file.unlink()

                if to_gray:
                    # Konwertuj i usuń kropki/zafarb w locie
                    converted, _ = detect_and_convert_image(src_file, output_channels=3)
                    cv2.imwrite(str(dest_file), converted)
                elif use_symlink:
                    os.symlink(src_file, dest_file)
                else:
                    shutil.copy2(src_file, dest_file)

    reduction = ((total_found - total_selected) / total_found * 100) if total_found > 0 else 0
    print("-" * 70)
    print("Podsumowanie:")
    print(f"  Przeszukano sekwencji:          {sequences_count}")
    print(f"  Klatek z wybranej kamery:       {total_found}")
    if total_skipped_camera > 0:
        print(f"  Pominięto z innej kamery:       {total_skipped_camera} klatek")
    print(f"  Wyselekcjonowano do datasetu:   {total_selected} klatek")
    print(f"  Redukcja rozmiaru:              -{reduction:.1f}%")
    print("=" * 70)

    return {
        "total_found": total_found,
        "total_selected": total_selected,
        "sequences_count": sequences_count,
        "reduction_pct": reduction,
    }


def main():
    defaults = load_sample_params()

    parser = argparse.ArgumentParser(
        description="Inteligentne próbkowanie klatek wideo z drona z wyborem kamery (camera_rgb / camera_raw)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(SOURCE_FRAMES_DIR),
        help=f"Katalog wejściowy ze wszystkimi klatkami (domyślnie: {SOURCE_FRAMES_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(RAW_DIR),
        help=f"Katalog docelowy na wyselekcjonowane klatki (domyślnie: {RAW_DIR})"
    )
    parser.add_argument(
        "--stride", "-s",
        type=int,
        default=defaults["stride"],
        help=f"Krok próbkowania (co N-ta klatka). Domyślnie z params.yaml: {defaults['stride']}"
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=defaults["camera"],
        choices=["camera_rgb", "camera_raw", "all"],
        help=f"Wybór podfolderu kamery do pobrania klatek (domyślnie: {defaults['camera']})"
    )
    parser.add_argument(
        "--keep-camera-dir",
        action="store_true",
        help="Zachowaj podfolder kamery w ścieżce wyjściowej (domyślnie jest usuwany dla czystości)"
    )
    parser.add_argument(
        "--to-gray",
        action="store_true",
        default=defaults["to_gray"],
        help="Automatycznie konwertuj wybrane klatki na czysty czarno-biały (usuwa kropki i zafarb)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Indeks startowej klatki (domyślnie: 0)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Utwórz dowiązania symboliczne zamiast fizycznego kopiowania plików (oszczędza dysk)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wyczyść katalog wyjściowy przed skopiowaniem nowych klatek"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko symulacja (pokaż ile klatek zostanie wybranych bez zapisywania)"
    )

    args = parser.parse_args()

    sample_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        stride=args.stride,
        camera=args.camera,
        strip_camera_dir=not args.keep_camera_dir,
        to_gray=args.to_gray,
        offset=args.offset,
        use_symlink=args.symlink,
        clear_output=args.clear,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
