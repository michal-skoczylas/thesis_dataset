import os
import argparse
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np

from config import RAW_DIR

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def detect_and_convert_image(
    img_path: Path,
    output_channels: int = 3,
    bayer_pattern: int = cv2.COLOR_BayerRG2GRAY
) -> Tuple[np.ndarray, str]:
    """
    Automatycznie wykrywa typ obrazu:
      1. RGB_COLOR: obraz z kamery RGB (zafarb barwny) -> konwersja na czarno-biały (COLOR_BGR2GRAY).
      2. RAW_BAYER: surowy sensor z mozaiką Bayera (czarne kropki / raster) -> debayering do czystego czarno-białego (COLOR_BayerRG2GRAY).
      3. ALREADY_GRAY: czysty obraz monochromatyczny -> zachowanie bez zniekształceń.

    Zwraca: (obraz_wynikowy, typ_obrazu)
    """
    # Wczytaj jako BGR
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Nie udało się wczytać pliku: {img_path}")

    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    color_diff = np.max(np.abs(b.astype(int) - g.astype(int))) + np.max(np.abs(g.astype(int) - r.astype(int)))

    path_lower = str(img_path).lower()

    # 1. Sprawdzenie, czy to obraz kolorowy RGB
    if color_diff > 15 or "camera_rgb" in path_lower:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        detected_type = "RGB_COLOR"

    else:
        # 2. Obraz ma identyczne kanały lub jest surowy. Sprawdź mozaikę Bayera (kropki).
        raw_gray = b
        even_rows = raw_gray[0::2, :]
        odd_rows = raw_gray[1::2, :]
        min_h = min(even_rows.shape[0], odd_rows.shape[0])
        alternating_diff = np.abs(even_rows[:min_h].astype(float) - odd_rows[:min_h].astype(float)).mean()

        if "camera_raw" in path_lower or alternating_diff > 18:
            # Wykryto mozaikę Bayera (czarne kropki) -> debayering usuwa kropki i przywraca gładki obraz
            gray = cv2.cvtColor(raw_gray, bayer_pattern)
            detected_type = "RAW_BAYER (usunięto kropki)"
        else:
            gray = raw_gray
            detected_type = "ALREADY_GRAY"

    # Wyjście: 3 kanały (R=G=B) dla 100% kompatybilności z sieciami CNN/YOLO/DINO lub 1 kanał
    if output_channels == 3:
        output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        output_img = gray

    return output_img, detected_type


def convert_dataset_to_grayscale(
    input_dir: Path,
    output_dir: Path = None,
    in_place: bool = False,
    output_channels: int = 3
) -> Dict[str, int]:
    """
    Przetwarza cały zbiór danych, wykrywając format każdego zdjęcia i konwertując do czystego czarno-białego.
    Zachowuje strukturę katalogów i podfolderów.
    """
    input_dir = Path(input_dir).resolve()
    if in_place:
        output_dir = input_dir
    else:
        if output_dir is None:
            raise ValueError("Musisz podać output_dir lub włączyć flagę --in-place")
        output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Katalog wejściowy nie istnieje: {input_dir}")

    # Znajdź wszystkie pliki graficzne
    all_images = []
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                all_images.append(root_path / f)

    total_images = len(all_images)
    if total_images == 0:
        print(f"[Grayscale Converter] Nie znaleziono obrazów w: {input_dir}")
        return {}

    print("=" * 68)
    print("  KONWERSJA DO CZYSTEGO CZARNO-BIAŁEGO (DETEKCJA RGB / RAW BAYER)")
    print("=" * 68)
    print(f"Katalog wejściowy:   {input_dir}")
    print(f"Katalog wyjściowy:   {output_dir} {'(W MIEJSCU / IN-PLACE)' if in_place else ''}")
    print(f"Liczba obrazów:      {total_images}")
    print(f"Kanały wyjściowe:    {output_channels} {'(R=G=B, kompatybilne z YOLO/DINO)' if output_channels == 3 else '(1 kanał grayscale)'}")
    print("-" * 68)

    stats = {
        "RGB_COLOR": 0,
        "RAW_BAYER (usunięto kropki)": 0,
        "ALREADY_GRAY": 0,
        "ERRORS": 0
    }

    for idx, img_path in enumerate(all_images, start=1):
        rel_path = img_path.relative_to(input_dir)
        target_path = output_dir / rel_path

        try:
            converted, detected_type = detect_and_convert_image(
                img_path,
                output_channels=output_channels
            )
            stats[detected_type] = stats.get(detected_type, 0) + 1

            target_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_path), converted)

            if idx % 100 == 0 or idx == total_images:
                print(f"  [Postęp: {idx}/{total_images}] Ostatni: {rel_path} -> {detected_type}")

        except Exception as e:
            print(f"  [Błąd] Plik {rel_path}: {e}")
            stats["ERRORS"] += 1

    print("-" * 68)
    print("Podsumowanie konwersji:")
    print(f"  Łącznie przetworzono:        {total_images}")
    print(f"  Obrazy RGB -> czarno-białe:  {stats['RGB_COLOR']}")
    print(f"  Obrazy RAW z kropkami -> usunięto kropki (debayered): {stats['RAW_BAYER (usunięto kropki)']}")
    print(f"  Obrazy już czarno-białe:     {stats['ALREADY_GRAY']}")
    if stats["ERRORS"] > 0:
        print(f"  Błędy:                       {stats['ERRORS']}")
    print("=" * 68)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Automatyczna detekcja formatu (RGB vs RAW z kropkami) i konwersja na czysty czarno-biały"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(RAW_DIR),
        help=f"Katalog wejściowy z obrazami (domyślnie: {RAW_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Katalog wyjściowy na przekonwertowane obrazy"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Nadpisz pliki bezpośrednio w katalogu wejściowym (in-place)"
    )
    parser.add_argument(
        "--channels", "-c",
        type=int,
        choices=[1, 3],
        default=3,
        help="Liczba kanałów wyjściowych: 3 (R=G=B dla kompatybilności z YOLO/DINO) lub 1"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not args.in_place and output_path is None:
        # Jeśli nie podano outputu ani flagi in-place, domyślnie stwórz katalog _grayscale
        output_path = input_path.parent / f"{input_path.name}_grayscale"
        print(f"[Info] Nie podano --output ani --in-place. Wyniki zostaną zapisane w: {output_path}")

    convert_dataset_to_grayscale(
        input_dir=input_path,
        output_dir=output_path,
        in_place=args.in_place,
        output_channels=args.channels
    )


if __name__ == "__main__":
    main()
