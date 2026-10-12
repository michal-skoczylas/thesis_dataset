import os
import json
import argparse
from pathlib import Path
import requests
import cv2

from config import (
    DATA_DIR,
    ANNOTATIONS_DIR,
    PROCESSED_DIR,
    LABEL_STUDIO_URL,
    PROJECT_ID,
    get_auth_headers,
)


def export_annotations_from_ls(project_id: int = PROJECT_ID, output_file: Path = None) -> list:
    """Pobiera zatwierdzone przez człowieka adnotacje z Label Studio API i zapisuje do JSON."""
    if output_file is None:
        output_file = ANNOTATIONS_DIR / "dataset.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    headers = get_auth_headers()

    print(f"Pobieranie zatwierdzonych zadań z Label Studio (Projekt {project_id})...")
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export?exportType=JSON"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    tasks = r.json()
    print(f"Znaleziono {len(tasks)} oznaczonych zadań.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"Zapisano metadane adnotacji do pliku: {output_file}")
    return tasks


def generate_crops_from_annotations(json_path: Path = None, output_dir: Path = PROCESSED_DIR, padding: int = 0) -> int:
    """Wczytuje dataset.json i wycina obszary dłoni za pomocą OpenCV do katalogu processed."""
    if json_path is None:
        json_path = ANNOTATIONS_DIR / "dataset.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku adnotacji: {json_path}. Wykonaj najpierw eksport.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"Generowanie cropów dla {len(tasks)} wpisów z {json_path}...")
    crops_count = 0

    for task in tasks:
        img_url = task.get("data", {}).get("image", "")
        if not img_url:
            continue

        # Wyciągnięcie relatywnej ścieżki (np. labeled/valid/images/00480.jpg)
        rel_img_path = img_url.split("?d=")[-1]
        abs_img_path = DATA_DIR / rel_img_path

        if not abs_img_path.exists():
            # Fallback wyszukiwania pliku w całym DATA_DIR jeśli struktura podkatalogów uległa zmianie
            found = list(DATA_DIR.rglob(abs_img_path.name))
            if found:
                abs_img_path = found[0]
            else:
                continue

        img = cv2.imread(str(abs_img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue

                val = result.get("value", {})
                x = int(val["x"] * w_img / 100)
                y = int(val["y"] * h_img / 100)
                w = int(val["width"] * w_img / 100)
                h = int(val["height"] * h_img / 100)

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(w_img, x + w + padding)
                y2 = min(h_img, y + h + padding)

                crop_img = img[y1:y2, x1:x2]
                if crop_img.size != 0:
                    base_name = abs_img_path.stem
                    box_id = result.get("id", "0")
                    crop_filename = f"{base_name}_{box_id}.jpg"
                    crop_path = output_dir / crop_filename

                    cv2.imwrite(str(crop_path), crop_img)
                    crops_count += 1

    print(f"Zakończono! Zapisano {crops_count} wyciętych dłoni w: {output_dir}")
    return crops_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eksport adnotacji z Label Studio i wycinanie dłoni (cropy)")
    parser.add_argument("--export-only", action="store_true", help="Tylko eksport do JSON z Label Studio")
    parser.add_argument("--crop-only", action="store_true", help="Tylko wycinanie cropów z zapisanego dataset.json")
    parser.add_argument("--project-id", type=int, default=PROJECT_ID, help="ID projektu Label Studio")
    parser.add_argument("--json-file", type=str, default=str(ANNOTATIONS_DIR / "dataset.json"), help="Ścieżka do dataset.json")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR), help="Katalog wyjściowy cropów")
    parser.add_argument("--padding", type=int, default=0, help="Dodatkowy margines w pikselach dla cropa")
    args = parser.parse_args()

    json_file = Path(args.json_file)
    output_dir = Path(args.output_dir)

    if args.export_only:
        export_annotations_from_ls(project_id=args.project_id, output_file=json_file)
    elif args.crop_only:
        generate_crops_from_annotations(json_path=json_file, output_dir=output_dir, padding=args.padding)
    else:
        export_annotations_from_ls(project_id=args.project_id, output_file=json_file)
        generate_crops_from_annotations(json_path=json_file, output_dir=output_dir, padding=args.padding)