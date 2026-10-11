import os
import json
import glob
import requests

PROJECT_ID = "1"
API_TOKEN = "WKLEJ_TUTAJ_SWOJ_AKTUALNY_TOKEN"
BASE_URL = "http://label-studio:8080/api"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

# Folder widziany z perspektywy Label Studio (do URL-i obrazów)
LS_DATA_ROOT = "/opt/data/labeled"
# Folder widziany z perspektywy kontenera, w którym uruchamia się ten skrypt
# (powinien być tym samym wolumenem zamontowanym w kontenerze Airflow)
LOCAL_DATA_ROOT = "/opt/airflow/data/labeled"  # <- DOSTOSUJ jeśli trzeba

LABEL_NAME = "hand"  # nazwa etykiety skonfigurowanej w Labeling Interface


def yolo_to_ls_bbox(cx, cy, w, h):
    """Konwertuje YOLO (środek, szer/wys) na Label Studio (lewy-górny róg, szer/wys) w %."""
    x = (cx - w / 2) * 100
    y = (cy - h / 2) * 100
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "width": round(w * 100, 3),
        "height": round(h * 100, 3),
        "rotation": 0
    }


def build_task(image_filename, labels_file_path):
    """Buduje jeden task JSON dla Label Studio."""
    # Ścieżka URL, którą Label Studio użyje do wyświetlenia obrazu
    image_url = f"/data/local-files/?d=labeled/{image_filename}"

    results = []
    if os.path.exists(labels_file_path):
        with open(labels_file_path, "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # YOLO: class cx cy w h
                _, cx, cy, w, h = parts
                bbox = yolo_to_ls_bbox(float(cx), float(cy), float(w), float(h))
                results.append({
                    "id": f"{image_filename}_{idx}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": 100,
                    "original_height": 100,
                    "image_rotation": 0,
                    "value": {
                        "x": bbox["x"],
                        "y": bbox["y"],
                        "width": bbox["width"],
                        "height": bbox["height"],
                        "rotation": 0,
                        "rectanglelabels": [LABEL_NAME]
                    }
                })

    return {
        "data": {"image": image_url},
        "predictions": [{
            "model_version": "grounding-dino",
            "result": results
        }]
    }


def main():
    print("Konwersja YOLO -> JSON i import do Label Studio...")

    tasks = []
    image_files = sorted(glob.glob(os.path.join(LOCAL_DATA_ROOT, "**", "*.jpg"), recursive=True))
    image_files += sorted(glob.glob(os.path.join(LOCAL_DATA_ROOT, "**", "*.png"), recursive=True))

    print(f"Znaleziono {len(image_files)} obrazów.")

    for img_path in image_files:
        # Ścieżka relatywna do LOCAL_DATA_ROOT -> do URL-a
        rel_path = os.path.relpath(img_path, LOCAL_DATA_ROOT)
        labels_path = os.path.splitext(img_path)[0] + ".txt"
        tasks.append(build_task(rel_path, labels_path))

    if not tasks:
        print("Brak obrazów do zaimportowania.")
        return

    print(f"Wysyłanie {len(tasks)} zadań do Label Studio...")
    url = f"{BASE_URL}/projects/{PROJECT_ID}/import"
    response = requests.post(url, headers=HEADERS, json=tasks)

    if response.status_code in (200, 201):
        print(f"Sukces! Zaimportowano {len(tasks)} zadań z gotowymi ramkami.")
    else:
        raise Exception(f"Błąd importu: {response.status_code} - {response.text}")


if __name__ == "__main__":
    main()