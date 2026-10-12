import os
import argparse
from pathlib import Path
import requests

from config import (
    BASE_DIR,
    DATA_DIR,
    LABELED_DIR,
    LABEL_STUDIO_URL,
    PROJECT_ID,
    LS_STORAGE_PATH,
    LABEL_STUDIO_XML_CONFIG,
    get_auth_headers,
)


def ensure_project(headers: dict, project_id: int) -> int:
    """Sprawdza czy projekt w Label Studio istnieje. Jeśli nie, tworzy go z konfiguracją wykrywania dłoni."""
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        print(f"Projekt o ID {project_id} istnieje w Label Studio.")
        return project_id

    print(f"Projekt {project_id} nie został znaleziony. Tworzenie nowego projektu w Label Studio...")
    create_url = f"{LABEL_STUDIO_URL}/api/projects"
    payload = {
        "title": "Thesis Drone Hand Gestures",
        "description": "Projekt detekcji i weryfikacji gestów dłoni",
        "label_config": LABEL_STUDIO_XML_CONFIG
    }
    create_resp = requests.post(create_url, headers=headers, json=payload)
    create_resp.raise_for_status()
    created_id = create_resp.json()["id"]
    print(f"Utworzono projekt o ID {created_id}.")
    return created_id


def ensure_local_storage(headers: dict, project_id: int):
    """Rejestruje Local Storage w projekcie przez API, jeśli jeszcze nie istnieje."""
    url = f"{LABEL_STUDIO_URL}/api/storages/localfiles?project={project_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    storages = r.json()

    for storage in storages:
        if storage.get("path") == LS_STORAGE_PATH:
            print("Magazyn Local Storage (/opt/data/labeled) jest już podpięty.")
            return storage.get("id")

    print(f"Rejestracja nowego Local Storage ({LS_STORAGE_PATH}) w Label Studio...")
    payload = {
        "project": project_id,
        "path": LS_STORAGE_PATH,
        "title": "Dataset Auto Storage",
        "use_blob_urls": True
    }
    resp = requests.post(f"{LABEL_STUDIO_URL}/api/storages/localfiles", headers=headers, json=payload)
    resp.raise_for_status()
    storage_id = resp.json()["id"]
    print("Zarejestrowano Local Storage pomyślnie.")
    return storage_id


def import_dataset(labeled_dir: Path = LABELED_DIR, project_id: int = PROJECT_ID):
    headers = get_auth_headers()
    project_id = ensure_project(headers, project_id)
    ensure_local_storage(headers, project_id)

    # Wyszukaj wszystkie obrazy w labeled_dir (.jpg, .jpeg, .png)
    supported_exts = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in labeled_dir.rglob("*") if p.suffix.lower() in supported_exts]

    print(f"Znaleziono {len(all_images)} obrazów w {labeled_dir}. Przygotowywanie zadań z predykcjami...")

    tasks = []
    for img_path in sorted(all_images):
        # Sprawdź odpowiadający plik etykiety YOLO (.txt)
        txt_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not txt_path.exists():
            txt_path = img_path.with_suffix(".txt")

        # Ścieżka relatywna do montowania w kontenerze Label Studio (/opt/data)
        rel_to_data = img_path.relative_to(DATA_DIR)
        ls_path = str(rel_to_data).replace("\\", "/")

        results = []
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        results.append({
                            "id": str(i),
                            "from_name": "label",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "x": max(0.0, (cx - w / 2.0) * 100),
                                "y": max(0.0, (cy - h / 2.0) * 100),
                                "width": min(100.0, w * 100),
                                "height": min(100.0, h * 100),
                                "rectanglelabels": ["hand"]
                            }
                        })

        task = {"data": {"image": f"/data/local-files/?d={ls_path}"}}
        if results:
            task["predictions"] = [{"result": results}]
        tasks.append(task)

    if not tasks:
        print("Brak zadań do zaimportowania.")
        return

    import_url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import"
    print(f"Wysyłanie {len(tasks)} zadań do projektu {project_id}...")
    resp = requests.post(import_url, headers=headers, json=tasks)
    resp.raise_for_status()
    print("=== Pomyślnie zaimportowano zadania do Label Studio! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import oznaczonych klatek do Label Studio")
    parser.add_argument("--labeled-dir", type=str, default=str(LABELED_DIR), help="Katalog z oznaczonymi obrazami")
    parser.add_argument("--project-id", type=int, default=PROJECT_ID, help="ID projektu w Label Studio")
    args = parser.parse_args()

    import_dataset(labeled_dir=Path(args.labeled_dir), project_id=args.project_id)