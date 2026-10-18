import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

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


def parse_session_from_filename(stem: str) -> str:
    """Ekstraktuje identyfikator sesji z nazwy pliku."""
    parts = stem.rsplit("_", 1)
    if len(parts) > 1:
        return parts[0]
    return "session_default"


def ensure_project(headers: dict, project_id: int = None, title: str = None) -> int:
    """Sprawdza czy projekt w Label Studio istnieje po ID lub tytule. Jeśli nie, tworzy go."""
    # 1. Jeśli podano ID, sprawdź czy istnieje
    if project_id is not None:
        url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            p_data = resp.json()
            p_title = p_data.get("title", "")
            print(f"Projekt o ID {project_id} ('{p_title}') istnieje w Label Studio.")
            return project_id

    # 2. Sprawdź po tytule wśród wszystkich projektów
    if title:
        list_resp = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
        if list_resp.status_code == 200:
            projects = list_resp.json().get("results", [])
            for p in projects:
                if p.get("title") == title:
                    print(f"Znaleziono istniejący projekt '{title}' (ID {p.get('id')}).")
                    return p.get("id")

    # 3. Utwórz nowy projekt
    target_title = title or (f"Thesis Drone Project {project_id}" if project_id else "Thesis Drone Hand Gestures")
    print(f"Tworzenie nowego projektu w Label Studio: '{target_title}'...")
    create_url = f"{LABEL_STUDIO_URL}/api/projects"
    payload = {
        "title": target_title,
        "description": "Projekt detekcji i weryfikacji gestów dłoni z drona",
        "label_config": LABEL_STUDIO_XML_CONFIG
    }
    create_resp = requests.post(create_url, headers=headers, json=payload)
    create_resp.raise_for_status()
    created_id = create_resp.json()["id"]
    print(f"Utworzono projekt o ID {created_id} ('{target_title}').")
    return created_id


def ensure_local_storage(headers: dict, project_id: int):
    """Rejestruje Local Storage w projekcie przez API, jeśli jeszcze nie istnieje."""
    url = f"{LABEL_STUDIO_URL}/api/storages/localfiles?project={project_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    storages = r.json()

    for storage in storages:
        if storage.get("path") == LS_STORAGE_PATH:
            return storage.get("id")

    print(f"Rejestracja Local Storage ({LS_STORAGE_PATH}) w projekcie {project_id}...")
    payload = {
        "project": project_id,
        "path": LS_STORAGE_PATH,
        "title": "Dataset Auto Storage",
        "use_blob_urls": True
    }
    resp = requests.post(f"{LABEL_STUDIO_URL}/api/storages/localfiles", headers=headers, json=payload)
    resp.raise_for_status()
    storage_id = resp.json()["id"]
    return storage_id


def build_tasks_from_labeled(labeled_dir: Path):
    """Wyszukuje wszystkie obrazy i tworzy zadania z predykcjami dla Label Studio."""
    supported_exts = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in labeled_dir.rglob("*") if p.suffix.lower() in supported_exts]

    tasks_by_session = defaultdict(list)
    for img_path in sorted(all_images):
        txt_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not txt_path.exists():
            txt_path = img_path.with_suffix(".txt")

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

        session_id = parse_session_from_filename(img_path.stem)
        tasks_by_session[session_id].append(task)

    return tasks_by_session


def import_dataset(
    labeled_dir: Path = LABELED_DIR,
    project_id: int = PROJECT_ID,
    split_coop: int = 1,
    coop_names: str = "Michał,Kolega"
):
    headers = get_auth_headers()
    labeled_dir = Path(labeled_dir)
    tasks_by_session = build_tasks_from_labeled(labeled_dir)

    total_tasks = sum(len(ts) for ts in tasks_by_session.values())
    print(f"\nZnaleziono {total_tasks} klatek w {len(tasks_by_session)} unikalnych sesjach wideo.")

    if split_coop <= 1:
        # Tryb jednoosobowy (standardowy)
        p_id = ensure_project(headers, project_id=project_id)
        ensure_local_storage(headers, p_id)

        all_tasks = [t for ts in tasks_by_session.values() for t in ts]
        _send_tasks_to_project(headers, p_id, all_tasks)
        print(f"\n✅ Zadania zostały wgrane do projektu {p_id} ({LABEL_STUDIO_URL}/projects/{p_id})")
    else:
        # Tryb Co-op: podział na N osób (np. Michał i Kolega)
        names = [n.strip() for n in coop_names.split(",")][:split_coop]
        while len(names) < split_coop:
            names.append(f"Osoba_{len(names)+1}")

        print(f"\n👥 Aktywowano tryb CO-OP dla {split_coop} osób: {', '.join(names)}")

        # Równomierny podział sesji (interleaved)
        sorted_sessions = sorted(list(tasks_by_session.keys()))
        user_tasks = defaultdict(list)
        user_sessions_count = defaultdict(int)

        for s_idx, s_id in enumerate(sorted_sessions):
            user_idx = s_idx % split_coop
            user_tasks[user_idx].extend(tasks_by_session[s_id])
            user_sessions_count[user_idx] += 1

        created_project_ids = []
        for u_idx, name in enumerate(names):
            p_title = f"Thesis Drone - {name}"
            # Dla pierwszego usera próbuj powiązać z PROJECT_ID jeśli to projekt 1
            cand_p_id = PROJECT_ID if (u_idx == 0 and PROJECT_ID is not None) else None
            p_id = ensure_project(headers, project_id=cand_p_id, title=p_title)
            ensure_local_storage(headers, p_id)
            created_project_ids.append(p_id)

            tasks_for_user = user_tasks[u_idx]
            print(f"\n👤 [Użytkownik {u_idx+1}: {name}] Wgrywanie {len(tasks_for_user)} zadań ({user_sessions_count[u_idx]} sesji) do projektu ID {p_id}...")
            _send_tasks_to_project(headers, p_id, tasks_for_user)

        print("\n" + "=" * 70)
        print("🎉 ZAKOŃCZONO IMPORT CO-OP DO LABEL STUDIO!")
        print("=" * 70)
        for u_idx, name in enumerate(names):
            p_id = created_project_ids[u_idx]
            print(f"👤 {name:<10} -> Projekt ID: {p_id:<3} | URL: {LABEL_STUDIO_URL}/projects/{p_id} ({len(user_tasks[u_idx])} klatek)")
        print("-" * 70)
        print("💡 Aby udostępnić Label Studio koledze przez bezpieczny tunel (bez instalacji):")
        print("   cloudflared tunnel --url http://localhost:8081")
        print("   (kolega otwiera wygenerowany link HTTPS w przeglądarce i oznacza swój projekt!)")
        print("=" * 70 + "\n")


def _send_tasks_to_project(headers: dict, project_id: int, tasks: list):
    """Filtruje już istniejące zadania w projekcie i wysyła wyłącznie nowe."""
    existing_tasks_resp = requests.get(f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks?page_size=10000", headers=headers)
    existing_urls = set()
    if existing_tasks_resp.status_code == 200:
        existing_data = existing_tasks_resp.json()
        if isinstance(existing_data, list):
            existing_urls = {t.get("data", {}).get("image") for t in existing_data if "data" in t}

    new_tasks = [t for t in tasks if t.get("data", {}).get("image") not in existing_urls]
    if not new_tasks:
        print(f"   ℹ️  Projekt {project_id}: Wszystkie zadania ({len(existing_urls)}) są już wgrane.")
        return

    import_url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/import"
    resp = requests.post(import_url, headers=headers, json=new_tasks)
    resp.raise_for_status()
    print(f"   ✅ Projekt {project_id}: Pomyślnie zaimportowano {len(new_tasks)} nowych zadań.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import oznaczonych klatek do Label Studio (tryb solo lub co-op)")
    parser.add_argument("--labeled-dir", type=str, default=str(LABELED_DIR), help="Katalog z oznaczonymi obrazami")
    parser.add_argument("--project-id", type=int, default=PROJECT_ID, help="Domyślne ID projektu")
    parser.add_argument("--split-coop", type=int, default=1, help="Liczba osób do podziału zadań (np. 2)")
    parser.add_argument("--coop-names", type=str, default="Michał,Kolega", help="Imiona/nazwy osób do projektów, oddzielone przecinkiem")
    args = parser.parse_args()

    import_dataset(
        labeled_dir=Path(args.labeled_dir),
        project_id=args.project_id,
        split_coop=args.split_coop,
        coop_names=args.coop_names
    )