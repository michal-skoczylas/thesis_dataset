import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import requests
import yaml

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

KNOWN_GESTURES = ["closehand", "openhand", "dislike", "shaka", "left", "right", "ok", "v"]
KNOWN_DISTANCES = ["1_4metra", "1_7metra", "2metry"]


def load_params():
    """Wczytuje parametry z params.yaml jeśli plik istnieje."""
    params_file = BASE_DIR / "params.yaml"
    if params_file.exists():
        try:
            with open(params_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}


def parse_session_from_filename(stem: str) -> str:
    """Ekstraktuje identyfikator sesji z nazwy pliku."""
    parts = stem.rsplit("_", 1)
    if len(parts) > 1:
        return parts[0]
    return "session_default"


def extract_metadata_from_stem(stem: str):
    """Wyciąga gest i odległość z nazwy sesji/pliku."""
    s_low = stem.lower()
    gesture = next((g for g in sorted(KNOWN_GESTURES, key=len, reverse=True) if g in s_low), "unknown")
    distance = next((d for d in KNOWN_DISTANCES if d in s_low), "unknown")
    return gesture, distance


def ensure_project(headers: dict, project_id: int = None, title: str = None) -> int:
    """Sprawdza czy projekt w Label Studio istnieje po ID lub tytule. Jeśli istnieje z inną nazwą, aktualizuje nazwę."""
    if project_id is not None:
        url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}"
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            p_data = resp.json()
            curr_title = p_data.get("title", "")
            if title and curr_title != title:
                # Zaktualizuj nazwę projektu na żądaną
                patch_resp = requests.patch(url, headers=headers, json={"title": title})
                if patch_resp.status_code == 200:
                    print(f"Zaktualizowano nazwę projektu ID {project_id}: '{curr_title}' -> '{title}'")
            return project_id

    # Sprawdź po tytule wśród wszystkich projektów
    if title:
        list_resp = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
        if list_resp.status_code == 200:
            projects = list_resp.json().get("results", [])
            for p in projects:
                if p.get("title") == title:
                    return p.get("id")

    # Utwórz nowy projekt
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


def clear_project_tasks(headers: dict, project_id: int):
    """Usuwa wszystkie zadania z projektu (pod kątem re-partycjonowania)."""
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks/"
    resp = requests.delete(url, headers=headers)
    if resp.status_code in (200, 204):
        print(f"   🗑️  Wyczyszczono wcześniejsze zadania w projekcie ID {project_id}.")
    else:
        print(f"   ⚠️  Nie udało się wyczyścić zadań w projekcie {project_id}: {resp.status_code}")


def ensure_local_storage(headers: dict, project_id: int):
    """Rejestruje Local Storage w projekcie przez API, jeśli jeszcze nie istnieje."""
    url = f"{LABEL_STUDIO_URL}/api/storages/localfiles?project={project_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    storages = r.json()

    for storage in storages:
        if storage.get("path") == LS_STORAGE_PATH:
            return storage.get("id")

    payload = {
        "project": project_id,
        "path": LS_STORAGE_PATH,
        "title": "Dataset Auto Storage",
        "use_blob_urls": True
    }
    resp = requests.post(f"{LABEL_STUDIO_URL}/api/storages/localfiles", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


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


def partition_sessions(tasks_by_session: dict, num_users: int = 2, strategy: str = "interleaved", ratio_str: str = "50:50"):
    """
    Dzieli sesje wideo na użytkowników wg wybranej strategii:
    1. 'interleaved': naprzemienne przydzielanie sesji (zrównoważony podział gestów i odległości).
    2. 'distance': podział wg odległości od drona (np. User 1: 1.4m + 1.7m, User 2: 2.0m + 1.7m).
    3. 'gesture': podział wg typów gestów (np. User 1: closehand/openhand/ok/v, User 2: shaka/dislike/left/right).
    4. 'ratio': podział proporcjonalny według zadanego stosunku (np. 60:40, 70:30).
    """
    sorted_sessions = sorted(list(tasks_by_session.keys()))
    user_sessions = defaultdict(list)

    if strategy == "distance" and num_users == 2:
        # User 1: bliższe ujęcia (1.4m + połowa 1.7m), User 2: dalsze ujęcia (2.0m + połowa 1.7m)
        s_14 = [s for s in sorted_sessions if "1_4metra" in s]
        s_17 = [s for s in sorted_sessions if "1_7metra" in s]
        s_20 = [s for s in sorted_sessions if "2metry" in s]

        user_sessions[0].extend(s_14)
        user_sessions[1].extend(s_20)

        for idx, s in enumerate(s_17):
            u_idx = idx % 2
            user_sessions[u_idx].append(s)

    elif strategy == "gesture" and num_users == 2:
        # User 1: 4 gesty (closehand, openhand, ok, v), User 2: 4 gesty (shaka, dislike, left, right)
        g_group_0 = {"closehand", "openhand", "ok", "v"}
        for s in sorted_sessions:
            gest, _ = extract_metadata_from_stem(s)
            if gest in g_group_0:
                user_sessions[0].append(s)
            else:
                user_sessions[1].append(s)

    elif strategy == "ratio":
        # Parsowanie proporcji np. "60:40" lub "70:30"
        try:
            parts = [float(p.strip()) for p in ratio_str.replace("/", ":").replace(",", ":").split(":")]
            total_ratio = sum(parts)
            target_pcts = [p / total_ratio for p in parts]
        except Exception:
            target_pcts = [1.0 / num_users] * num_users

        total_frames = sum(len(tasks_by_session[s]) for s in sorted_sessions)
        target_frames = [int(round(pct * total_frames)) for pct in target_pcts]

        # Greedy packing sesji do osiągnięcia celu
        current_frames = [0] * num_users
        for s in sorted(sorted_sessions, key=lambda s: len(tasks_by_session[s]), reverse=True):
            s_frames = len(tasks_by_session[s])
            # Wybierz usera, który jest najbardziej poniżej celu
            def deficit(u):
                return target_frames[u] - current_frames[u]
            best_u = max(range(num_users), key=deficit)
            user_sessions[best_u].append(s)
            current_frames[best_u] += s_frames

    else:
        # Domyślny podział naprzemienny (interleaved)
        for idx, s in enumerate(sorted_sessions):
            user_sessions[idx % num_users].append(s)

    return user_sessions


def import_dataset(
    labeled_dir: Path = LABELED_DIR,
    project_id: int = PROJECT_ID,
    split_coop: int = None,
    coop_names: str = None,
    strategy: str = None,
    ratio: str = None,
    reassign: bool = False
):
    # Wczytaj konfigurację z params.yaml
    params = load_params().get("coop", {})
    default_names = params.get("names", ["dane michal", "dane janek"])
    default_strategy = params.get("strategy", "interleaved")
    default_ratio = params.get("ratio", "50:50")

    if coop_names:
        names = [n.strip() for n in coop_names.split(",") if n.strip()]
    elif isinstance(default_names, list):
        names = default_names
    else:
        names = ["dane michal", "dane janek"]

    if split_coop is None:
        split_coop = len(names)

    if strategy is None:
        strategy = default_strategy
    if ratio is None:
        ratio = default_ratio

    headers = get_auth_headers()
    labeled_dir = Path(labeled_dir)
    tasks_by_session = build_tasks_from_labeled(labeled_dir)
    total_tasks = sum(len(ts) for ts in tasks_by_session.values())

    print(f"\n📂 Odnaleziono {total_tasks} klatek w {len(tasks_by_session)} sesjach wideo.")

    if split_coop <= 1:
        # Tryb solo (pojedynczy projekt)
        p_id = ensure_project(headers, project_id=project_id, title=names[0] if names else None)
        ensure_local_storage(headers, p_id)
        if reassign:
            clear_project_tasks(headers, p_id)

        all_tasks = [t for ts in tasks_by_session.values() for t in ts]
        _send_tasks_to_project(headers, p_id, all_tasks)
        print(f"\n✅ Zadania zostały wgrane do projektu {p_id} ('{names[0]}').")
        return

    # Tryb Co-op
    print(f"👥 Tryb CO-OP dla {split_coop} osób:")
    print(f"   • Projekty         : {', '.join(names)}")
    print(f"   • Strategia podziału: {strategy.upper()} ({f'proporcja {ratio}' if strategy == 'ratio' else ''})")
    print(f"   • Wymuszenie czyszczenia (reassign): {'TAK' if reassign else 'NIE (tylko brakujące)'}")

    user_sessions = partition_sessions(tasks_by_session, num_users=split_coop, strategy=strategy, ratio_str=ratio)

    created_project_ids = []
    user_tasks_summary = []

    for u_idx, name in enumerate(names):
        cand_p_id = (u_idx + 1) if u_idx < 2 else None
        p_id = ensure_project(headers, project_id=cand_p_id, title=name)
        ensure_local_storage(headers, p_id)
        created_project_ids.append(p_id)

        if reassign:
            clear_project_tasks(headers, p_id)

        sessions = user_sessions[u_idx]
        tasks = [t for s in sessions for t in tasks_by_session[s]]
        user_tasks_summary.append((name, p_id, len(sessions), len(tasks), sessions))

        print(f"\n👤 [{name}] Wgrywanie {len(tasks)} klatek ({len(sessions)} sesji) do projektu ID {p_id}...")
        _send_tasks_to_project(headers, p_id, tasks)

    print("\n" + "=" * 72)
    print(f"🎉 PODZIAŁ CO-OP ({strategy.upper()}) ZOSTAŁ ZASTOSOWANY W LABEL STUDIO!")
    print("=" * 72)
    for name, p_id, n_sess, n_tasks, sessions in user_tasks_summary:
        sample_s = ", ".join(sessions[:2]) + ("..." if len(sessions) > 2 else "")
        print(f"📁 Projekt ID: {p_id:<3} | Tytuł: '{name}'")
        print(f"   • Przypisano klatek: {n_tasks} ({n_sess} sesji: {sample_s})")
        print(f"   • URL lokalny       : {LABEL_STUDIO_URL}/projects/{p_id}")
        print("-" * 72)
    print("💡 Aby uruchomić tunel dla kolegi: make tunnel")
    print("======================================================================\n")


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
    print(f"   ✅ Projekt {project_id}: Zaimportowano {len(new_tasks)} nowych zadań.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import oznaczonych klatek do Label Studio z konfigurowalnym podziałem Co-op")
    parser.add_argument("--labeled-dir", type=str, default=str(LABELED_DIR), help="Katalog z oznaczonymi obrazami")
    parser.add_argument("--project-id", type=int, default=PROJECT_ID, help="Domyślne ID projektu")
    parser.add_argument("--split-coop", type=int, default=None, help="Liczba osób do podziału (domyślnie 2)")
    parser.add_argument("--coop-names", type=str, default=None, help="Nazwy projektów oddzielone przecinkiem, np. 'dane michal,dane janek'")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["interleaved", "distance", "gesture", "ratio"],
        default=None,
        help="Strategia podziału zadań: interleaved (naprzemiennie), distance (wg odległości), gesture (wg gestów), ratio (wg proporcji)"
    )
    parser.add_argument("--ratio", type=str, default=None, help="Stosunek podziału dla strategii ratio, np. '50:50', '60:40'")
    parser.add_argument("--reassign", action="store_true", help="Wyczyść istniejące zadania w projektach przed nowym podziałem")

    args = parser.parse_args()

    import_dataset(
        labeled_dir=Path(args.labeled_dir),
        project_id=args.project_id,
        split_coop=args.split_coop,
        coop_names=args.coop_names,
        strategy=args.strategy,
        ratio=args.ratio,
        reassign=args.reassign
    )