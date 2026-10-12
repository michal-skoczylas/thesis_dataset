import os
import glob
import sqlite3
import requests

# ==== KONFIGURACJA ====
PROJECT_ID = 1
BASE_URL = "http://label-studio:8080/api"
LS_DB_PATH = "/opt/airflow/ls_data/label_studio.sqlite3"
DATA_DIR = "/opt/airflow/data/labeled"
LS_STORAGE_PATH = "/opt/data/labeled"

def get_token():
    """Automatycznie wyciąga token administratora z bazy SQLite Label Studio."""
    if not os.path.exists(LS_DB_PATH):
        raise FileNotFoundError(f"Brak pliku bazy: {LS_DB_PATH}. Upewnij się, że wolumeny są zmapowane.")
    
    conn = sqlite3.connect(f"file:{LS_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT key FROM authtoken_token ORDER BY created DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise Exception("Brak tokena. Zaloguj się przynajmniej raz do UI Label Studio, aby wygenerować klucz.")
    return row[0]

def ensure_local_storage(headers):
    """Rejestruje Local Storage w projekcie przez API (zastępuje wyklikiwanie w UI)."""
    # Sprawdzenie istniejących magazynów
    r = requests.get(f"{BASE_URL}/storages/localfiles?project={PROJECT_ID}", headers=headers)
    r.raise_for_status()
    storages = r.json()
    
    for storage in storages:
        if storage.get("path") == LS_STORAGE_PATH:
            print("Local Storage jest już poprawnie zarejestrowany.")
            return

    # Jeśli nie istnieje, utwórz nowy
    print("Rejestracja nowego Local Storage w Label Studio...")
    payload = {
        "project": PROJECT_ID,
        "path": LS_STORAGE_PATH,
        "title": "Airflow Auto Storage",
        "use_blob_urls": True
    }
    requests.post(f"{BASE_URL}/storages/localfiles", headers=headers, json=payload).raise_for_status()

def main():
    token = get_token()
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    # Odblokowanie zabezpieczeń dostępu do plików
    ensure_local_storage(headers)

    # Przygotowanie zadań (JSON)
    tasks = []
    for img in glob.glob(f"{DATA_DIR}/**/*.jpg", recursive=True):
        txt = img.replace("/images/", "/labels/").replace(".jpg", ".txt")
        ls_path = img.replace("/opt/airflow/data/", "")
        
        results = []
        if os.path.exists(txt):
            with open(txt) as f:
                for i, line in enumerate(f):
                    parts = line.split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        results.append({
                            "id": str(i),
                            "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                            "value": {
                                "x": (cx - w/2) * 100, "y": (cy - h/2) * 100,
                                "width": w * 100, "height": h * 100,
                                "rectanglelabels": ["hand"]
                            }
                        })
        
        task = {"data": {"image": f"/data/local-files/?d={ls_path}"}}
        if results:
            task["predictions"] = [{"result": results}]
        tasks.append(task)

    # Wysłanie zadań do Label Studio
    print(f"Wysyłanie {len(tasks)} zadań z przypisanymi predykcjami...")
    requests.post(f"{BASE_URL}/projects/{PROJECT_ID}/import", headers=headers, json=tasks).raise_for_status()
    print("=== GOTOWE ===")

if __name__ == "__main__":
    main()