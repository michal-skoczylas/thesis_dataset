import os
import sqlite3
import requests
import json
import cv2

# ==== KONFIGURACJA ====
PROJECT_ID = 1
BASE_URL = "http://label-studio:8080/api"
LS_DB_PATH = "/opt/airflow/ls_data/label_studio.sqlite3"

DATA_DIR = "/opt/airflow/data"
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def get_token():
    conn = sqlite3.connect(f"file:{LS_DB_PATH}?mode=ro", uri=True)
    row = conn.execute("SELECT key FROM authtoken_token ORDER BY created DESC LIMIT 1").fetchone()
    conn.close()
    if not row: raise Exception("Brak tokena w bazie!")
    return row[0]

def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    token = get_token()
    headers = {"Authorization": f"Token {token}"}

    print("Pobieranie zatwierdzonych zadań z Label Studio...")
    # exportType=JSON pobiera tylko te zadania, które zostały zatwierdzone przez człowieka
    url = f"{BASE_URL}/projects/{PROJECT_ID}/export?exportType=JSON"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    
    tasks = r.json()
    print(f"Znaleziono {len(tasks)} oznaczonych obrazów.")

    # 1. Zapisz metadane dla Gita
    json_path = os.path.join(ANNOTATIONS_DIR, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    print(f"Zapisano konfigurację JSON dla Gita: {json_path}")

    # 2. Wytnij obrazy w OpenCV dla DVC
    crops_count = 0
    for task in tasks:
        img_url = task["data"]["image"]
        rel_img_path = img_url.split("?d=")[-1] 
        abs_img_path = os.path.join(DATA_DIR, rel_img_path)

        if not os.path.exists(abs_img_path):
            continue

        img = cv2.imread(abs_img_path)
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        for annotation in task["annotations"]:
            for result in annotation.get("result", []):
                if result["type"] != "rectanglelabels": continue
                
                val = result["value"]
                x = int(val["x"] * w_img / 100)
                y = int(val["y"] * h_img / 100)
                w = int(val["width"] * w_img / 100)
                h = int(val["height"] * h_img / 100)
                
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)

                crop_img = img[y1:y2, x1:x2]

                if crop_img.size != 0:
                    base_name = os.path.basename(abs_img_path).split(".")[0]
                    box_id = result["id"]
                    crop_filename = f"{base_name}_{box_id}.jpg"
                    crop_path = os.path.join(PROCESSED_DIR, crop_filename)
                    
                    cv2.imwrite(crop_path, crop_img)
                    crops_count += 1

    print(f"Zakończono! Zapisano {crops_count} wyciętych dłoni do {PROCESSED_DIR}")

if __name__ == "__main__":
    main()