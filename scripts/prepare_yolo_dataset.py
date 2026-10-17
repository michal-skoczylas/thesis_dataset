import os
import sys
import json
import shutil
import random
import zipfile
import argparse
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import cv2

# Dodanie ścieżki do importu konfiguracji
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

try:
    from config import (
        LABELED_DIR,
        DATA_DIR,
        ANNOTATIONS_DIR,
        LABEL_STUDIO_URL,
        PROJECT_ID,
        get_auth_headers,
    )
except ImportError:
    LABELED_DIR = BASE_DIR / "data" / "labeled"
    DATA_DIR = BASE_DIR / "data"
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    LABEL_STUDIO_URL = "http://localhost:8081"
    PROJECT_ID = 1

    def get_auth_headers():
        return {}

DEFAULT_YOLO_DIR = DATA_DIR / "yolo_dataset"
DEFAULT_CNN_DIR = DATA_DIR / "cnn_dataset"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# Kanoniczne klasy gestów w projekcie
KNOWN_GESTURES = ["closehand", "openhand", "dislike", "shaka", "left", "right", "ok", "v"]
KNOWN_DISTANCES = ["1_4metra", "1_7metra", "2metry", "1.4m", "1.7m", "2.0m", "2m"]


def parse_session_from_filename(stem: str) -> str:
    """
    Ekstraktuje identyfikator sesji z nazwy pliku.
    Format: <odleglosc>_<gest>_<data-godzina>_<klatka>
    Przykład: 1_4metra_closehand_2026-10-15-14-45-08_00000 -> 1_4metra_closehand_2026-10-15-14-45-08
    """
    parts = stem.rsplit("_", 1)
    if len(parts) > 1:
        return parts[0]
    return "session_default"


def extract_metadata_from_stem(stem: str):
    """Wyciąga gest i odległość z nazwy pliku."""
    s_low = stem.lower()
    gesture = next((g for g in sorted(KNOWN_GESTURES, key=len, reverse=True) if g in s_low), "unknown")
    distance = next((d for d in KNOWN_DISTANCES if d in s_low), "unknown")
    return gesture, distance


def fetch_submitted_tasks_from_ls(project_id: int = PROJECT_ID, save_to: Path = None) -> list:
    """Pobiera wyłącznie zadania zatwierdzone przez użytkownika (Submit) z Label Studio API."""
    import requests

    if save_to is None:
        save_to = ANNOTATIONS_DIR / "dataset.json"

    print(f"📡 Łączenie z Label Studio ({LABEL_STUDIO_URL}, Projekt {project_id})...")
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export?exportType=JSON"
    headers = get_auth_headers()

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tasks = r.json()
        print(f"📥 Pomyślnie pobrano {len(tasks)} zatwierdzonych zadań (Submit) z Label Studio.")

        save_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        return tasks
    except Exception as e:
        print(f"⚠️  Nie udało się pobrać danych bezpośrednio z API Label Studio: {e}")
        if save_to.exists():
            print(f"📂 Wczytuję ostatnio wyeksportowany plik: {save_to}")
            with open(save_to, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


def load_items_from_label_studio(tasks: list) -> list:
    """
    Konwertuje zadania z Label Studio (posiadające zatwierdzenie Submit) na obiekty do zbioru YOLO.
    Uwzględnia poprawki naniesione przez człowieka w UI Label Studio.
    """
    items = []
    skipped_no_annotation = 0
    missing_images = 0

    for task in tasks:
        annotations = task.get("annotations", [])
        valid_annotations = [a for a in annotations if not a.get("was_cancelled", False)]
        if not valid_annotations:
            skipped_no_annotation += 1
            continue

        img_url = task.get("data", {}).get("image", "")
        if not img_url:
            continue

        # Rozwiązanie ścieżki do obrazu
        if "?d=" in img_url:
            rel_img_path = img_url.split("?d=")[-1]
            abs_img_path = DATA_DIR / rel_img_path
        else:
            abs_img_path = DATA_DIR / Path(img_url).name

        if not abs_img_path.exists():
            cands = list(DATA_DIR.rglob(abs_img_path.name))
            if cands:
                abs_img_path = cands[0]
            else:
                missing_images += 1
                continue

        # Bierzemy najnowszą zatwierdzoną adnotację człowieka
        latest_annotation = valid_annotations[-1]
        results = latest_annotation.get("result", [])

        boxes = []
        for res in results:
            if res.get("type") != "rectanglelabels":
                continue
            val = res.get("value", {})
            x_pct = float(val.get("x", 0))
            y_pct = float(val.get("y", 0))
            w_pct = float(val.get("width", 0))
            h_pct = float(val.get("height", 0))

            w_norm = max(0.0, min(1.0, w_pct / 100.0))
            h_norm = max(0.0, min(1.0, h_pct / 100.0))
            cx_norm = max(0.0, min(1.0, (x_pct + w_pct / 2.0) / 100.0))
            cy_norm = max(0.0, min(1.0, (y_pct + h_pct / 2.0) / 100.0))

            if w_norm > 0 and h_norm > 0:
                boxes.append((0, [cx_norm, cy_norm, w_norm, h_norm]))

        gesture, distance = extract_metadata_from_stem(abs_img_path.stem)

        items.append({
            "img": abs_img_path,
            "boxes": boxes,
            "is_positive": len(boxes) > 0,
            "gesture": gesture,
            "distance": distance,
            "source": "label_studio",
            "task_id": task.get("id")
        })

    if skipped_no_annotation > 0:
        print(f"ℹ️  Pominięto {skipped_no_annotation} zadań bez zatwierdzenia (brak przycisku 'Submit').")
    if missing_images > 0:
        print(f"⚠️  Nie odnaleziono plików graficznych dla {missing_images} zadań.")

    return items


def load_items_from_labeled_dir(labeled_dir: Path) -> list:
    """Fallback: Wczytuje dane bezpośrednio z katalogu data/labeled (wszystkie predykcje)."""
    images_dir = labeled_dir / "train" / "images"
    labels_dir = labeled_dir / "train" / "labels"

    if not images_dir.exists():
        candidate = labeled_dir / "images"
        if candidate.exists():
            images_dir = candidate
            labels_dir = labeled_dir / "labels"
        else:
            raise FileNotFoundError(f"Nie znaleziono katalogu ze zdjęciami w {images_dir}")

    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in supported_exts])

    items = []
    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cls_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            if all(0.0 <= c <= 1.0 for c in coords):
                                boxes.append((cls_id, coords))
                        except ValueError:
                            pass

        gesture, distance = extract_metadata_from_stem(img_path.stem)
        items.append({
            "img": img_path,
            "boxes": boxes,
            "is_positive": len(boxes) > 0,
            "gesture": gesture,
            "distance": distance,
            "source": "labeled_dir"
        })
    return items


def compare_and_report_diff(current_items: list, manifest_path: Path = MANIFEST_PATH):
    """Porównuje bieżący zbiór klatek z poprzednim stanem i raportuje przyrost danych."""
    current_filenames = {it["img"].name for it in current_items}
    current_boxes_count = sum(len(it["boxes"]) for it in current_items)
    current_gestures = defaultdict(int)
    for it in current_items:
        current_gestures[it["gesture"]] += len(it["boxes"])

    old_manifest = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
        except Exception:
            old_manifest = None

    print("\n" + "=" * 68)
    if old_manifest:
        old_filenames = set(old_manifest.get("image_filenames", []))
        added_files = current_filenames - old_filenames
        removed_files = old_filenames - current_filenames
        old_boxes = old_manifest.get("total_boxes", 0)
        boxes_diff = current_boxes_count - old_boxes

        print("📊 RAPORT PRZYROSTU ZBIORU DANYCH (DATASET DIFF)")
        print("=" * 68)
        print(f"Poprzednia wersja zbioru : {old_manifest.get('timestamp', 'nieznana')} ({len(old_filenames)} klatek, {old_boxes} rąk)")
        print(f"Nowa wersja zbioru       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({len(current_filenames)} klatek, {current_boxes_count} rąk)")
        print("-" * 68)
        if added_files:
            print(f"🆕 NOWO DODANE ZDJĘCIA  : +{len(added_files)} klatek ({'+' if boxes_diff >= 0 else ''}{boxes_diff} rąk)")
            sample_added = sorted(list(added_files))[:3]
            print(f"   Przykładowe nowe      : {', '.join(sample_added)}{' ...' if len(added_files) > 3 else ''}")
        else:
            print("ℹ️  Brak nowych klatek od ostatniego przygotowania.")

        if removed_files:
            print(f"🗑️  Usunięte klatki       : -{len(removed_files)} klatek")
    else:
        print("📊 RAPORT INICJALIZACJI ZBIORU DANYCH")
        print("=" * 68)
        print("Utworzono początkową bazę datasetu (brak wcześniejszego manifestu).")
        print(f"Łącznie klatek: {len(current_filenames)}, łącznie wykrytych rąk: {current_boxes_count}")

    print("-" * 68)
    print("Rozkład gestów (liczba dłoni w zbiorze):")
    for g in KNOWN_GESTURES:
        if current_gestures[g] > 0:
            print(f"   • {g:<10}: {current_gestures[g]} wycinków dłoni")
    print("=" * 68 + "\n")


def save_manifest(
    current_items: list,
    train_items: list,
    val_items: list,
    total_crops: int,
    manifest_path: Path = MANIFEST_PATH,
    source: str = "ls"
):
    """Zapisuje bieżący stan datasetu do pliku manifestu pod kątem historii i audytu."""
    gestures_breakdown = defaultdict(lambda: {"train": 0, "val": 0, "total": 0})

    for it in train_items:
        g = it["gesture"]
        n_b = len(it["boxes"])
        gestures_breakdown[g]["train"] += n_b
        gestures_breakdown[g]["total"] += n_b

    for it in val_items:
        g = it["gesture"]
        n_b = len(it["boxes"])
        gestures_breakdown[g]["val"] += n_b
        gestures_breakdown[g]["total"] += n_b

    manifest_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "total_images": len(current_items),
        "train_images": len(train_items),
        "val_images": len(val_items),
        "total_boxes": sum(len(it["boxes"]) for it in current_items),
        "total_crops": total_crops,
        "gestures_distribution": dict(gestures_breakdown),
        "image_filenames": sorted([it["img"].name for it in current_items])
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Zaktualizowano historię datasetu w: {manifest_path}")


def prepare_dataset(
    source: str = "ls",
    labeled_dir: Path = LABELED_DIR,
    json_path: Path = None,
    yolo_dir: Path = DEFAULT_YOLO_DIR,
    cnn_dir: Path = DEFAULT_CNN_DIR,
    val_ratio: float = 0.2,
    split_strategy: str = "temporal",
    crop_padding: int = 5,
    sync_ls: bool = True,
    export_zip: bool = False,
    seed: int = 42
):
    """
    Przygotowuje spójny zbiór danych pod dwustopniowy potok:
    1. Etap 1: data/yolo_dataset (dla YOLO Nano - detekcja dłoni)
    2. Etap 2: data/cnn_dataset (dla Custom CNN/ResNet - klasyfikacja gestu na cropach dłoni)
    Oba zbiory posiadają IDENTYCZNY podział train/val, zapobiegając data leakage!
    """
    random.seed(seed)
    yolo_dir = Path(yolo_dir)
    cnn_dir = Path(cnn_dir)
    if json_path is None:
        json_path = ANNOTATIONS_DIR / "dataset.json"

    raw_items = []

    if source == "ls":
        print("🎯 Źródło danych: Label Studio (wyłącznie zatwierdzone zadania 'Submit')")
        tasks = []
        if sync_ls:
            tasks = fetch_submitted_tasks_from_ls(save_to=json_path)
        elif json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                tasks = json.load(f)

        if not tasks:
            print("\n" + "!" * 70)
            print("⚠️  BRAK ZATWIERDZONYCH ZADAŃ W LABEL STUDIO!")
            print("!" * 70)
            print("W Label Studio wgrano klatki z predykcjami, ale żadna nie ma jeszcze")
            print("zatwierdzenia przyciskiem 'Submit' (lub 'Update').")
            print("")
            print("Co zrobić:")
            print("1. Otwórz w przeglądarce: http://localhost:8081")
            print("2. Sprawdź klatki, dopasuj ewentualnie ramkę i kliknij niebieski przycisk 'Submit'.")
            print("3. Uruchom ponownie: make prepare-yolo")
            print("")
            print("Wskazówka: Jeśli chcesz tymczasowo wytrenować model na wszystkich predykcjach")
            print("przed ich ręczną weryfikacją w UI, uruchom:")
            print("   python3 scripts/prepare_yolo_dataset.py --source labeled-dir")
            print("!" * 70 + "\n")
            return None

        raw_items = load_items_from_label_studio(tasks)
    else:
        print(f"🎯 Źródło danych: Katalog surowy predykcji ({labeled_dir})")
        raw_items = load_items_from_labeled_dir(labeled_dir)

    if not raw_items:
        print("❌ Brak poprawnych klatek do utworzenia datasetu.")
        return None

    # Porównaj i wyświetl diff z poprzednią wersją
    compare_and_report_diff(raw_items)

    # Grupowanie według sesji wideo pod kątem bezpiecznego podziału
    session_items = defaultdict(list)
    for it in raw_items:
        s_id = parse_session_from_filename(it["img"].stem)
        session_items[s_id].append(it)

    train_items = []
    val_items = []

    if split_strategy == "temporal":
        print("✂️  Strategia podziału: TEMPORAL (chronologiczny w ramach sesji - brak data leakage)")
        for s_id, items in session_items.items():
            items_sorted = sorted(items, key=lambda x: x["img"].name)
            n_items = len(items_sorted)
            if n_items == 1:
                train_items.append(items_sorted[0])
            else:
                n_val = max(1, int(round(n_items * val_ratio)))
                n_train = n_items - n_val
                train_items.extend(items_sorted[:n_train])
                val_items.extend(items_sorted[n_train:])

    elif split_strategy == "session":
        print("✂️  Strategia podziału: SESSION (grupowy podział całych nagrań)")
        sessions_list = list(session_items.keys())
        random.shuffle(sessions_list)
        n_val_sessions = max(1, int(round(len(sessions_list) * val_ratio)))
        val_session_ids = set(sessions_list[:n_val_sessions])

        for s_id, items in session_items.items():
            if s_id in val_session_ids:
                val_items.extend(items)
            else:
                train_items.extend(items)

    else:
        print("✂️  Strategia podziału: RANDOM")
        all_flattened = [it for items in session_items.values() for it in items]
        random.shuffle(all_flattened)
        n_val = int(round(len(all_flattened) * val_ratio))
        val_items = all_flattened[:n_val]
        train_items = all_flattened[n_val:]

    # =========================================================================
    # 1. GENEROWANIE DATASETU YOLO (ETAP 1: DETEKCJA DŁONI)
    # =========================================================================
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    for split in ["train", "val"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_yolo_split(items, split_name):
        for it in items:
            dest_img = yolo_dir / "images" / split_name / it["img"].name
            dest_lbl = yolo_dir / "labels" / split_name / f"{it['img'].stem}.txt"

            shutil.copy2(it["img"], dest_img)

            if it["is_positive"]:
                yolo_lines = []
                for _, coords in it["boxes"]:
                    c_str = " ".join(f"{c:.6f}" for c in coords)
                    yolo_lines.append(f"0 {c_str}\n")
                with open(dest_lbl, "w", encoding="utf-8") as lf:
                    lf.writelines(yolo_lines)
            else:
                dest_lbl.touch()

    write_yolo_split(train_items, "train")
    write_yolo_split(val_items, "val")

    data_yaml_content = """# Thesis Hand Detection Dataset (Stage 1)
# Źródło: Label Studio (tylko zatwierdzone zadania Submit)

path: .
train: images/train
val: images/val

names:
  0: hand

nc: 1
"""
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(data_yaml_content)

    # =========================================================================
    # 2. GENEROWANIE DATASETU CUSTOM CNN (ETAP 2: KLASYFIKACJA GESTÓW NA CROPACH)
    # =========================================================================
    if cnn_dir.exists():
        shutil.rmtree(cnn_dir)

    crops_metadata_rows = []
    total_crops = 0

    def write_cnn_split(items, split_name):
        nonlocal total_crops
        split_crops = 0

        for it in items:
            if not it["is_positive"]:
                continue

            img = cv2.imread(str(it["img"]))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            stem = it["img"].stem
            gesture = it["gesture"]
            distance = it["distance"]

            gesture_dir = cnn_dir / split_name / gesture
            gesture_dir.mkdir(parents=True, exist_ok=True)

            for b_idx, (_, coords) in enumerate(it["boxes"]):
                cx, cy, bw, bh = coords
                # Przeliczenie ze znormalizowanego formatu YOLO na piksele
                x1 = int(round((cx - bw / 2.0) * w_img))
                y1 = int(round((cy - bh / 2.0) * h_img))
                x2 = int(round((cx + bw / 2.0) * w_img))
                y2 = int(round((cy + bh / 2.0) * h_img))

                # Dodanie marginesu (padding) z zabezpieczeniem granic obrazu
                x1_pad = max(0, x1 - crop_padding)
                y1_pad = max(0, y1 - crop_padding)
                x2_pad = min(w_img, x2 + crop_padding)
                y2_pad = min(h_img, y2 + crop_padding)

                crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size > 0:
                    crop_filename = f"{stem}_box{b_idx}.jpg"
                    crop_dest = gesture_dir / crop_filename
                    cv2.imwrite(str(crop_dest), crop)

                    split_crops += 1
                    total_crops += 1

                    crops_metadata_rows.append({
                        "crop_filename": crop_filename,
                        "split": split_name,
                        "gesture": gesture,
                        "distance": distance,
                        "source_image": it["img"].name,
                        "box_idx": b_idx,
                        "crop_width": crop.shape[1],
                        "crop_height": crop.shape[0],
                        "x1": x1_pad,
                        "y1": y1_pad,
                        "x2": x2_pad,
                        "y2": y2_pad
                    })
        return split_crops

    cnn_train_crops = write_cnn_split(train_items, "train")
    cnn_val_crops = write_cnn_split(val_items, "val")

    # Zapis metadanych cropów (świetne do analiz w pracy inżynierskiej)
    if crops_metadata_rows:
        meta_csv = cnn_dir / "crops_metadata.csv"
        fieldnames = ["crop_filename", "split", "gesture", "distance", "source_image", "box_idx", "crop_width", "crop_height", "x1", "y1", "x2", "y2"]
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(crops_metadata_rows)

    # Zapisz manifest i historię datasetu
    save_manifest(raw_items, train_items, val_items, total_crops, source=source)

    print("\n" + "=" * 68)
    print("✅ PAKIET DWUSTOPNIOWEGO ZBIORU DANYCH ZOSTAŁ PRZYGOTOWANY!")
    print("=" * 68)
    print("1️⃣  ETAP 1: YOLO DETECTOR (Wykrywanie pozycji dłoni):")
    print(f"   📁 Ścieżka            : {yolo_dir}")
    print(f"   📄 Plik data.yaml     : {yaml_path}")
    print(f"   📈 Klatki TRAIN / VAL : {len(train_items)} / {len(val_items)} klatek (łącznie: {len(train_items) + len(val_items)})")
    print("")
    print("2️⃣  ETAP 2: CUSTOM CNN / RESNET (Klasyfikacja gestów):")
    print(f"   📁 Ścieżka (ImageFolder): {cnn_dir}")
    print(f"   📈 Cropy TRAIN / VAL  : {cnn_train_crops} / {cnn_val_crops} wycinków dłoni")
    print(f"   🏷️  Liczba klas gestów : {len([d for d in (cnn_dir / 'train').iterdir() if d.is_dir()]) if (cnn_dir / 'train').exists() else 0}")
    print(f"   📊 Tabela metadanych  : {cnn_dir / 'crops_metadata.csv'}")
    print("=" * 68)

    # Opcjonalne spakowanie do ZIP pod Colab
    if export_zip:
        yolo_zip = yolo_dir.parent / f"{yolo_dir.name}.zip"
        cnn_zip = cnn_dir.parent / f"{cnn_dir.name}.zip"

        print(f"\n📦 Tworzenie archiwów ZIP pod Google Colab / PC:")
        with zipfile.ZipFile(yolo_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(yolo_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(yolo_dir)
                    zf.write(full_p, arcname=str(rel_p))
        print(f"   • YOLO ZIP : {yolo_zip} ({yolo_zip.stat().st_size / (1024*1024):.2f} MB)")

        with zipfile.ZipFile(cnn_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(cnn_dir):
                for f in files:
                    full_p = Path(root) / f
                    rel_p = full_p.relative_to(cnn_dir)
                    zf.write(full_p, arcname=str(rel_p))
        print(f"   • CNN ZIP  : {cnn_zip} ({cnn_zip.stat().st_size / (1024*1024):.2f} MB)")

    return yolo_dir, cnn_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Przygotowanie spójnego datasetu dla YOLO (Etap 1) i Custom CNN (Etap 2)")
    parser.add_argument(
        "--source",
        type=str,
        choices=["ls", "labeled-dir"],
        default="ls",
        help="Źródło danych: 'ls' (tylko Submit z Label Studio, domyślne) lub 'labeled-dir' (wszystkie z data/labeled)"
    )
    parser.add_argument("--labeled-dir", type=str, default=str(LABELED_DIR), help="Katalog z predykcjami (używany przy --source labeled-dir)")
    parser.add_argument("--json-file", type=str, default=str(ANNOTATIONS_DIR / "dataset.json"), help="Ścieżka do dataset.json")
    parser.add_argument("--yolo-dir", type=str, default=str(DEFAULT_YOLO_DIR), help="Katalog wyjściowy datasetu YOLO")
    parser.add_argument("--cnn-dir", type=str, default=str(DEFAULT_CNN_DIR), help="Katalog wyjściowy datasetu Custom CNN")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Ułamek zbioru walidacyjnego (domyślnie 0.2 = 20%)")
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["temporal", "session", "random"],
        default="temporal",
        help="Strategia podziału: temporal (chronologiczny w ramach sesji), session (całe nagrania), random"
    )
    parser.add_argument("--crop-padding", type=int, default=5, help="Margines w pikselach dodawany do cropa dłoni")
    parser.add_argument("--no-sync", action="store_true", help="Nie pobieraj na nowo z API Label Studio, użyj lokalnego dataset.json")
    parser.add_argument("--zip", action="store_true", help="Spakuj wynikowe zbiory do yolo_dataset.zip i cnn_dataset.zip (pod Colab)")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowości")

    args = parser.parse_args()

    prepare_dataset(
        source=args.source,
        labeled_dir=Path(args.labeled_dir),
        json_path=Path(args.json_file),
        yolo_dir=Path(args.yolo_dir),
        cnn_dir=Path(args.cnn_dir),
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        crop_padding=args.crop_padding,
        sync_ls=not args.no_sync,
        export_zip=args.zip,
        seed=args.seed
    )
