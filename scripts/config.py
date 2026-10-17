import os
import sys
from pathlib import Path

# Ustal katalog główny projektu niezależnie od tego, skąd skrypt jest uruchamiany
BASE_DIR = Path(__file__).resolve().parent.parent

# Proste ładowanie .env bez wymuszania zewnętrznych bibliotek
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val

# Ścieżki katalogów danych
raw_data_dir = os.getenv("DATA_DIR", "data")
DATA_DIR = (BASE_DIR / raw_data_dir).resolve() if not Path(raw_data_dir).is_absolute() else Path(raw_data_dir)

SOURCE_FRAMES_DIR = DATA_DIR / "source_frames"
RAW_DIR = DATA_DIR / "raw"
RAW_UPSCALED_DIR = DATA_DIR / "raw_upscaled"
LABELED_DIR = DATA_DIR / "labeled"
LABELED_UPSCALED_DIR = DATA_DIR / "labeled_upscaled"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_UPSCALED_DIR = DATA_DIR / "processed_upscaled"
WEIGHTS_DIR = BASE_DIR / "weights"

# Konfiguracja Label Studio
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8081").rstrip("/")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN", "")
PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

# Ścieżka przechowywania plików w kontenerze Label Studio
LS_STORAGE_PATH = "/opt/data/labeled"

# Szablon interfejsu adnotacji dla rąk w Label Studio
LABEL_STUDIO_XML_CONFIG = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="hand" background="#27ae60"/>
  </RectangleLabels>
</View>"""


def get_auth_headers():
    if not LABEL_STUDIO_TOKEN:
        raise ValueError(
            "Brak tokena Label Studio! Ustaw zmienną LABEL_STUDIO_TOKEN w pliku .env "
            "lub w zmiennych środowiskowych."
        )
    return {
        "Authorization": f"Token {LABEL_STUDIO_TOKEN}",
        "Content-Type": "application/json"
    }
