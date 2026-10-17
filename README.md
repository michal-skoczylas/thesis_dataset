# 🚁 Thesis Dataset Automation & Pipeline

System wersjonowania danych, automatycznej detekcji rąk i przygotowywania zbioru danych pod pracę inżynierską (gesty dłoni z drona).

---

## ⚡ Szybki start na nowym urządzeniu

Klonowanie i przygotowanie środowiska zajmuje 2 minuty:

```bash
# 1. Klonowanie repozytorium
git clone <url_repozytorium>
cd dane

# 2. Utworzenie pliku konfiguracji środowiska
make setup
# Uzupełnij token w pliku .env (LABEL_STUDIO_TOKEN=...)

# 3. Zainstalowanie zależności Pythona
pip install -r requirements.txt

# 4. Pobranie danych z Google Drive (DVC)
make pull

# 5. Uruchomienie Label Studio (w tle)
make ls-up
```

Label Studio będzie dostępne pod adresem: [http://localhost:8081](http://localhost:8081).

---

## 🔄 Przepływ pracy (Pipeline)

Pełny proces przygotowania danych od surowych serii klatek z drona po gotowy dataset:

```
[Surowe klatki z drona] (np. data/source_frames lub katalog zewnętrzny)
        │
        ▼  make sample (bierz co N-tą klatkę, domyślnie stride=10)
   [data/raw] (wyselekcjonowane, unikalne klatki do datasetu)
        │
   ┌────┴──────────────────────────────────────────┐
   ▼                                               ▼
Wariant 1: STANDARDOWY                     Wariant 2: AI UPSCALED
(make pipeline-standard)                   (make pipeline-upscaled)
   │                                               │
   ▼ make detect                                   ▼ make upscale
[data/labeled]                             [data/raw_upscaled] (1280x1280)
   │                                               │
   ▼ make import-ls -> UI -> export-ls             ▼ make detect-upscaled
[data/annotations/dataset.json]            [data/labeled_upscaled]
   │                                               │
   ▼ make crop                                     ▼ make crop-upscaled
[data/processed] (cropy dłoni)             [data/processed_upscaled] (cropy AI)
```

---

## 🎞️ Krok 0: Próbkowanie klatek z drona (Frame Subsampling)

Kamery dronów rejestrują wideo z prędkością 12, 30 lub 60 kl./s, generując w każdej sesji dwa podfoldery:
- `camera_rgb` (obraz barwny)
- `camera_raw` (surowy bufor sensora z mozaiką Bayera)

Skrypt `scripts/sample_frames.py` pozwala wybrać **konkretną kamerę** (domyślnie `camera_rgb`) i pobrać **co N-tą klatkę** (np. co 10.), całkowicie pomijając klatki z drugiej kamery (brak duplikatów):

### 1. Szybki podgląd (symulacja bez zapisywania plików):
```bash
make sample-dry-run INPUT=/sciezka/do/klatek CAMERA=camera_rgb STRIDE=10
```

### 2. Wykonanie próbkowania do `data/raw`:
```bash
# Z domyślnymi parametrami z params.yaml:
make sample INPUT=/sciezka/do/klatek

# Z wyborem kamery i własnym krokiem:
make sample INPUT=/sciezka/do/klatek CAMERA=camera_rgb STRIDE=10

# Próbkowanie z automatyczną konwersją do czystego czarno-białego w locie:
make sample INPUT=/sciezka/do/klatek CAMERA=camera_rgb STRIDE=10 TO_GRAY=1
```

### 3. Zaawansowane opcje przez skrypt Pythona:
```bash
# Podstawowe z wyborem kamery:
python3 scripts/sample_frames.py --input /sciezka/do/klatek --output data/raw --camera camera_rgb --stride 10

# Oszczędność dysku (tworzy dowiązania symboliczne zamiast kopii plików):
python3 scripts/sample_frames.py --input /sciezka/do/klatek --output data/raw --camera camera_rgb --stride 10 --symlink

# Wyczyszczenie data/raw przed próbkowaniem nowej partii:
python3 scripts/sample_frames.py --input /sciezka/do/klatek --output data/raw --camera camera_rgb --stride 10 --clear
```

> [!TIP]
> **Automatyczne czyszczenie ścieżek:**
> Skrypt automatycznie usuwa zbędny poziom podfolderu `camera_rgb/` z katalogu docelowego, dzięki czemu w `data/raw/` klatki lądują bezpośrednio pod folderem danej sesji (np. `1_4metra/closehand/2026-10-15-14-45-08/00040.jpg`).

---

## 📸 Krok 0.5: Normalizacja do czarno-białego (Usuwanie kropek i konwersja RGB)

Kamery dronów mogą generować dwa rodzaje plików:
1. **RAW z czarnymi kropkami (`camera_raw`):** Surowa mozaika filtrów Bayera (CFA) zapisana jako monochrom. Powoduje efekt czarnego rastru/kropek i drastycznie psuje ostrość.
2. **Kolorowe RGB (`camera_rgb`):** Klatki z silnym zielono-niebieskim zafarbem w sztucznym oświetleniu.

Skrypt `scripts/convert_grayscale.py` automatycznie rozpoznaje format każdego pliku:
- Dla **RAW**: wykonuje debayering (`COLOR_BayerRG2GRAY`), który **całkowicie usuwa czarne kropki** i przywraca gładką, pełną rozdzielczość.
- Dla **RGB**: konwertuje na czysty monochromatyczny obraz.
- Zachowuje pełną strukturę podfolderów i zapisuje obrazy jako 3-kanałowe ($R=G=B$), kompatybilne z YOLO/DINO.

### Uruchomienie:
```bash
# Konwersja w miejscu (nadpisuje pliki w data/raw czystym czarno-białym bez kropek):
make to-gray-inplace INPUT=data/raw

# Lub zapis do nowego folderu:
make to-gray INPUT=data/raw OUTPUT=data/raw_clean
```

---

## 🛠️ Dostępne polecenia (Makefile)

Wszystkie operacje wykonasz jednym poleceniem:

| Polecenie | Opis |
|-----------|------|
| `make help` | Wyświetla listę wszystkich dostępnych poleceń |
| `make sample` | Próbkuje klatki z drona (co N-tą klatkę, domyślnie z `params.yaml`) |
| `make sample-dry-run` | Symulacja próbkowania (podgląd w terminalu bez kopiowania plików) |
| `make to-gray-inplace` | Konwertuje na czarno-biały w miejscu (usuwa czarne kropki i zafarb RGB) |
| `make to-gray` | Konwertuje do nowego folderu (`INPUT=... OUTPUT=...`) |
| `make pipeline-standard` | Uruchamia pełny potok standardowy (`detect` $\to$ `crop`) |
| `make pipeline-upscaled` | Uruchamia pełny potok AI (`upscale` $\to$ `detect_upscaled` $\to$ `crop_upscaled`) |
| `make pipeline-all` | Przelicza oba potoki równolegle w DVC |
| `make upscale-benchmark` | Szybki test FPS i czasu inferencji upscalingu na Apple Silicon |
| `make ls-up` | Uruchamia Label Studio w Dockerze (port 8081) |
| `make ls-down` | Zatrzymuje Label Studio |
| `make import-ls` | Importuje wykryte ramki i zdjęcia do Label Studio |
| `make export-ls` | Pobiera zatwierdzone przez człowieka adnotacje do JSON |
| `make pull` | Pobiera najnowszy zbiór danych z Google Drive (`dvc pull`) |
| `make push` | Wysyła wygenerowane dane na Google Drive (`dvc push`) |
| `make status` | Pokazuje status potoków oraz danych DVC |

---

## 📦 Wersjonowanie danych (DVC)

Zbiory danych są wersjonowane przez **DVC** i powiązane z commitami w **Git**:
* `data/raw.dvc` – wersjonuje oryginalne klatki z drona.
* `dvc.lock` – automatycznie blokuje i wersjonuje:
  * `data/raw`
  * `data/labeled` (predykcje)
  * `data/processed` (wycięte dłonie)
  * kod skryptów oraz parametry z `params.yaml`

Gdy wygenerujesz nową wersję zbioru lub dodasz nowe zdjęcia:
```bash
make pipeline
make push
git add dvc.lock data/annotations/dataset.json
git commit -m "Zaktualizowano dataset po adnotacji nowej partii zdjęć"
git push
```

Na innym komputerze wystarczy:
```bash
git pull
make pull
```
I masz natychmiast identyczny stan danych!
