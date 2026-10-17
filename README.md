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

## 🔄 Przepływ pracy (Dwa warianty potoku)

System wspiera dwa równoległe potoki badawcze (idealne do porównania w pracy inżynierskiej):

### 1. Wariant STANDARDOWY (Baseline - bez upscalingu)
```
[data/raw] (zdjęcia 320x320)
     │
     ▼  make detect (lub make pipeline-standard)
[data/labeled] (predykcje bounding boxów rąk z Grounding DINO)
     │
     ▼  make crop
[data/processed] (wycięte dłonie z oryginalnych zdjęć)
```

### 2. Wariant AI (Super-Resolution - Real-ESRGAN Compact x4)
```
[data/raw] (zdjęcia 320x320)
     │
     ▼  make upscale (lub make pipeline-upscaled)
[data/raw_upscaled] (wysoka rozdzielczość 1280x1280, MPS / Metal)
     │
     ▼  make detect-upscaled
[data/labeled_upscaled] (detekcja na ostrych klatkach)
     │
     ▼  make crop-upscaled
[data/processed_upscaled] (wycięte dłonie w wysokiej rozdzielczości)
```

---

## 🛠️ Dostępne polecenia (Makefile)

Wszystkie operacje wykonasz jednym poleceniem:

| Polecenie | Opis |
|-----------|------|
| `make help` | Wyświetla listę wszystkich dostępnych poleceń |
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
