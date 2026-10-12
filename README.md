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

Proces przygotowania danych dzieli się na 4 proste kroki:

```
[data/raw] (zdjęcia z drona)
     │
     ▼  make detect (lub dvc repro detect)
[data/labeled] (predykcje bounding boxów rąk z Grounding DINO)
     │
     ▼  make import-ls
[Label Studio UI] (weryfikacja / poprawianie ramek przez człowieka)
     │
     ▼  make export-ls
[data/annotations/dataset.json] (zatwierdzone ramki, śledzone w Git)
     │
     ▼  make crop (lub dvc repro crop)
[data/processed] (wycięte dłonie pod trening klasyfikatora gestów)
```

---

## 🛠️ Dostępne polecenia (Makefile)

Wszystkie operacje wykonasz jednym poleceniem:

| Polecenie | Opis |
|-----------|------|
| `make help` | Wyświetla listę wszystkich dostępnych poleceń |
| `make ls-up` | Uruchamia Label Studio w Dockerze (lekki kontener, port 8081) |
| `make ls-down` | Zatrzymuje Label Studio |
| `make pull` | Pobiera najnowszy zbiór danych z Google Drive (`dvc pull`) |
| `make push` | Wysyła wygenerowane dane na Google Drive (`dvc push`) |
| `make detect` | Uruchamia etap detekcji rąk modelem Grounding DINO |
| `make import-ls` | Importuje wykryte ramki i zdjęcia do Label Studio |
| `make export-ls` | Pobiera zatwierdzone przez człowieka adnotacje do JSON |
| `make crop` | Wytłacza i zapisuje wycięte dłonie w `data/processed` |
| `make pipeline` | Uruchamia pełny potok DVC (`dvc repro`) |
| `make status` | Pokazuje status potoku oraz danych DVC |

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
