# 📚 Dokumentacja Projektu: Thesis Dataset Automation & Pipeline

Kompletny zestaw dokumentacji technicznej i operacyjnej dla systemu automatyzacji zbioru danych, dwustopniowej detekcji dłoni (YOLO Nano) oraz klasyfikacji gestów (Custom CNN / ResNet) z drona.

---

## 📑 Spis Dokumentów

1. [**Od A do Z: Pełny Przewodnik po Potoku Danych (Pipeline Guide)**](file:///Users/michalskoczylas/Documents/studia/thesis/dane/docs/pipeline_guide.md)
   - Przewodnik krok po kroku przez cały proces: od surowych nagrań wideo z drona, przez sampling, autolabeling, weryfikację w 2 osoby, aż po trening modeli i Active Learning.

2. [**Jak z Label Studio (Submit) powstaje zbiór treningowy YOLO i CNN**](file:///Users/michalskoczylas/Documents/studia/thesis/dane/docs/label_studio_export_to_dataset.md)
   - Szczegółowe wyjaśnienie mechanizmu eksportu: dlaczego tylko zatwierdzone zadania (`Submit`) wchodzą do zbioru, jak działa temporal split (ochrona przed data leakage), jak generowane są wycinki dla CNN oraz jak działa śledzenie przyrostu danych (diff manifest).

3. [**Kooperacja i Etykietowanie we dwoje (Michał & Janek)**](file:///Users/michalskoczylas/Documents/studia/thesis/dane/docs/coop_annotation.md)
   - Jak bezpiecznie udostępnić Label Studio koledze przez Cloudflare Tunnel bez instalowania Dockera/Pythona, jak konfigurować strategie podziału zadań (`params.yaml`) i jak efektywnie weryfikować klatki.

4. [**Trening Modeli (Lokalnie CUDA/MPS oraz Google Colab)**](file:///Users/michalskoczylas/Documents/studia/thesis/dane/docs/training_and_colab.md)
   - Instrukcja uruchamiania treningu detektora YOLO Nano (lokalny PC z NVIDIA RTX, Mac z Apple Silicon lub darmowy GPU T4 w Google Colab) oraz klasyfikatora CNN.

---

## 🧠 Szybka Ściągawka Komend (Cheat Sheet)

| Cel | Polecenie |
|---|---|
| **Start Label Studio w Dockerze** | `make ls-up` |
| **Udostępnienie koledze (Tunnel)** | `make tunnel` |
| **Próbkowanie klatek z wideo** | `make sample STRIDE=10 CAMERA=camera_rgb` |
| **Konwersja do czarno-białego** | `make to-gray-inplace INPUT=data/raw` |
| **Autodetekcja dłoni (DINO baseline)** | `make detect` |
| **Szybka autodetekcja YOLO (Active Learning)** | `make detect-yolo` |
| **Podział i wgranie do LS (Co-op 50/50)** | `make import-ls-coop` |
| **Budowa datasetu YOLO & CNN (z Submit)** | `make prepare-yolo-zip` |
| **Trening YOLO lokalnie** | `make train-yolo` |
| **Wysłanie danych do Google Drive (DVC)** | `make push` |
| **Pobranie danych na nowej maszynie** | `make pull` |
