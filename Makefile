.PHONY: help setup ls-up ls-down ls-logs pull push detect detect-upscaled import-ls export-ls crop crop-upscaled upscale upscale-benchmark pipeline-standard pipeline-upscaled pipeline-all pipeline status

help:
	@echo "======================================================================"
	@echo "              Thesis Dataset Automation & Pipeline                    "
	@echo "======================================================================"
	@echo "Zarzadzanie uslugami:"
	@echo "  make ls-up              - Uruchamia Label Studio w tle (Docker)"
	@echo "  make ls-down            - Zatrzymuje Label Studio"
	@echo "  make ls-logs            - Pokazuje logi Label Studio"
	@echo ""
	@echo "Wariant 1: Potok STANDARDOWY (BEZ upscalingu - baseline):"
	@echo "  make detect             - Wykrywa dlonie na data/raw -> data/labeled"
	@echo "  make crop               - Wytnij wycinki dloni -> data/processed"
	@echo "  make pipeline-standard  - Uruchamia caly potok standardowy (detect -> crop)"
	@echo ""
	@echo "Wariant 2: Potok AI (Z UPSCALINGIEM - super-resolution):"
	@echo "  make upscale            - Przeskaluj obrazy AI: data/raw -> data/raw_upscaled"
	@echo "  make upscale-benchmark  - Szybki test FPS i czasu inferencji na klatce"
	@echo "  make detect-upscaled    - Wykrywa dlonie na data/raw_upscaled -> data/labeled_upscaled"
	@echo "  make crop-upscaled      - Wytnij wycinki dloni -> data/processed_upscaled"
	@echo "  make pipeline-upscaled  - Uruchamia potok AI (upscale -> detect -> crop)"
	@echo ""
	@echo "Wspolne zadania potoku i Label Studio:"
	@echo "  make import-ls          - Wgrywa zdjecia z predykcjami do Label Studio"
	@echo "  make export-ls          - Pobiera zatwierdzone adnotacje z Label Studio"
	@echo "  make pipeline-all       - Przelicza oba potoki naraz (DVC repro)"
	@echo "  make status             - Sprawdza stan potoku i plikow DVC"
	@echo ""
	@echo "Wersjonowanie i synchronizacja (Google Drive):"
	@echo "  make pull               - Pobiera dane z chmury (dvc pull)"
	@echo "  make push               - Wysyla dane na Google Drive (dvc push)"
	@echo "======================================================================"

setup:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Utworzono .env na podstawie .env.example"; else echo ".env juz istnieje"; fi

ls-up:
	docker compose up -d

ls-down:
	docker compose down

ls-logs:
	docker compose logs -f

pull:
	dvc pull

push:
	dvc push

# --- Wariant 1: Standardowy ---
detect:
	dvc repro detect

crop:
	dvc repro crop

pipeline-standard:
	dvc repro detect crop

# --- Wariant 2: Z upscalingiem ---
upscale:
	dvc repro upscale

upscale-benchmark:
	python3 scripts/upscale_images.py --benchmark

detect-upscaled:
	dvc repro detect_upscaled

crop-upscaled:
	dvc repro crop_upscaled

pipeline-upscaled:
	dvc repro upscale detect_upscaled crop_upscaled

# --- Wspolne / Pelne ---
import-ls:
	python3 scripts/import_to_ls.py

export-ls:
	python3 scripts/process_annotations.py --export-only

pipeline-all:
	dvc repro

pipeline:
	dvc repro

status:
	dvc status
