.PHONY: help setup ls-up ls-down ls-logs pull push sample sample-dry-run to-gray to-gray-inplace detect detect-upscaled import-ls export-ls crop crop-upscaled upscale upscale-benchmark pipeline-standard pipeline-upscaled pipeline-all pipeline status

help:
	@echo "======================================================================"
	@echo "              Thesis Dataset Automation & Pipeline                    "
	@echo "======================================================================"
	@echo "Zarzadzanie uslugami:"
	@echo "  make ls-up              - Uruchamia Label Studio w tle (Docker)"
	@echo "  make ls-down            - Zatrzymuje Label Studio"
	@echo "  make ls-logs            - Pokazuje logi Label Studio"
	@echo ""
	@echo "Przygotowanie danych (Sampling i Normalizacja):"
	@echo "  make sample             - Próbkuje co N-tą klatkę z wybranej kamery (camera_rgb / camera_raw)"
	@echo "                            Przykład: make sample INPUT=/sciezka CAMERA=camera_rgb STRIDE=10"
	@echo "  make sample-dry-run     - Symulacja próbkowania (podgląd bez kopiowania plików)"
	@echo "  make to-gray            - Wykrywa RGB/RAW i konwertuje na czysty czarno-biały (usuwa kropki)"
	@echo "                            Przykład: make to-gray INPUT=data/raw OUTPUT=data/raw_gray"
	@echo "  make to-gray-inplace    - Konwertuje na czarno-biały w miejscu (nadpisuje pliki)"
	@echo "                            Przykład: make to-gray-inplace INPUT=data/raw"
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
	@echo "  make pull               - Pobiera dane z chmury (dvc pull)"
	@echo "  make push               - Wysyla dane na Google Drive (dvc push)"
	@echo ""
	@echo "Trening YOLO (Etap 1 - Detektor Dloni):"
	@echo "  make prepare-yolo       - Dzieli dane na train/val (GroupSplit po sesjach)"
	@echo "  make prepare-yolo-zip   - Przygotowuje dane i pakuje do yolo_dataset.zip (pod Colab)"
	@echo "  make train-yolo         - Rozpoczyna trening YOLO Nano (auto: CUDA/MPS/CPU)"
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

# --- Przygotowanie danych (Sampling i Konwersja) ---
sample:
	python3 scripts/sample_frames.py $(if $(STRIDE),--stride $(STRIDE),) $(if $(CAMERA),--camera $(CAMERA),) $(if $(filter 1 true,$(TO_GRAY)),--to-gray,) $(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),) $(if $(OFFSET),--offset $(OFFSET),)

sample-dry-run:
	python3 scripts/sample_frames.py --dry-run $(if $(STRIDE),--stride $(STRIDE),) $(if $(CAMERA),--camera $(CAMERA),) $(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),)

to-gray:
	python3 scripts/convert_grayscale.py $(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),)

to-gray-inplace:
	python3 scripts/convert_grayscale.py --in-place $(if $(INPUT),--input $(INPUT),)

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

# --- Trening YOLO (Etap 1: Detektor dłoni) ---
prepare-yolo:
	python3 scripts/prepare_yolo_dataset.py $(if $(SOURCE),--source $(SOURCE),) $(if $(SPLIT),--split-strategy $(SPLIT),) $(if $(RATIO),--val-ratio $(RATIO),)

prepare-yolo-zip:
	python3 scripts/prepare_yolo_dataset.py --zip $(if $(SOURCE),--source $(SOURCE),) $(if $(SPLIT),--split-strategy $(SPLIT),) $(if $(RATIO),--val-ratio $(RATIO),)

train-yolo:
	python3 scripts/train_yolo.py $(if $(MODEL),--model $(MODEL),) $(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch $(BATCH),) $(if $(DEVICE),--device $(DEVICE),)
