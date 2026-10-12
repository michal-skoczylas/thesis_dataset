.PHONY: help setup ls-up ls-down ls-logs pull push detect import-ls export-ls crop pipeline status

help:
	@echo "======================================================================"
	@echo "              Thesis Dataset Automation & Pipeline                    "
	@echo "======================================================================"
	@echo "Zarzadzanie uslugami:"
	@echo "  make ls-up       - Uruchamia Label Studio w tle (Docker)"
	@echo "  make ls-down     - Zatrzymuje Label Studio"
	@echo "  make ls-logs     - Pokazuje logi Label Studio"
	@echo ""
	@echo "Przeplyw danych (ML & Adnotacje):"
	@echo "  make detect      - Krok 1: Wykrywa dlonie przez Grounding DINO (DVC)"
	@echo "  make import-ls   - Krok 2: Wgrywa zdjecia z predykcjami do Label Studio"
	@echo "  make export-ls   - Krok 3: Pobiera zatwierdzone adnotacje z Label Studio"
	@echo "  make crop        - Krok 4: Wytnij wycinki dloni (data/processed, DVC)"
	@echo "  make pipeline    - Uruchamia pelny potok DVC (dvc repro)"
	@echo ""
	@echo "Wersjonowanie i synchronizacja (Google Drive):"
	@echo "  make pull        - Pobiera dane z chmury (dvc pull)"
	@echo "  make push        - Wysyla dane na Google Drive (dvc push)"
	@echo "  make status      - Sprawdza stan potoku i plikow DVC"
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

detect:
	dvc repro detect

import-ls:
	python3 scripts/import_to_ls.py

export-ls:
	python3 scripts/process_annotations.py --export-only

crop:
	dvc repro crop

pipeline:
	dvc repro

status:
	dvc status
