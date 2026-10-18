# 🏋️ Trening Modeli: Detektor YOLO Nano i Klasyfikator CNN

Ten przewodnik wyjaśnia, jak uruchomić proces uczenia maszynowego dla obu etapów systemu:
- **Etap 1:** Detektor dłoni (YOLO Nano: `yolo11n.pt` lub `yolov8n.pt`)
- **Etap 2:** Klasyfikator gestów (Custom CNN / ResNet)

---

## 🖥️ Etap 1: Trening Detektora YOLO Nano

Do dyspozycji są dwie opcje treningu: lokalnie na własnej karcie graficznej lub w darmowym środowisku Google Colab.

### Opcja A: Trening Lokalny (PC NVIDIA RTX lub Mac Apple Silicon)

Skrypt `scripts/train_yolo.py` automatycznie wykrywa sprzęt i dobiera właściwy akcelerator:
- **NVIDIA GPU (CUDA):** używa `cuda:0` i optymalizacji Tensor Core.
- **Mac Apple Silicon (M1/M2/M3/M4):** używa `mps` (Metal Performance Shaders).
- **Procesor (CPU):** fallback w przypadku braku akceleratora.

Uruchomienie przez Makefile:
```bash
make train-yolo
```

Lub z własnymi hiperparametrami:
```bash
python3 scripts/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --patience 25 \
  --device auto
```

Po zakończeniu treningu:
- Wykresy uczenia (loss, mAP50, mAP50-95, PR curve) zapisują się w katalogu `runs/detect/hand_detector/`.
- Najlepsze wagi zostają automatycznie skopiowane do:
  ```text
  weights/yolo_hand_detector.pt
  ```

---

### Opcja B: Trening w Google Colab (Darmowe GPU T4 / A100)

Jeśli chcesz trenować model w chmurze bez obciążania własnego komputera:

1. **Przygotowanie paczki danych:**
   W terminalu na swoim komputerze wykonaj:
   ```bash
   make prepare-yolo-zip
   ```
   W katalogu `data/` powstanie plik `yolo_dataset.zip`.
2. **Przesłanie na Dysk Google:**
   Wgraj plik `yolo_dataset.zip` do głównego katalogu swojego Dysku Google (lub do podkatalogu, np. `thesis_data/`).
3. **Uruchomienie Notebooka:**
   Otwórz plik `notebooks/train_yolo_colab.ipynb` w [Google Colab](https://colab.research.google.com).
   - W menu Colaba wybierz: **Środowisko wykonawcze -> Zmień typ środowiska wykonawczego -> T4 GPU**.
   - Uruchom kolejne komórki notebooka.
4. **Co dzieje się w Colabie:**
   - Notebook montuje Twój Dysk Google (`drive.mount('/content/drive')`).
   - Rozpakowuje `yolo_dataset.zip` do szybkiego dysku RAM/SSD Colaba.
   - Uruchamia trening YOLO z akceleracją CUDA (zajmuje ~10–15 minut na 50 epok).
   - Zapisuje plik z najlepszymi wagami `best.pt` bezpośrednio z powrotem na Twój Dysk Google.
5. **Wdrożenie wag lokalnie:**
   Pobierz plik `best.pt` z Dysku Google i umieść go w projekcie jako:
   ```text
   weights/yolo_hand_detector.pt
   ```

---

## ⚡ Włączenie pętli Active Learning

Gdy plik `weights/yolo_hand_detector.pt` znajduje się w katalogu `weights/`, system automatycznie przełącza się w tryb Active Learning:
- Przy następnym uruchomieniu autodetekcji nie musisz już czekać na powolny model Grounding DINO.
- Wykonaj:
  ```bash
  make detect-yolo
  ```
- Nowo wytrenowany model YOLO przetworzy kolejne 1800 klatek z prędkością **50–100 FPS**, od razu nanosząc predykcje i przyspieszając pracę Twoją i Janka o rząd wielkości!

---

## 🖐️ Etap 2: Trening Klasyfikatora Gestów (Custom CNN)

Zbiór wycinków dłoni przygotowany przez `make prepare-yolo` znajduje się w:
```text
data/cnn_dataset/
├── train/
│   ├── closehand/
│   ├── openhand/
│   ├── ok/
│   └── ...
└── val/
    ├── closehand/
    ├── openhand/
    └── ...
```

Format ten jest natywnie obsługiwany przez bibliotekę PyTorch:
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder("data/cnn_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("data/cnn_dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

Dzięki temu, że podział na `train` i `val` został wykonany według tych samych unikalnych sesji co w zbiorze YOLO, wykluczone jest ryzyko przenikania kadrów i zakłamania wyników ewaluacji.
