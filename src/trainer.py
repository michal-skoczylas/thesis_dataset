import time
import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import get_device


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 40,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    experiment_name: str = "custom_cnn_v1",
    save_dir: Optional[str] = None,
    log_dir: str = "runs/gesture_cnn",
    device: Optional[torch.device] = None,
    live_plot: bool = True,
) -> nn.Module:
    """
    Uniwersalny silnik treningowy PyTorch zintegrowany z TensorBoardem i wykresem na żywo.

    - Rysuje wykres Loss & Accuracy na żywo w komórce notatnika (live_plot=True).
    - Równolegle loguje wszystkie skalary do TensorBoarda (runs/gesture_cnn).
    - Zapisuje najlepszy checkpoint do folderu weights/.
    """
    dev = device or get_device()
    model = model.to(dev)

    # Ścieżka zapisu wag
    if save_dir is None:
        save_path = Path("weights")
        if not save_path.exists() and Path("../weights").exists():
            save_path = Path("../weights")
    else:
        save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_weights_file = save_path / f"{experiment_name}_best.pt"

    # TensorBoard writer
    tb_log_path = Path(log_dir) / experiment_name
    writer = SummaryWriter(log_dir=str(tb_log_path))
    print(f"🚀 Rozpoczynam trening '{experiment_name}' na: {dev}")
    print(f"📊 Logi TensorBoard: {tb_log_path}")

    # Rejestracja grafu obliczeniowego
    try:
        sample_batch, _ = next(iter(val_loader))
        writer.add_graph(model, sample_batch.to(dev))
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.1)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "log_table": []
    }

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # 1. FAZA TRENINGOWA
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(dev), targets.to(dev)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == targets.data).item()
            train_total += inputs.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # 2. FAZA WALIDACYJNA
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(dev), targets.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == targets.data).item()
                val_total += inputs.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Zapis do historii
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        # 3. REJESTRACJA W TENSORBOARD
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_train_acc, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # 4. MONITOROWANIE NAJLEPSZEGO WYNIKU
        marker = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, best_weights_file)
            marker = "⭐ Best!"

        log_line = (
            f"{epoch:^7} | {epoch_train_loss:^10.4f} | {epoch_train_acc*100:^9.2f}% | "
            f"{epoch_val_loss:^10.4f} | {epoch_val_acc*100:^9.2f}% | {current_lr:^9.6f} {marker}"
        )
        history["log_table"].append(log_line)

        # 5. RYSOWANIE WYKRESU NA ŻYWO W KOMÓRCE JUPYTERA
        if live_plot:
            try:
                from IPython.display import clear_output
                import matplotlib.pyplot as plt

                clear_output(wait=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

                epochs_range = range(1, len(history["train_loss"]) + 1)

                # Wykres Loss
                ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
                ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
                ax1.set_title(f"Loss: {experiment_name} (Epoka {epoch}/{num_epochs})", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Epoka")
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Wykres Accuracy
                ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
                ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#d62728", lw=2)
                ax2.set_title(f"Accuracy (Max Val: {best_val_acc*100:.1f}%)", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Epoka")
                ax2.set_ylabel("%")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                plt.tight_layout()
                plt.show()

                # Wyświetlanie ostatnich wierszy tabeli
                print(f"{'Epoka':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^9}")
                print("-" * 70)
                for line in history["log_table"][-6:]:
                    print(line)
            except Exception:
                if epoch % 5 == 0 or epoch == num_epochs or marker:
                    print(log_line)
        else:
            if epoch % 5 == 0 or epoch == num_epochs or marker:
                print(log_line)

    time_elapsed = time.time() - start_time
    print("-" * 70)
    print(f"🏁 Trening ukończony w: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Najwyższy Val Accuracy: {best_val_acc * 100:.2f}%")
    print(f"💾 Najlepsze wagi zapisano w: {best_weights_file}")

    writer.close()
    model.load_state_dict(best_model_weights)
    return model


def sanity_check_overfit(
    model: nn.Module,
    data_loader: DataLoader,
    batch_size: int = 16,
    num_epochs: int = 80,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
    live_plot: bool = True,
    target_accuracy: float = 0.99,
    target_loss: float = 0.05,
    clone_model: bool = True,
) -> bool:
    """
    Sanity Check: Przeuczanie modelu na 1 wybranym batchu (Karpathy Recipe).

    Pobiera dokładnie 1 batch (np. 16 obrazów), wyłącza regularyzację (Dropout p=0,
    weight_decay=0) i trenuje sieć wyłącznie na tych samych próbkach przez N epok.

    Cel:
        Train Loss powinien spaść do ~0.0, a Train Accuracy osiągnąć 100%.

    Parametry:
        model: badana architektura sieci PyTorch
        data_loader: DataLoader ze zbiorem (np. train_loader lub val_loader)
        batch_size: liczba próbek w pojedynczym batchu do zapamiętania (np. 16 lub 32)
        num_epochs: liczba iteracji na tym jednym batchu (domyślnie 80)
        lr: learning rate dla optymalizatora Adam (domyślnie 0.001)
        device: CPU / MPS / CUDA
        live_plot: czy rysować wykres zbieżności na żywo w komórce
        target_accuracy: próg dokładności uznający test za zdany (domyślnie 0.99 = 100%)
        target_loss: próg funkcji straty (domyślnie < 0.05)
        clone_model: jeśli True, testuje kopię modelu, nie niszcząc wag oryginału

    Zwraca:
        bool: True jeśli sanity check zakończył się sukcesem, False w przeciwnym razie.
    """
    dev = device or get_device()
    test_model = copy.deepcopy(model).to(dev) if clone_model else model.to(dev)

    # 1. Wyciągnięcie dokładnie jednego batcha ze zbioru
    sample_images, sample_labels = next(iter(data_loader))
    if sample_images.size(0) > batch_size:
        sample_images = sample_images[:batch_size]
        sample_labels = sample_labels[:batch_size]

    sample_images = sample_images.to(dev)
    sample_labels = sample_labels.to(dev)
    actual_batch_size = sample_images.size(0)

    # 2. Wyłączenie regularyzacji (Dropout) dla deterministycznego zapamiętywania
    saved_dropouts = {}
    for name, module in test_model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            saved_dropouts[name] = module.p
            module.p = 0.0

    # 3. Konfiguracja optymalizatora bez weight decay (brak kary L2)
    optimizer = optim.Adam(test_model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []
    history_logs = []

    print(f"\n======================================================================")
    print(f"🧪 [SANITY CHECK] Przeuczanie modelu na 1 batchu ({actual_batch_size} obrazów)")
    print(f"   Architektura : {model.__class__.__name__}")
    print(f"   Urządzenie   : {dev} | Epoki: {num_epochs} | LR: {lr}")
    print(f"   Cel          : Loss < {target_loss} oraz Accuracy >= {target_accuracy * 100:.0f}%")
    print(f"======================================================================\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        test_model.train()
        optimizer.zero_grad()

        outputs = test_model(sample_images)
        loss = criterion(outputs, sample_labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        acc = (preds == sample_labels).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        log_line = f"Epoka {epoch:3d}/{num_epochs:3d} | Loss: {loss.item():.5f} | Accuracy: {acc * 100:6.2f}%"
        history_logs.append(log_line)

        # Rysowanie wykresu na żywo w komórce notatnika
        if live_plot:
            try:
                from IPython.display import clear_output
                import matplotlib.pyplot as plt

                clear_output(wait=True)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

                epochs_range = range(1, len(losses) + 1)

                # Wykres Loss
                ax1.plot(epochs_range, losses, color="#1f77b4", lw=2, label="Train Loss (1 batch)")
                ax1.axhline(target_loss, color="green", linestyle="--", alpha=0.7, label=f"Target (<{target_loss})")
                ax1.set_title(f"Loss: {model.__class__.__name__} (1 batch, {actual_batch_size} img)", fontsize=11, fontweight="bold")
                ax1.set_xlabel("Epoka")
                ax1.set_ylabel("CrossEntropy Loss")
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Wykres Accuracy
                ax2.plot(epochs_range, [a * 100 for a in accuracies], color="#2ca02c", lw=2, label="Train Accuracy")
                ax2.axhline(target_accuracy * 100, color="green", linestyle="--", alpha=0.7, label=f"Target ({target_accuracy * 100:.0f}%)")
                ax2.set_title(f"Accuracy: {acc * 100:.1f}% (Cel: 100%)", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Epoka")
                ax2.set_ylabel("%")
                ax2.set_ylim(-5, 105)
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                plt.tight_layout()
                plt.show()

                print("-" * 55)
                for line in history_logs[-5:]:
                    print(line)
            except Exception:
                if epoch % 10 == 0 or epoch == num_epochs:
                    print(log_line)
        else:
            if epoch % 10 == 0 or epoch == num_epochs:
                print(log_line)

    # Przywrócenie pierwotnych wartości Dropout
    for name, module in test_model.named_modules():
        if name in saved_dropouts:
            module.p = saved_dropouts[name]

    elapsed = time.time() - start_time
    final_loss = losses[-1]
    final_acc = accuracies[-1]
    passed = (final_acc >= target_accuracy) and (final_loss <= target_loss)

    print("\n" + "=" * 70)
    if passed:
        print(f"✅ [SANITY CHECK PASSED] Sukces! Czas: {elapsed:.1f}s")
        print(f"   Wyniki: Loss = {final_loss:.5f} | Accuracy = {final_acc * 100:.2f}%")
        print(f"   Model pomyślnie przeuczył się na 1 batchu do 100%!")
        print(f"   Wnioski: Przepływ gradientów, rozmiary tensorów, etykiety i funkcja straty")
        print(f"            działają w 100% poprawnie. Możesz bezpiecznie trenować pełny zbiór!")
    else:
        print(f"❌ [SANITY CHECK FAILED] Model NIE osiągnął 100% dokładności! Czas: {elapsed:.1f}s")
        print(f"   Wyniki: Loss = {final_loss:.5f} (oczekiwano < {target_loss}) | Accuracy = {final_acc * 100:.2f}%")
        print(f"\n🔍 Możliwe przyczyny problemu:")
        print(f"   1. Za mały / za duży Learning Rate (obecny: {lr}) -> spróbuj lr=1e-3 lub lr=5e-4.")
        print(f"   2. Zablokowany przepływ gradientu lub odcięte warstwy (np. brak parametrów w optimizerze).")
        print(f"   3. Zbyt mała pojemność sieci lub martwe neurony ReLU.")
        print(f"   4. Błąd w normalizacji wejścia lub mapowaniu etykiet.")
    print("=" * 70 + "\n")

    return passed

