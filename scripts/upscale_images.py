import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RAW_DIR, RAW_UPSCALED_DIR, WEIGHTS_DIR, BASE_DIR


REALESR_GENERAL_X4V3_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
)


class SRVGGNetCompact(nn.Module):
    """
    SRVGGNetCompact - ultralekka sieć konwolucyjna typu VGG używana w Real-ESRGAN Compact.
    Wyjątkowo szybka (~1.2M wag), dedykowana do rzeczywistych zdjęć (odszumianie + rekonstrukcja).
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu"
    ):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat)
        ])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def ensure_weights_exist(weights_path: Path, url: str = REALESR_GENERAL_X4V3_URL) -> Path:
    """Pobiera wagi modelu jeśli nie istnieją lokalnie."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights_path.exists():
        print(f"[Upscaler] Brak wag modelu w {weights_path}. Pobieranie z oficjalnego repozytorium...")
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"[Upscaler] Pobrano pomyślnie ({weights_path.stat().st_size / (1024*1024):.2f} MB).")
    return weights_path


def select_best_device(requested_device: str = "auto") -> torch.device:
    """Wykrywa najlepsze dostępne urządzenie: MPS (Apple Silicon) -> CUDA -> CPU."""
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class ImageUpscaler:
    """Klasa zarządzająca wczytywaniem modelu i wnioskowaniem super-rozdzielczości."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        scale: int = 4,
        device: str = "auto"
    ):
        self.scale = scale
        self.device = select_best_device(device)
        print(f"[Upscaler] Inicjalizacja akceleratora: {self.device.type.upper()}")

        if weights_path is None:
            weights_path = WEIGHTS_DIR / "realesr-general-x4v3.pth"
        self.weights_path = ensure_weights_exist(weights_path)

        # Wczytanie architektury i wag
        self.model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu"
        )

        checkpoint = torch.load(str(self.weights_path), map_location="cpu")
        key = "params_ema" if "params_ema" in checkpoint else "params"
        state_dict = checkpoint[key] if key in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        # Warmup modelu na urządzeniu
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64, device=self.device)
            _ = self.model(dummy)
            if self.device.type == "mps":
                torch.mps.synchronize()

    def upscale_np(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Zwiększa rozdzielczość pojedynczego obrazu numpy BGR [H, W, C].
        Zwraca przeskalowany obraz BGR [H*scale, W*scale, C] w uint8.
        """
        h_orig, w_orig = img_bgr.shape[:2]

        # Konwersja BGR -> RGB i normalizacja do [0, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            if self.device.type == "mps":
                torch.mps.synchronize()

        # Konwersja z powrotem do numpy BGR
        out_np = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out_bgr = cv2.cvtColor((out_np * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Jeśli wybrano scale=2 zamiast natywnego 4x, przeskalowujemy wysokiej jakości filtrem
        if self.scale != 4:
            target_w = int(w_orig * self.scale)
            target_h = int(h_orig * self.scale)
            out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return out_bgr

    def upscale_file(self, input_file: Path, output_file: Path) -> float:
        """Przeskalowuje pojedynczy plik graficzny i zwraca czas wykonania w sekundach."""
        img = cv2.imread(str(input_file))
        if img is None:
            raise ValueError(f"Nie można wczytać pliku: {input_file}")

        t0 = time.perf_counter()
        upscaled = self.upscale_np(img)
        elapsed = time.perf_counter() - t0

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), upscaled)
        return elapsed

    def upscale_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    ) -> dict:
        """
        Przeskalowuje wszystkie obrazy z podkatalogów, zachowując strukturę ścieżek.
        Zwraca statystyki wydajności.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]
        total_images = len(image_paths)

        if total_images == 0:
            print(f"[Upscaler] Nie znaleziono żadnych obrazów w katalogu {input_dir}")
            return {"total": 0, "fps": 0.0, "total_time": 0.0}

        print(f"[Upscaler] Rozpoczynanie upscalingu {total_images} obrazów (skala {self.scale}x)...")
        print(f"[Upscaler] Wejście: {input_dir} -> Wyjście: {output_dir}")

        start_all = time.perf_counter()
        times: List[float] = []

        for idx, img_path in enumerate(image_paths, start=1):
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path

            t_img = self.upscale_file(img_path, out_path)
            times.append(t_img)

            if idx % 10 == 0 or idx == total_images:
                cur_fps = idx / (time.perf_counter() - start_all)
                print(f"  [Postęp: {idx}/{total_images}] Prędkość: {cur_fps:.2f} FPS (ostatni: {t_img*1000:.1f} ms)")

        total_time = time.perf_counter() - start_all
        avg_fps = total_images / total_time if total_time > 0 else 0.0

        print(f"\n[Upscaler] Zakończono!")
        print(f"  Przetworzono obrazów: {total_images}")
        print(f"  Łączny czas:          {total_time:.2f} s")
        print(f"  Średnia przepustowość: {avg_fps:.2f} FPS ({1000/avg_fps:.1f} ms / obraz)")

        return {
            "total": total_images,
            "total_time": total_time,
            "fps": avg_fps,
            "avg_ms": 1000 / avg_fps if avg_fps > 0 else 0.0
        }


def main():
    parser = argparse.ArgumentParser(description="Szybki upscaling obrazów przy użyciu Real-ESRGAN Compact AI")
    parser.add_argument("--input", "-i", type=str, default=str(RAW_DIR), help="Plik wejściowy lub katalog")
    parser.add_argument("--output", "-o", type=str, default=str(RAW_UPSCALED_DIR), help="Plik wyjściowy lub katalog")
    parser.add_argument("--scale", "-s", type=int, choices=[2, 4], default=4, help="Współczynnik skali (2 lub 4)")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"], help="Urządzenie obliczeniowe")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Ścieżka do wag modelu")
    parser.add_argument("--benchmark", action="store_true", help="Uruchom tryb testowy wydajności (benchmark)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights) if args.weights else None

    upscaler = ImageUpscaler(weights_path=weights_path, scale=args.scale, device=args.device)

    if args.benchmark or input_path.is_file():
        if input_path.is_dir():
            sample_files = [p for p in input_path.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            if not sample_files:
                print("Brak obrazów do benchmarku!")
                sys.exit(1)
            test_file = sample_files[0]
        else:
            test_file = input_path

        print(f"\n[Benchmark] Testowanie wydajności na pliku: {test_file}")
        img = cv2.imread(str(test_file))
        h, w = img.shape[:2]
        print(f"[Benchmark] Wymiary wejściowe: {w}x{h}")

        # Warmup
        _ = upscaler.upscale_np(img)

        runs = 20
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            res = upscaler.upscale_np(img)
            times.append(time.perf_counter() - t0)

        avg_t = sum(times) / len(times)
        fps = 1.0 / avg_t
        h_out, w_out = res.shape[:2]
        print(f"[Benchmark] Wymiary wyjściowe: {w_out}x{h_out}")
        print(f"[Benchmark] Średni czas:       {avg_t * 1000:.2f} ms")
        print(f"[Benchmark] Przepustowość:     {fps:.2f} FPS")

        if not input_path.is_dir():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), res)
            print(f"[Benchmark] Zapisano wynikowy obraz do: {output_path}")
    else:
        upscaler.upscale_folder(input_path, output_path)


if __name__ == "__main__":
    main()
