from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from .dataset import get_device


def export_model(
    model: nn.Module,
    output_dir: str = "weights",
    model_name: str = "gesture_cnn",
    input_size: Tuple[int, int, int, int] = (1, 3, 128, 128),
    device: torch.device = None,
):
    """
    Eksportuje model do formatów produkcyjnych dla drona:
    1. TorchScript (.torchscript.pt)
    2. ONNX (.onnx)
    """
    dev = device or get_device()
    model.eval()
    model = model.to(dev)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_size).to(dev)

    # 1. TorchScript
    ts_file = out_path / f"{model_name}.torchscript.pt"
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(ts_file))
        print(f"✅ Zapisano TorchScript : {ts_file} ({ts_file.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"⚠️ Błąd eksportu TorchScript: {e}")

    # 2. ONNX
    onnx_file = out_path / f"{model_name}.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_crop"],
            output_names=["gesture_logits"],
            dynamic_axes={"input_crop": {0: "batch_size"}, "gesture_logits": {0: "batch_size"}},
        )
        print(f"✅ Zapisano ONNX        : {onnx_file} ({onnx_file.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"⚠️ Błąd eksportu ONNX: {e}")
