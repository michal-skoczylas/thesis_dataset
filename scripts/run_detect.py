import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def find_python_with_ml():
    """Wyszukuje interpreter Pythona z pakietem autodistill_grounding_dino."""
    # 1. Sprawdź bieżący interpreter
    try:
        import autodistill_grounding_dino  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    # 2. Sprawdź znane lokalne środowiska Conda na maszynie (np. hand_autolabel)
    home = Path.home()
    candidates = [
        Path("/opt/homebrew/Caskroom/miniforge/base/envs/hand_autolabel/bin/python"),
        home / "miniforge3/envs/hand_autolabel/bin/python",
        home / "miniconda3/envs/hand_autolabel/bin/python",
        home / ".conda/envs/hand_autolabel/bin/python",
    ]

    # Sprawdź również katalog envs względem aktywnego sys.executable
    curr_bin = Path(sys.executable)
    if "envs" in curr_bin.parts:
        base_envs = curr_bin.parents[1]
        candidates.append(base_envs / "hand_autolabel" / "bin" / "python")
    else:
        candidates.append(curr_bin.parent.parent / "envs" / "hand_autolabel" / "bin" / "python")

    for cand in candidates:
        if cand.exists() and os.access(str(cand), os.X_OK):
            # Zweryfikuj czy ten interpreter ma moduł
            test_cmd = [str(cand), "-c", "import autodistill_grounding_dino"]
            res = subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return str(cand)

    return None


def main():
    extra_args = sys.argv[1:]
    ml_python = find_python_with_ml()

    if ml_python:
        print(f"[Runner] Wykryto lokalne środowisko Pythona z obsługą GPU (MPS/CUDA): {ml_python}")
        cmd = [ml_python, str(BASE_DIR / "scripts" / "detect_hands.py")] + extra_args
    else:
        print("[Runner] Brak lokalnego pakietu ML. Uruchamianie w kontenerze Docker (model-ml)...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{BASE_DIR}:/workspace",
            "-w", "/workspace",
            "model-ml",
            "python", "scripts/detect_hands.py"
        ] + extra_args

    print(f"[Runner] Wykonuję: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
