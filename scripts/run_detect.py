import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    # Sprawdź czy zależności ML są zainstalowane w lokalnym środowisku Pythona
    has_local_ml = False
    try:
        import autodistill_grounding_dino
        has_local_ml = True
    except ImportError:
        has_local_ml = False

    extra_args = sys.argv[1:]

    if has_local_ml:
        print("[Runner] Wykryto lokalne środowisko z autodistill_grounding_dino. Uruchamianie lokalne...")
        cmd = [sys.executable, str(BASE_DIR / "scripts" / "detect_hands.py")] + extra_args
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
