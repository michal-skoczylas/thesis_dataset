import os
import argparse
from pathlib import Path
import numpy as np
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

from config import RAW_DIR, LABELED_DIR


class MaxHandsGroundingDino(GroundingDINO):
    def __init__(self, ontology, max_detections=2):
        super().__init__(ontology=ontology)
        self.max_detections = max_detections

    def predict(self, input_image):
        detections = super().predict(input_image)
        if len(detections) <= self.max_detections:
            return detections
        best_indices = np.argsort(detections.confidence)[-self.max_detections:][::-1]
        return detections[best_indices]


def run_detection(input_dir: Path, output_dir: Path, max_hands: int = 2, extension: str = ".png", target_class: str = "hand"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ontology = CaptionOntology({target_class: target_class})
    print(f"Ładowanie modelu Grounding DINO (maksymalnie {max_hands} detekcji na kadr)...")
    base_model = MaxHandsGroundingDino(ontology=ontology, max_detections=max_hands)

    print(f"Rozpoczęcie detekcji w: {input_dir}")
    dataset = base_model.label(
        input_folder=str(input_dir),
        extension=extension,
        output_folder=str(output_dir),
    )

    classes_file = output_dir / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write(f"{target_class}\n")

    print(f"Zakończono pre-etykietowanie! Wyniki zapisano w: {output_dir}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detekcja rąk za pomocą Grounding DINO")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DIR), help="Katalog wejściowy ze zdjęciami")
    parser.add_argument("--output-dir", type=str, default=str(LABELED_DIR), help="Katalog wyjściowy na adnotacje YOLO")
    parser.add_argument("--max-hands", type=int, default=2, help="Maksymalna liczba rąk na obraz")
    parser.add_argument("--extension", type=str, default=".png", help="Rozszerzenie przeszukiwanych plików")
    parser.add_argument("--class-name", type=str, default="hand", help="Nazwa klasy")
    args = parser.parse_args()

    run_detection(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_hands=args.max_hands,
        extension=args.extension,
        target_class=args.class_name
    )