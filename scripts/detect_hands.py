import os
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import numpy as np


class OneHandGroundingDino(GroundingDINO):
    def predict(self,input):
        detections = super().predict(input)
        
        if len(detections)<=2:
            return detections
        
        best_indices = np.argsort(detections.confidence)[-2:][::-1]
        return detections[best_indices]
    
    
if __name__ == "__main__":
    ontology = CaptionOntology({
        "hand" : "hand"
    })
    print("Loading Grounding DINO...")
    base_model = OneHandGroundingDino(ontology=ontology)
    
    input_dir = "/opt/airflow/data/raw"
    output_dir = "/opt/airflow/data/labeled"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting detection in dir: {input_dir}")
    dataset = base_model.label(
        input_folder=input_dir,
        extension=".png",
        output_folder=output_dir,
    )
    print(f"Finished labeling, results saved in {output_dir}")