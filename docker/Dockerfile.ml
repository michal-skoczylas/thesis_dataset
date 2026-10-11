FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/airflow
# Instalacja PyTorch i wymaganych pakietów ML
RUN pip install --no-cache-dir torch torchvision numpy opencv-python-headless scikit-learn roboflow transformers==4.38.2
RUN pip install --no-cache-dir autodistill-grounding-dino

CMD ["python"]