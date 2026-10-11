from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner':'dane',
    'depends_on_past': False,
    'retries': 1,
}


with DAG(
    dag_id='drone_gesture_pipeline',
    default_args=default_args,
    description='Pipeline ML do detekcji gestów dłoni z drona',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None, # Na razie uruchamiamy ręcznie
    catchup=False,
) as dag:
    # Krok 1: Pobranie najnowszych zdjęć z Google Drive przez DVC
    # Komenda wykonuje się wewnątrz kontenera Airflow, który ma zamontowany Twój folder projektu
    pull_data = BashOperator(
        task_id='dvc_pull_raw_data',
        bash_command='git config --global --add safe.directory /opt/airflow && cd /opt/airflow && dvc pull'
    )

    # Krok 2: Uruchomienie detekcji ML
    # Na ten moment to placeholder. Docelowo podepniemy tu DockerOperator, 
    # który podniesie Twój obraz `Dockerfile.ml` z GPU/PyTorch.
    process_images = BashOperator(
        task_id='run_grounding_dino_detection',
        bash_command='docker run --rm --volumes-from dane-airflow-scheduler-1 model-ml python /opt/airflow/scripts/detect_hands.py'
    )
    import_to_label_studio = BashOperator(
        task_id = 'import_to_label_studio',
        bash_command='docker run --rm --network dane_default --volumes-from dane-airflow-scheduler-1 model-ml python /opt/airflow/scripts/import_to_ls.py',    
    )

    # Zdefiniowanie kolejności wykonywania zadań
    pull_data >> process_images>>import_to_label_studio