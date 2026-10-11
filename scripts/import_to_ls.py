import shutil
import requests

def main():
    labeled_dir = "/opt/airflow/data/labeled"
    zip_path = "/opt/airflow/data/temp_import"
    
    # 1. Pakujemy zdjęcia i pliki .txt do paczki ZIP w tle
    shutil.make_archive(zip_path, "zip", labeled_dir)
    
    # 2. Konfiguracja API Label Studio (zwróć uwagę na port 8080 - tak widzi go Docker)
    PROJECT_ID = "1" # Upewnij się, że to ID Twojego projektu
    API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA5ODk0NTA1MiwiaWF0IjoxNzkxNzQ1MDUyLCJqdGkiOiI0Mjk0NTdhY2JlMWY0MjU1YTY2ZDlkOGI3NmE5MjFiNCIsInVzZXJfaWQiOiIxIn0.aOyYsnt-zKl9T-clE5eG5iGp0KTURuVNpP07vLyaQpE" 
    
    url = f"http://label-studio:8080/api/projects/{PROJECT_ID}/import"
    headers = {"Authorization": f"Token {API_TOKEN}"}
    
    print("Wysyłanie wygenerowanych etykiet do Label Studio...")
    with open(zip_path + ".zip", "rb") as f:
        response = requests.post(url, headers=headers, files={"file": f})
        
    if response.status_code == 201:
        print("Sukces! Dane zaimportowane. Możesz rozpocząć ręczną weryfikację.")
    else:
        print(f"Błąd API: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()