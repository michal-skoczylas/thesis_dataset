# 👥 Przewodnik Kooperacji: Etykietowanie we dwoje (Michał & Janek)

Ten dokument opisuje procedurę podziału pracy przy oznaczaniu i weryfikacji zbioru danych na 2 osoby, z wykorzystaniem tunelu **Cloudflare Tunnel** i dedykowanych projektów w Label Studio.

---

## 🎯 Główne Założenia

1. **Brak instalacji u kolegi:** Kolega (Janek) **nie musi** instalować Dockera, Pythona, DVC ani klonować repozytorium. Potrzebuje jedynie przeglądarki internetowej (Chrome / Firefox / Safari).
2. **Niezależne projekty:**
   - Projekt 1: `dane michal` (dla Ciebie)
   - Projekt 2: `dane janek` (dla kolegi)
3. **Sprawiedliwy podział (Interleaved):** Klatki nie są dzielone blokowo (np. Janek nie dostaje wyłącznie odległości 2.0m, a Michał 1.4m). Podział następuje naprzemiennie lub równomiernie według gestów, co uodparnia proces na stronniczość adnotatora (annotator bias).

---

## 🚀 Krok po Kroku: Jak uruchomić sesję oznaczania

### 1. Uruchomienie Label Studio na Twoim komputerze
W głównym katalogu projektu:
```bash
make ls-up
```
Sprawdź, czy serwis działa: otwórz w przeglądarce [http://localhost:8081](http://localhost:8081).

---

### 2. Wygenerowanie bezpiecznego tunelu dla kolegi
W osobnym oknie terminala uruchom:
```bash
make tunnel
```
Po 2–3 sekundach Cloudflare wyświetli adres URL kończący się na `.trycloudflare.com`, np.:
```text
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at:                                          |
|  https://apple-banana-cherry-delta.trycloudflare.com                                       |
+--------------------------------------------------------------------------------------------+
```
> 🔗 **Skopiuj ten link i wyślij go koledze.** Tunel działa w pełni szyfrowanym kanałem HTTPS i automatycznie przekazuje ruch na Twój lokalny port `8081`.

---

### 3. Logowanie i Konto dla Kolegi
- Obaj wchodzicie na adres URL (Ty możesz przez `localhost:8081`, Janek przez link `trycloudflare.com`).
- Logujecie się swoimi kontami.
- Na liście projektów:
  - Michał klika projekt: **`dane michal`**
  - Janek klika projekt: **`dane janek`**

---

### 4. Konfiguracja Strategii Podziału w `params.yaml`
Zachowanie polecenia `make import-ls-coop` jest w pełni konfigurowalne w pliku [params.yaml](file:///Users/michalskoczylas/Documents/studia/thesis/dane/params.yaml):

```yaml
coop:
  names: ["dane michal", "dane janek"]
  strategy: "interleaved"   # Dostępne strategie: interleaved | gesture | distance | ratio
  ratio: "50:50"            # Proporcja zadań np. "50:50", "60:40", "70:30"
```

| Strategia | Opis | Kiedy stosować? |
|---|---|---|
| `interleaved` | **(Rekomendowana)** Klatki są rozdzielane na przemian (klatka 1 dla Michała, klatka 2 dla Janka). Obaj dostają identyczny przekrój przez wszystkie gesty i odległości. | Domyślna dla rzetelnej pracy zespołowej. |
| `gesture` | Zadania dzielone są według typów gestów (np. Michał robi `closehand`, `ok`, `shaka`, a Janek `openhand`, `dislike`, `v`). | Gdy jedna osoba chce wyspecjalizować się w konkretnym geście. |
| `distance` | Zadania dzielone są według dystansu (np. Michał robi 1.4m, Janek 2.0m). | Rzadziej stosowana, może wprowadzić annotator bias. |
| `ratio` | Proporcjonalny podział według zadanego stosunku (np. Michał 70%, Janek 30%). | Gdy jeden ze współpracowników ma mniej wolnego czasu. |

---

## ⌨️ Ściągawka dla Oznaczającego w Label Studio

Wyślij tę sekcję swojemu koledze przed rozpoczęciem pracy:

### Skróty klawiszowe (przyspieszają pracę x5):
- `1` – wybór narzędzia prostokąta (Bounding Box)
- `Ctrl + Enter` (lub `Cmd + Enter`) – **Submit** (zatwierdzenie zadania i przejście do następnego)
- `Backspace` / `Delete` – usunięcie zaznaczonej ramki (np. fałszywy alarm)
- `Spacja + Przeciągnięcie myszą` – przesuwanie widoku zdjęcia (Pan)
- `Kółko myszy` – przybliżanie/oddalanie zdjęcia (Zoom)

### Co zrobić w 4 typowych sytuacjach:
1. **Model wykrył dłoń idealnie:** Po prostu naciśnij `Ctrl + Enter` (`Submit`).
2. **Model wykrył dłoń, ale uciął palce lub zaznaczył za dużo tła:** Przesuń krawędzie prostokąta myszą i naciśnij `Ctrl + Enter`.
3. **Model nie znalazł dłoni w ogóle:** Wybierz narzędzie prostokąta (`1`), narysuj ramkę wokół dłoni i naciśnij `Ctrl + Enter`.
4. **Model zaznaczył coś w tle, co nie jest dłonią (fałszywy alarm):** Kliknij błędną ramkę, naciśnij `Backspace` i naciśnij `Ctrl + Enter`.
