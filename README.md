# ADHD Classifier — EEG-based Binary Classification


## 🚀 Proces Rekrutacji

Rekrutacja w tym semestrze składa się z dwóch kluczowych kroków:

1.  **Etap 1:** Wypełnienie formularza zgłoszeniowego oraz rozwiązanie **zadania rekrutacyjnego** (szczegóły poniżej).
2.  **Etap 2:** Rozmowa rekrutacyjna.

> [!TIP]
> **Masz już doświadczenie?** Jeśli posiadasz w swoim portfolio ciekawe projekty AI, możesz zostać zaproszony bezpośrednio do 2. etapu z pominięciem zadania technicznego. Pamiętaj, aby pochwalić się nimi w formularzu!

---

## 🛠️ Zadanie Rekrutacyjne: Klasyfikacja ADHD na podstawie EEG

Twoim celem jest stworzenie **binarnego klasyfikatora**, który na podstawie sygnałów EEG potrafi zdiagnozować ADHD.

### 📊 Dane
Skorzystaj ze zbioru danych dostępnego tutaj: **[[EEG Dataset for ADHD](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data)]**.  
Dane są w formacie tabelarycznym (`.csv`) i zawierają:
* Kanały sygnału EEG.
* Identyfikator osoby (`ID`).
* Klasę docelową (`Class`).

> [!NOTE]
> **Nie znasz się na EEG? Nie przejmuj się!** Nie oczekujemy, że od razu będziesz wszystko wiedział. Potraktuj to zadanie jako świetną okazję, by dowiedzieć się czegoś nowego, doczytać o podstawach analizy takich sygnałów.

### ✅ Kryteria Oceny (Na co zwracamy uwagę?)

Podczas sprawdzania Twojego rozwiązania, będziemy oceniać następujące aspekty:

* **Exploratory Data Analysis (EDA):** Krótka analiza sygnałów. Podpowiedź: sprawdź, jakie cechy sygnału EEG są istotne w kontekście ADHD.
* **Przygotowanie danych:** Poprawne załadowanie, czyszczenie i podział zbioru.
    * *Ważne:* Zastosuj **Subject Split** – upewnij się, że dane tej samej osoby nie znajdują się jednocześnie w zbiorze treningowym i testowym.
* **Modelowanie:** Implementacja klasyfikatora przy użyciu jednej z bibliotek: `PyTorch`, `TensorFlow/Keras` lub `Scikit-Learn`.
    * *Ważne:* Mile widziane będzie porównanie kilku podejść/architektur/zestawów hiperparametrów.
* **Jakość kodu:** Postaw na kod modularny. Podziel rozwiązanie na czytelne pliki/moduły zamiast jednego, długiego Notebooka.
* **Ewaluacja:** Dobór odpowiednich metryk, analiza `Confusion Matrix` oraz czytelne wizualizacje wyników.

---

## 💡 Ważna informacja

W tym zadaniu **wysoka skuteczność (Accuracy) nie jest priorytetem**. Bardziej niż wynik, interesuje nas Twoje podejście, jakość napisanego kodu oraz świadome korzystanie z narzędzi. 

*Jeśli wspierasz się narzędziami AI (np. Copilot, ChatGPT), rób to z głową – przygotuj się na uzasadnienie swoich decyzji projektowych podczas rozmowy.*

---

## 📬 Jak oddać zadanie?

1. Skonfiguruj repozytorium (publiczne).
2. Umieść w nim kod oraz krótką instrukcję uruchomienia w `README.md`.
3. Prześlij link w formularzu zgłoszeniowym.

**W razie jakichkolwiek pytań śmiało kontaktuj się na discordzie: kamil4343**

**Powodzenia! Czekamy na Twoje zgłoszenie! 🧠🔥**
