# Raport analityczny — SCANWAY (SCW.WA) intraday timing

> **Cel projektu:** zbudować model, który w trakcie sesji GPW podpowiada
> "lokalny szczyt" — moment, w którym w najbliższej godzinie (4 świece 15-min)
> cena prawdopodobnie spadnie o ≥0.5%, czyli kiedy *teraz* był dobry moment na
> sprzedaż.
>
> Ten dokument podsumowuje wizualnie: (1) eksplorację danych (EDA),
> (2) inżynierię cech, (3) proces wyboru modelu i (4) porównanie kandydatów.
> Wykresy są generowane skryptem [`tools/make_report_charts.py`](../tools/make_report_charts.py)
> z prawdziwych danych — odtworzysz je komendą `py tools/make_report_charts.py`.

---

## 1. EDA — co siedzi w danych

### 1.1 Cena spółki (kontekst długoterminowy)

![Cena tygodniowa SCW](assets/01_cena.svg)

SCANWAY to **mała spółka o ekstremalnej zmienności**: w oknie danych cena
przeszła z ~40 PLN do >450 PLN i z powrotem do ~300 PLN. To kluczowy kontekst —
silne trendy i duże wahania oznaczają, że "lokalny szczyt" jest pojęciem
bardzo zależnym od reżimu rynkowego (inaczej wygląda w trendzie bocznym
2024, inaczej w euforii 2026).

### 1.2 Dwa horyzonty danych

| Zbiór | Zakres | Liczność | Rola |
|---|---|---|---|
| Dzienny (`data/scw_d.csv`) | 2023-10-11 → 2026-06-19 | **669 sesji** | kontekst (trend, zmienność, dzień tygodnia) |
| Intraday (yfinance, live) | ~ostatnie 60 dni | **1847 świec / 57 sesji** | lokalna dynamika sesji (target + cechy) |

> ⚠️ **Najważniejsze ograniczenie całego projektu:** model uczy się "szczytu"
> tylko na **57 sesjach** intraday. Co więcej, świece z jednego dnia są ze sobą
> silnie skorelowane — efektywna liczba *niezależnych* obserwacji jest bliższa
> ~57 niż 1847. Dlatego walidacja dzieli dane **po dniach**, nie po wierszach
> (patrz §3), i dlatego żadnego wyniku nie wolno traktować jako gwarancji.

### 1.3 Sezonowość dzienna (dzień tygodnia)

![Sezonowość dzienna](assets/02_sezonowosc_dzienna.svg)

Piątek historycznie ma najwyższy średni zwrot (+0.54%) i najwyższy wskaźnik
domknięcia luki, wtorek jest najsłabszy (+0.04%). To realna obserwacja na
669 sesjach — ale uwaga: cecha `dow_num` w modelu koduje dzień liniowo
(Pon=0 … Pt=4), co dla modeli liniowych jest słabym kodowaniem (poniedziałek
i piątek są "daleko", choć w cyklu tygodnia sąsiadują).

### 1.4 Sezonowość śróddzienna — **najsilniejszy sygnał w całych danych**

![Sezonowość śróddzienna](assets/03_sezonowosc_sroddzienna.svg)

**Otwarcie 09:00 jest dziennym maksimum w 41% sesji.** Żadna inna godzina nie
zbliża się nawet do 10%. To jest najmocniejszy, najbardziej intuicyjny sygnał
w całym zbiorze: dla tej spółki bardzo często "szczyt jest na otwarciu".
Cecha `minute_of_day` powinna więc być jedną z najważniejszych w modelu.

> 🔧 **Pułapka inżynieryjna powiązana z tym sygnałem** — patrz §2.3.

---

## 2. Inżynieria cech

### 2.1 Cechy śróddzienne (dynamika bieżącej sesji)

| Cecha | Co mierzy |
|---|---|
| `dist_from_vwap_pct` | odległość ceny od VWAP liczonego od otwarcia |
| `ret_5`, `ret_1` | momentum z 5 / 1 ostatnich świec |
| `vol_zscore` | nietypowość wolumenu (z-score, okno 20 świec) |
| `minute_of_day` | pora dnia (godz×60 + min) — koduje sygnał z §1.4 |
| `rsi_14` | RSI(14) — wyprzedanie / wykupienie |

### 2.2 Cechy dzienne (kontekst, z `.shift(1)` — bez przecieku)

`prior_day_ret_pct`, `realized_vol_10d_pct`, `dist_from_52w_high_pct`,
`up_streak`, `gap_pct`, `dow_num`. Wszystkie przesunięte o 1 dzień, więc na
danej sesji model używa wyłącznie informacji znanej **przed** jej otwarciem.
To poprawnie zabezpiecza przed *look-ahead bias*.

### 2.3 Ryzyko: `vol_zscore` wycina początek sesji

`vol_zscore` używa `rolling(20)` — wymaga 20 wcześniejszych świec **tego samego
dnia**. Świece bez kompletu okna dostają NaN i są usuwane przez `dropna`.
Sesja GPW 9:00–17:00 to ~33 świece 15-min, więc tracimy z treningu **pierwsze
~20 świec każdego dnia — w tym świecę 09:00**, czyli dokładnie ten moment,
który w 41% przypadków jest szczytem dnia (§1.4).

> **Rekomendacja:** policzyć `vol_zscore` per dzień z mniejszym `min_periods`
> (np. 5) albo zastąpić oknem kroczącym międzysesyjnym, żeby nie kasować
> najbardziej informacyjnego fragmentu sesji.

---

## 3. Proces wyboru modelu

### 3.1 Walidacja chronologiczna (walk-forward)

Dane dzielone są na kolejne okna **tylko do przodu w czasie** (train zawsze
starszy niż test), z podziałem po dacie sesji (`n_splits=6`). To jedyny
uczciwy sposób oceny na danych czasowych — losowy `train_test_split` dałby
zawyżone wyniki, bo model "widziałby" sąsiednie fragmenty przyszłości.

### 3.2 Bilans klas — źródło pułapki metrycznej

![Bilans klas](assets/04_bilans_klas.svg)

Target jest niezbalansowany ~**77% / 23%**. Przy takiej proporcji metryka
**F1 jest zwodnicza**: model, który zawsze typuje klasę większościową,
dostaje wysoki recall i wysokie F1 — mimo że nie ma żadnej zdolności
predykcyjnej.

### 3.3 Decyzja: wybór po ROC AUC, baseline wykluczony

To jest sedno poprawki w [`src/backtest.py`](../src/backtest.py)
(funkcja `pick_best_model`):

- **kryterium = `avg_roc_auc`** (zdolność rozróżniania klas niezależnie od
  progu) zamiast `avg_f1`,
- **`baseline_most_frequent` wykluczony** z automatycznego wyboru (zostaje
  tylko jako punkt odniesienia).

---

## 4. Porównanie modeli (rzeczywiste wyniki backtestu)

### 4.1 F1 vs ROC AUC — dlaczego F1 mylił

![Porównanie modeli](assets/05_porownanie_modeli.svg)

Czytanie wykresu:

- **`baseline_most_frequent`** ma **najwyższe F1 (0.871)**, ale
  **ROC AUC = 0.500** — to dosłownie rzut monetą. Gdyby wybierać po F1
  (stara logika), program zapisałby właśnie ten bezużyteczny model. Tak też
  się stało w pierwszym uruchomieniu.
- **`gradient_boosting`** ma drugie F1 (0.818), ale jego AUC (0.605) jest
  niższe niż u zwycięzcy — jego F1 też jest podbite wysokim recall.
- **`logistic_regression`** ma **najsłabsze F1 (0.688)**, ale
  **najwyższe ROC AUC (0.642)** — czyli faktycznie najlepiej rozróżnia
  "szczyt" od "nie-szczytu".

### 4.2 Ranking właściwy (po ROC AUC, bez baseline)

![Ranking ROC AUC](assets/06_ranking_auc.svg)

| Pozycja | Model | ROC AUC | Komentarz |
|---|---|---|---|
| 🥇 | **logistic_regression** | **0.642** | wybór produkcyjny — najlepsza rozdzielczość, prosty i interpretowalny |
| 🥈 | random_forest | 0.630 | blisko, ale trudniejszy w interpretacji |
| 🥉 | extra_trees | 0.611 | — |
| 4 | gradient_boosting | 0.605 | mylił przez wysokie F1 |
| 5 | svm_rbf | 0.599 | brak prostej interpretacji per-cecha |
| 6 | knn | 0.580 | najsłabszy z realnych |

### 4.3 Werdykt i uczciwa ocena siły sygnału

**Wybrany model: `logistic_regression`.** Jest najlepszy *względem
pozostałych kandydatów* i jako liniowy daje czytelną interpretację wpływu
cech (backtest wypisuje teraz współczynniki — funkcja `explain_model`).

Ale trzeba to powiedzieć wprost: **AUC ≈ 0.64 to sygnał słaby.** 0.5 to
przypadek, 1.0 to ideał — 0.64 znaczy "lekko lepiej niż rzut monetą". Dla
małej, mocno zmiennej spółki to nie jest niespodzianka. Praktyczny wniosek:
- traktować alert jako **jedną z przesłanek**, nie automat do sprzedaży,
- realnie najwięcej wnosi prosta reguła z EDA: **09:00 bardzo często jest
  szczytem dnia** (§1.4) — to warto wykorzystać niezależnie od modelu ML.

---

## 5. Następne kroki (rekomendacje)

1. **Naprawić `vol_zscore`** (§2.3), żeby nie kasować świecy 09:00 z treningu.
2. **Lepsze kodowanie dnia tygodnia** — zamiast liniowego `dow_num` użyć
   sin/cos albo one-hot.
3. **Więcej danych intraday** — 57 sesji to za mało; im dłuższa historia
   intraday, tym stabilniejszy backtest (patrz uwaga niżej o danych).
4. **Kalibracja progu** `alert_probability_threshold` na podstawie krzywej
   precision-recall, a nie domyślnego 0.6.

---

## Uwaga o danych o mniejszym opóźnieniu

Pytałeś o dokładniejsze dane historyczne SCANWAY z opóźnieniem mniejszym niż
15 min. **Nie mogłem ich pobrać** — to środowisko nie ma dostępu do internetu
(błąd certyfikatu SSL przy każdej próbie sieciowej). Dlatego nie zweryfikuję
tego za Ciebie; poniżej rzetelna ocena opcji, którą możesz sprawdzić u siebie:

- **yfinance/Yahoo (obecne źródło):** interwał min. ~1 min dla bieżących
  ~7 dni i 15 min do ~60 dni wstecz — i tak dane są **opóźnione ~15 min** dla
  GPW (Yahoo nie ma realtime dla warszawskiej giełdy). To realne ograniczenie
  Twojego obecnego pipeline'u.
- **Stooq** (już w `config.yaml` jako fallback dla danych dziennych): ma dane
  intraday 5-min, ale również opóźnione i niepełne dla małych spółek GPW.
- **Realtime / niskie opóźnienie wymaga płatnego źródła:** dane GPW w czasie
  rzeczywistym są licencjonowane przez GPW. Brokerzy z API (np. **XTB xStation
  API**, **Bossa/DM BOŚ**) dają notowania ~realtime dla posiadaczy rachunku —
  to najrealniejsza droga do opóźnienia <15 min.
- **Dla mniejszego opóźnienia: 1-min zamiast 15-min.** Yahoo daje 1-min świece
  dla ostatnich ~7 dni. Można by zbierać je codziennie (cron/scheduler) i
  budować własny, gęstszy zbiór historyczny — to *nie* zmniejsza opóźnienia
  live (wciąż ~15 min), ale daje **dokładniejszy target** (precyzyjniejszy
  moment szczytu) i więcej próbek do treningu.

**Szczera konkluzja:** dokładniejszy *moment* sprzedaży da realnie tylko
źródło realtime od brokera GPW (płatne/rachunek). Darmowe źródła (Yahoo,
Stooq) nie zejdą poniżej ~15 min opóźnienia dla SCW.WA — można za to poprawić
*granularność historyczną* (1-min z Yahoo, zbierane na bieżąco), co pomoże
modelowi, ale nie samemu opóźnieniu alertu.
