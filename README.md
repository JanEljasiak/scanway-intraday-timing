# SCW (Scanway SA) - intraday timing toolkit

Narzedzie, ktore wysyla **jeden sygnal dziennie w momencie DZIENNEGO MAKSIMUM**
ceny akcji Scanway SA (GPW: SCW) - zeby pomoc trafic z momentem sprzedazy.
Model docelowy to tryb `peak` (`py main.py peak`). Wczesniejsze ujecie
"lokalnych szczytow" (wiele sygnalow dziennie) jest zachowane jako analiza
pomocnicza (`backtest`/`replay`/`evaluate`).

> 📊 **[Raport analityczny (EDA + model dziennego maksimum, z wykresami)](docs/RAPORT_ANALIZA.md)**
> - wizualne podsumowanie danych, cech, funkcji straty i wynikow.

## ⚠️ Disclaimer

To narzedzie analityczne, **nie system gwarantujacy zysk i nie porada
inwestycyjna**:

- SCW to mala, mocno zmienna spolka (sektor New Space) notowana realnie
  od pazdziernika 2023 - krotka historia oznacza wiekszy ryzyko, ze
  "wzorce" wykryte przez model to przeuczenie/szum, a nie realna
  prawidlowosc.
- Dane sroddzienne (15-min) z yfinance maja typowo ~15 min opoznienia i
  siegaja max ~60 dni wstecz - to twardy limit darmowego zrodla.
- Backtest "na papierze" nigdy nie gwarantuje powtarzalnosci na zywo.
- Autor (i Claude, ktory to napisal) nie sa doradcami inwestycyjnymi.

## Co jest w srodku

```
scanway-intraday-timing/
├── main.py                  # CLI: update-data / backtest / replay / evaluate / peak / live
├── config.yaml               # wszystkie parametry (ticker, progi, kara, itp.)
├── requirements.txt
├── data/
│   ├── scw_d.csv               # historia dzienna od debiutu (2023-10-11)
│   └── intraday_snapshot.csv   # realny snapshot intraday (~57 sesji)
├── models/                      # tu trafia wytrenowany model (po backtest)
├── src/
│   ├── config.py                 # wczytywanie config.yaml
│   ├── data_sources.py           # Stooq (dzienne) + yfinance (intraday live)
│   ├── features.py               # inzynieria cech + sezonowosc + target
│   ├── models.py                 # kandydaci ML do porownania
│   ├── backtest.py               # walk-forward walidacja + zapis najlepszego modelu
│   ├── replay.py                 # odtworzenie jednej sesji swieca-po-swiecy
│   ├── evaluate.py               # ocena dzien-po-dniu + test 'lepiej niz losowo'
│   ├── peak.py                   # tryb 'jeden szczyt dziennie' + asymetryczna kara
│   └── alert.py                  # alert do konsoli + petla live
├── tools/
│   └── make_report_charts.py    # generator wykresow SVG do raportu
├── docs/
│   └── RAPORT_ANALIZA.md         # raport analityczny z wykresami
└── tests/                        # testy offline (dzialaja bez internetu)
```

## Instalacja na nowym komputerze

```bash
git clone <adres-twojego-repo>
cd scanway-intraday-timing

py -m venv venv
source venv/Scripts/activate    # Windows (Git Bash); cmd: venv\Scripts\activate

pip install -r requirements.txt
```

Sprawdz, czy wszystko dziala (bez internetu, testuje tylko dane lokalne):

```bash
pytest tests/
```

## Uzycie

### 1. Odswiez dane i zobacz sezonowosc

```bash
py main.py update-data
```

Dociaga najnowsze dane dzienne ze Stooq (jesli jest internet - inaczej
korzysta z lokalnego `data/scw_d.csv`), pobiera dane intraday live i
pokazuje, ktore godziny/dni tygodnia historycznie wypadaly najlepiej.

### 2. Przetestuj i porownaj modele ML

```bash
py main.py backtest
```

Buduje zbior cech (dlugi kontekst dzienny + krotkie dane intraday),
testuje kilka modeli (regresja logistyczna, KNN, SVM, Random Forest,
Extra Trees, Gradient Boosting, opcjonalnie XGBoost) na walidacji
chronologicznej (walk-forward - bez przeciekow z przyszlosci), wypisuje
tabele porownawcza (sortowana wg ROC AUC) i zapisuje najlepszy model do
`models/best_model.joblib`. Wybor "najlepszego" ignoruje
`baseline_most_frequent` i opiera sie na ROC AUC, nie na F1 - przy
niezbalansowanych klasach (np. 77%/23%) F1 faworyzuje modele, ktore po
prostu zawsze przewiduja klase wiekszosciowa (wysoki recall, ale ROC
AUC=0.5, czyli zero realnej sily predykcyjnej).

Przyklad outputu:
```
              model  avg_precision  avg_recall  avg_f1  avg_roc_auc  n_folds
  gradient_boosting           0.58        0.51    0.54         0.61        6
       random_forest           0.55        0.49    0.52         0.58        6
 logistic_regression           0.52        0.55    0.53         0.57        6
baseline_most_frequent          0.00        0.00    0.00          NaN       6
                 ...
```
Jesli "prawdziwe" modele ledwo biją `baseline_most_frequent`, to sygnal,
ze rynek dla SCW w tym okresie jest bliski losowemu (co dla malej spolki
nie jest niespodzianka).

### 3. Odtworz jedna sesje swieca-po-swiecy (replay)

```bash
py main.py replay              # ostatnia historyczna sesja (z internetu)
py main.py replay --synthetic  # sesja ilustracyjna, dziala offline (bez internetu)
```

Odtwarza sesje OUT-OF-SAMPLE: model jest trenowany na wszystkich sesjach
OPROCZ tej odtwarzanej, wiec widzisz, jak zadzialalby na dniu, ktorego nie
widzial na treningu. Dla kazdej 15-min swiecy pokazuje prawdopodobienstwo
"lokalnego szczytu", porownuje alerty z faktycznymi momentami sprzedazy
(ground truth) i szczytem dnia, liczy ROC AUC sesji oraz wplyw progu alertu na
precyzje/pokrycie. To najlepszy sposob, zeby ZOBACZYC jak narzedzie dziala w
praktyce. Pelne omowienie z wykresami (w tym co dokladnie mierzy ROC AUC):
[docs/RAPORT_ANALIZA.md](docs/RAPORT_ANALIZA.md) (sekcje 4.4 i 5).

### 4. Ocena dzien-po-dniu + dowod, ze model bije losowy (evaluate)

```bash
py main.py evaluate                  # split chronologiczny 80/20 na realnych danych
py main.py evaluate --synthetic      # offline (dane ilustracyjne)
py main.py evaluate --train-frac 0.8 # zmien proporcje treningu
```

Trenuje model na 80% najwczesniejszych sesji i ocenia KAZDY dzien testowy
osobno (ROC AUC, precyzja, pokrycie). Dodatkowo robi **test permutacyjny**
(1000 modeli z przetasowanymi etykietami) i podaje p-value - czyli formalny
dowod, czy wynik jest lepszy niz losowy. Wypisuje tez jawna **formule**
regresji logistycznej (wagi cech). Omowienie: sekcja 6 w
[docs/RAPORT_ANALIZA.md](docs/RAPORT_ANALIZA.md).

### 5. Tryb peak - jeden sygnal dziennie = dzienne maksimum

```bash
py main.py peak                 # realne dane, jeden sygnal dziennie
py main.py peak --synthetic     # offline
```

W przeciwienstwie do `backtest`/`evaluate` (wiele lokalnych sygnalow), tryb
`peak` celuje w DZIENNE MAKSIMUM: target to 1 swieca/sesje (najwyzszy Close),
a model ma ASYMETRYCZNA kare. Decyzja: max 1 alert dziennie. Pokazuje, jak
blisko dziennego szczytu sprzedajesz (regret %) vs proste strategie, z testem
permutacyjnym. Omowienie: sekcja 7 w
[docs/RAPORT_ANALIZA.md](docs/RAPORT_ANALIZA.md).

**Jak liczona jest funkcja straty (szukanie dziennego maksimum).** Model
minimalizuje wazona entropie krzyzowa (weighted log-loss):

```
L = -(1/N) * SUMA_i  w[y_i] * [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
    w[1] = daily_high_fn_penalty   (klasa "szczyt")
    w[0] = 1                       (klasa "nie-szczyt")
```

Przegapienie prawdziwego szczytu (y=1, male p) jest mnozone przez
`daily_high_fn_penalty`, wiec kosztuje tyle razy wiecej niz falszywy alarm
(kod: `class_weight={0:1, 1:penalty}` w `src/peak.py`). Weryfikujemy to
trzema niezaleznymi testami out-of-sample: (1) **regret** = sredni % ponizej
dziennego szczytu, (2) **test permutacyjny** (p-value vs losowy wybor swiecy),
(3) **sweep kary** (wieksza kara => mniej przegapien). Wszystkie na realnych
danych w sekcji 7 raportu.

### 6. Uruchom alert live

```bash
py main.py live
```

W godzinach sesji GPW (domyslnie 9:00-17:00) odpytuje rynek co 15 minut,
liczy prawdopodobienstwo "lokalnego szczytu" wytrenowanym modelem i
wypisuje alert w konsoli, gdy przekroczy prog z `config.yaml`
(`alert_probability_threshold`).

## Konfiguracja (`config.yaml`)

Najwazniejsze parametry:

| Klucz | Znaczenie |
|---|---|
| `sell_horizon_bars` | ile 15-min swiec do przodu definiuje "niedaleka przyszlosc" (tryb local_top) |
| `sell_target_drop_pct` | jaki spadek % w tym oknie liczymy jako "tu warto bylo sprzedac" |
| `daily_high_fn_penalty` | tryb `peak`: ile razy mocniej karzemy przegapienie dziennego szczytu |
| `peak_tolerance_bars` | tryb `peak`: tolerancja (w swiecach) uznania alertu za trafiony |
| `alert_probability_threshold` | od jakiego prawdopodobienstwa wysylamy alert |
| `walk_forward_splits` | liczba okien w walidacji chronologicznej |

## Aktualizacja danych historycznych

`data/scw_d.csv` to seed danych od debiutu spolki. Komenda
`py main.py update-data` (i `backtest`/`live`) probuje go automatycznie
dociagac swiezszymi sesjami ze Stooq przy kazdym uruchomieniu - jesli nie
ma internetu, po prostu korzysta z tego co jest na dysku.
