# SCW (Scanway SA) - intraday timing toolkit

Narzedzie do analizy sezonowosci sroddziennej i sygnalizowania
prawdopodobnych "lokalnych szczytow" ceny akcji Scanway SA (GPW: SCW)
w ciagu sesji gieldowej.

> 📊 **[Raport analityczny (EDA + dobor modelu, z wykresami)](docs/RAPORT_ANALIZA.md)**
> - wizualne podsumowanie danych, inzynierii cech i porownania modeli.

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
├── main.py                  # CLI: update-data / backtest / live
├── config.yaml               # wszystkie parametry (ticker, progi, itp.)
├── .env.example               # szablon na dane do alertow (Telegram)
├── requirements.txt
├── data/
│   └── scw_d.csv               # historia dzienna od debiutu (2023-10-11)
├── models/                      # tu trafia wytrenowany model (po backtest)
├── src/
│   ├── config.py                 # wczytywanie config.yaml + .env
│   ├── data_sources.py           # Stooq (dzienne) + yfinance (intraday live)
│   ├── features.py               # inzynieria cech + sezonowosc
│   ├── models.py                 # kandydaci ML do porownania
│   ├── backtest.py               # walk-forward walidacja + zapis najlepszego modelu
│   └── alert.py                  # wysylka alertow + petla live
└── tests/
    └── test_features.py         # testy offline (dzialaja bez internetu)
```

## Instalacja na nowym komputerze

```bash
git clone <adres-twojego-repo>
cd scanway-intraday-timing

py -m venv venv
source venv/Scripts/activate    # Windows (Git Bash); cmd: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # opcjonalnie - jesli chcesz alerty na Telegram
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

### 4. Uruchom alert live

```bash
py main.py live
```

W godzinach sesji GPW (domyslnie 9:00-17:00) odpytuje rynek co 15 minut,
liczy prawdopodobienstwo "lokalnego szczytu" wytrenowanym modelem i
wysyla alert (konsola + Telegram, jesli skonfigurowany w `.env`), gdy
przekroczy prog z `config.yaml` (`alert_probability_threshold`).

## Konfiguracja (`config.yaml`)

Najwazniejsze parametry:

| Klucz | Znaczenie |
|---|---|
| `sell_horizon_bars` | ile 15-min swiec do przodu definiuje "niedaleka przyszlosc" |
| `sell_target_drop_pct` | jaki spadek % w tym oknie liczymy jako "tu warto bylo sprzedac" |
| `alert_probability_threshold` | od jakiego prawdopodobienstwa wysylamy alert |
| `walk_forward_splits` | liczba okien w walidacji chronologicznej |

## Alerty na Telegramie (opcjonalnie)

1. Utworz bota przez [@BotFather](https://t.me/BotFather), skopiuj token.
2. Wyslij dowolna wiadomosc do bota, potem otworz w przegladarce:
   `https://api.telegram.org/bot<TWOJ_TOKEN>/getUpdates` i odczytaj `chat_id`.
3. Wpisz oba do `.env`.

## Aktualizacja danych historycznych

`data/scw_d.csv` to seed danych od debiutu spolki. Komenda
`py main.py update-data` (i `backtest`/`live`) probuje go automatycznie
dociagac swiezszymi sesjami ze Stooq przy kazdym uruchomieniu - jesli nie
ma internetu, po prostu korzysta z tego co jest na dysku.
