# SCW (Scanway SA) — sygnał dziennego maksimum

Narzędzie ma **jeden cel**: raz w ciągu sesji GPW wysłać sygnał „**TERAZ jest
szczyt dnia — sprzedaj**” dla akcji Scanway SA (SCW.WA). Dostajesz jeden alert
→ sprzedajesz. To jest tryb **`peak`**.

> 📊 **[Jak to działa — back-end krok po kroku (z wizualizacjami)](docs/JAK_DZIALA.md)**
> · 📈 **[Pełny raport analityczny](docs/RAPORT_ANALIZA.md)**

> ⚠️ **Disclaimer:** narzędzie analityczne, **nie porada inwestycyjna i nie
> gwarancja zysku**. SCW to mała, bardzo zmienna spółka; dane intraday z
> yfinance mają ~15 min opóźnienia. Traktuj alert jako wsparcie decyzji.

---

## Szybki start

```bash
git clone <adres-repo>
cd scanway-intraday-timing

py -m venv venv
source venv/Scripts/activate    # Windows (Git Bash); cmd: venv\Scripts\activate
pip install -r requirements.txt

py main.py peak                 # <- GŁÓWNA komenda: sygnał dziennego maksimum
```

Test offline (bez internetu): `pytest tests/`

---

## Tryb `peak` — jeden sygnał dziennie

```bash
py main.py peak                 # realne dane
py main.py peak --synthetic     # offline (gdy brak danych intraday)
```

Co robi, w skrócie (pełne wyjaśnienie: [docs/JAK_DZIALA.md](docs/JAK_DZIALA.md)):

1. **Cel/etykieta:** dla każdej sesji jedna świeca = dzienne maksimum (najwyższy
   Close). Model uczy się ją rozpoznawać.
2. **Funkcja straty (asymetryczna):** przegapienie szczytu karane
   `daily_high_fn_penalty`× mocniej niż fałszywy alarm — bo przegapienie boli
   bardziej. Wzór (ważona entropia krzyżowa):

   ```
   L = -(1/N) * Σ  w[y] * [ y*log(p) + (1-y)*log(1-p) ]
       w[1] = daily_high_fn_penalty   (świeca-szczyt)
       w[0] = 1
   ```
3. **Decyzja:** model liczy `p(szczyt)` dla wszystkich ~33 świec dnia, bierze
   **jedną najwyższą**; jeśli `p ≥ alert_probability_threshold` → **alert**.
   Maksymalnie jeden sygnał dziennie.
4. **Ocena (na danych, których model nie widział):** „regret” = średni % poniżej
   dziennego maksimum, po jakim sprzedajesz, plus test permutacyjny (czy lepszy
   niż losowo) i sweep kary.

Wynik na realnych danych (out-of-sample): sprzedaż średnio **1.72% poniżej**
dziennego szczytu vs 2.78% przy losowym wyborze, **p-value 0.003**.

---

## Konfiguracja (`config.yaml`)

| Klucz | Znaczenie |
|---|---|
| `daily_high_fn_penalty` | jak mocno karać przegapienie szczytu (wyżej = częstsze alerty, mniejszy regret) |
| `alert_probability_threshold` | od jakiego `p` wysyłamy alert |
| `peak_tolerance_bars` | tolerancja (w świecach) uznania alertu za trafiony |
| `intraday_interval` / `intraday_period` | rozdzielczość i okno danych (domyślnie 15m / 60d) |

---

## Narzędzia pomocnicze (analiza, nie cel projektu)

Te komendy badały dane i uzasadniły wybór cech/modelu. Operują na pomocniczym
ujęciu „lokalnych szczytów” (wiele sygnałów dziennie) — nie na docelowym
dziennym maksimum.

| Komenda | Do czego |
|---|---|
| `py main.py update-data` | odświeża dane i pokazuje sezonowość (dzienną i śróddzienną) |
| `py main.py backtest` | porównuje rodziny modeli (walk-forward, ranking po ROC AUC) |
| `py main.py replay` | odtwarza jedną sesję świeca-po-świecy (out-of-sample) |
| `py main.py evaluate` | ocena dzień-po-dniu + test permutacyjny „lepiej niż losowo” |
| `py main.py live` | pętla alertowa w godzinach GPW (alert w konsoli) |

Szczegóły i wykresy: [docs/RAPORT_ANALIZA.md](docs/RAPORT_ANALIZA.md).

---

## Struktura projektu

```
scanway-intraday-timing/
├── main.py                  # CLI (peak + narzędzia pomocnicze)
├── config.yaml              # parametry (kara, próg, interwał, ...)
├── data/
│   ├── scw_d.csv            # historia dzienna od debiutu (2023-10-11)
│   └── intraday_snapshot.csv# realny snapshot intraday (~57 sesji)
├── src/
│   ├── peak.py              # ⭐ model dziennego maksimum (target, strata, decyzja)
│   ├── features.py          # inżynieria cech
│   ├── data_sources.py      # dane dzienne (Stooq) + intraday (yfinance)
│   ├── evaluate.py          # ocena dzień-po-dniu + testy
│   ├── replay.py            # odtworzenie sesji
│   ├── backtest.py          # porównanie rodzin modeli
│   ├── models.py · config.py · alert.py
├── tools/make_report_charts.py  # generator wykresów SVG
├── docs/JAK_DZIALA.md · RAPORT_ANALIZA.md
└── tests/                   # testy offline
```
