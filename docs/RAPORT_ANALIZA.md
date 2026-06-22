# Raport analityczny — SCANWAY (SCW.WA) intraday timing

> **Cel projektu (model docelowy):** wysłać **jeden sygnał dziennie** w momencie
> **dziennego maksimum** ceny — żebyś dostał alert „TERAZ jest górka” i mógł
> sprzedać. To jest tryb `peak` (komenda `py main.py peak`) i to on jest
> opisany jako model produkcyjny w tym raporcie.
>
> Plan dokumentu: (1) EDA — co siedzi w danych, (2) inżynieria cech,
> (3) **model dziennego maksimum** (target + funkcja straty + decyzja),
> (4) **wyniki modelu** (jak blisko górki sprzedajesz + dowody),
> (5) dlaczego regresja logistyczna i czym jest ROC AUC,
> (6) ograniczenia i następne kroki.
>
> Wszystkie wykresy generuje [`tools/make_report_charts.py`](../tools/make_report_charts.py)
> z prawdziwych danych (snapshot `data/intraday_snapshot.csv`, 57 sesji);
> odtworzysz je komendą `py tools/make_report_charts.py`.

> ℹ️ **Uwaga o historii projektu.** Pierwotnie model szukał *każdego* lokalnego
> momentu sprzedaży (wiele sygnałów dziennie, `target_local_top`). Zostało to
> **zmienione**: teraz celem jest **dzienne maksimum** (jeden sygnał). Dawne
> ujęcie żyje jeszcze w komendach pomocniczych `backtest`/`replay`/`evaluate`,
> które uzasadniły wybór cech i rodziny modelu — ale nie jest celem projektu.

---

## 1. EDA — co siedzi w danych

### 1.1 Cena spółki (kontekst długoterminowy)

![Cena tygodniowa SCW](assets/01_cena.svg)

SCANWAY to **mała spółka o ekstremalnej zmienności**: w oknie danych cena
przeszła z ~40 PLN do >450 PLN i z powrotem do ~300 PLN. To kluczowy kontekst —
silne trendy i duże wahania oznaczają, że „gdzie wypada dzienne maksimum” mocno
zależy od reżimu rynkowego (inaczej w trendzie bocznym 2024, inaczej w euforii
2026).

### 1.2 Dwa horyzonty danych

| Zbiór | Zakres | Liczność | Rola |
|---|---|---|---|
| Dzienny (`data/scw_d.csv`) | 2023-10-11 → 2026-06-19 | **669 sesji** | kontekst (trend, zmienność, dzień tygodnia) |
| Intraday (snapshot z yfinance) | ostatnie ~60 dni | **1847 świec / 57 sesji** | dynamika sesji (cechy + target) |

> ⚠️ **Najważniejsze ograniczenie całego projektu:** model uczy się dziennego
> maksimum tylko na **57 sesjach**. Każda sesja daje dokładnie jeden przykład
> „szczytu”, więc realnie mamy **57 przykładów pozytywnych** — bardzo mało.
> Dlatego walidacja dzieli dane **po dniach**, a żadnego wyniku nie wolno
> traktować jako gwarancji.

### 1.3 Sezonowość dzienna (dzień tygodnia)

![Sezonowość dzienna](assets/02_sezonowosc_dzienna.svg)

Piątek historycznie ma najwyższy średni zwrot (+0.54%), wtorek najsłabszy
(+0.04%). Realna obserwacja na 669 sesjach — ale uwaga: cecha `dow_num` koduje
dzień liniowo (Pon=0 … Pt=4), co dla modelu liniowego jest słabym kodowaniem
(poniedziałek i piątek są „daleko”, choć w cyklu tygodnia sąsiadują).

### 1.4 Sezonowość śróddzienna — **najważniejszy fakt dla naszego celu**

![Sezonowość śróddzienna](assets/03_sezonowosc_sroddzienna.svg)

**Otwarcie 09:00 jest dziennym maksimum w 41% sesji.** Żadna inna godzina nie
zbliża się do 10%. Skoro szukamy właśnie dziennego maksimum, to jest sygnał
numer jeden: dla tej spółki górka bardzo często wypada na otwarciu. Dlatego
cecha `minute_of_day` jest jedną z najważniejszych w modelu (i faktycznie ma
silną ujemną wagę — patrz §3).

### 1.5 Interwał 15 min — wybór, nie ograniczenie sprzętowe

![Interwały yfinance](assets/07_interwaly_yfinance.svg)

yfinance (darmowe źródło) narzuca twardy kompromis: im drobniejszy interwał,
tym krótsza historia. `1m` sięga tylko ~7 dni, `15m` daje ~60 dni (=~57 sesji)
przy 33 świecach/sesję — świadomy punkt równowagi (`config.yaml`:
`intraday_interval: "15m"`, `intraday_period: "60d"`).

| Interwał | Max historia | Świec/sesja | Werdykt |
|---|---|---|---|
| 1m | ~7 dni | ~480 | za krótka historia do treningu |
| 5m | ~60 dni | ~96 | możliwe, więcej szumu |
| **15m** | **~60 dni** | **~33** | **wybór: balans danych i szumu** |
| 30m | ~60 dni | ~16 | za grubo na timing śróddzienny |
| 1h | ~730 dni | ~8 | długa historia, za mało punktów w dniu |

> 15 min nie jest „maksymalną dokładnością” — to najlepszy kompromis dla
> *darmowego* źródła. Drobniej z sensowną historią = źródło płatne (patrz koniec).

---

## 2. Inżynieria cech

Te same cechy zasilają model dziennego maksimum.

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
danej sesji model używa wyłącznie informacji znanej **przed** jej otwarciem —
to poprawnie zabezpiecza przed *look-ahead bias*.

### 2.3 Uwaga o `vol_zscore` (sprawdzone na danych)

`vol_zscore` używa `rolling(20)` liczonego **globalnie** (na całym szeregu, nie
per dzień). W praktyce z 1867 świec do treningu trafia 1847 — tracimy tylko
**~20 świec łącznie** (początek pierwszej sesji), a **55 z 57 sesji zachowuje
swoją świecę 09:00**. Czyli wbrew obawom NIE tracimy poranków każdego dnia —
to ważne, bo 09:00 jest najczęstszym dziennym maksimum (§1.4).

> **Drobna niedoskonałość (nie błąd krytyczny):** ponieważ okno jest globalne,
> `vol_zscore` na otwarciu danego dnia korzysta po części z wolumenu z końcówki
> poprzedniej sesji. Czystsze byłoby okno per-dzień z `min_periods` — to
> kosmetyczna poprawka, target dziennego maksimum jest dostępny dla ~wszystkich
> sesji.

---

## 3. Model docelowy: DZIENNE MAKSIMUM (tryb `peak`)

```bash
py main.py peak                # realne dane, jeden sygnał dziennie
py main.py peak --synthetic    # offline (gdy brak danych intraday)
```

### 3.1 Target — dokładnie jeden szczyt na dzień

`target_daily_high = 1` dla świecy o **najwyższym Close** w danej sesji, `0` dla
reszty. Na naszych danych to **57 jedynek na 57 sesji** — dokładnie jeden
„dobry moment na sprzedaż” dziennie (osiągalny najlepszy kurs na zamknięciu
świecy). To zupełnie inny target niż wcześniejszy „lokalny szczyt” (wiele
jedynek dziennie, ujęcie pomocnicze).

Klasy są **silnie niezbalansowane**: ~**3%** świec to szczyt (1 z ~33), ~97% to
nie-szczyt. To wymusza specjalną funkcję straty (niżej).

### 3.2 Funkcja straty — większa kara za PRZEGAPIENIE szczytu

Model (regresja logistyczna) minimalizuje **ważoną entropię krzyżową**
(weighted log-loss). Dla świecy `i` z etykietą `yᵢ ∈ {0,1}` (`1` = dzienne
maksimum) i przewidywanym prawdopodobieństwem `pᵢ`:

```
L = − (1/N) · Σᵢ  w[yᵢ] · [ yᵢ · log(pᵢ) + (1 − yᵢ) · log(1 − pᵢ) ]

      w[1] = daily_high_fn_penalty   (waga klasy "szczyt")
      w[0] = 1                       (waga klasy "nie-szczyt")
```

Sedno jest w `w[1]`: przegapienie prawdziwego szczytu (`yᵢ=1`, ale `pᵢ` małe →
`log(pᵢ)` mocno ujemny) jest mnożone przez `daily_high_fn_penalty`, więc
kosztuje tyle razy więcej niż fałszywy alarm. To realizuje wymaganie „większa
kara za nieprzewidzenie alertu”. W kodzie: `class_weight = {0: 1, 1: penalty}`
w [`src/peak.py`](../src/peak.py) (`train_peak_model`). Domyślnie
`daily_high_fn_penalty: 12` w `config.yaml`.

### 3.3 Reguła decyzji — maksymalnie jeden alert dziennie

Dla każdej sesji bierzemy świecę o **najwyższym prawdopodobieństwie**. Jeśli
przekracza próg (`alert_probability_threshold`, domyślnie 0.6) → **alert**
(sprzedajesz). Jeśli żadna świeca nie przekroczy progu → **brak alertu** tego
dnia (co jest karane w ocenie jako „trzymasz do zamknięcia”). Tak gwarantujemy
**max 1 sygnał dziennie**, zgodnie z założeniem.

### 3.4 Formuła modelu (cały algorytm, jawnie)

```
z = b₀ + Σⱼ  wⱼ · (xⱼ − średniaⱼ) / odchylenieⱼ      # standaryzacja + ważona suma
p(szczyt) = 1 / (1 + e^(−z))                          # ściśnięcie do 0–1
ALERT (raz dziennie) dla świecy o najwyższym p, jeśli p ≥ 0.60
```

![Formuła modelu PEAK](assets/13_formula.svg)

Realne wagi modelu peak (wyraz wolny `b₀ = −2.91`, niski bo szczyt to tylko ~3%
świec):

- **`dist_from_vwap_pct` (+1.59)** — najsilniejsza cecha: im wyżej cena nad
  VWAP (wybicie ponad średnią sesji), tym większa szansa, że to dzienny szczyt.
- **`minute_of_day` (−1.23)** — im później w sesji, tym mniejsza szansa szczytu
  → model premiuje **wcześniejsze** godziny, zgodnie z §1.4 (górka często rano).
- **`ret_1` (+0.42)**, **`vol_zscore` (+0.35)** — świeży wzrost na podwyższonym
  wolumenie podbija prawdopodobieństwo szczytu.
- **`realized_vol_10d_pct` (−0.30)** — w okresach wysokiej zmienności dziennej
  model jest ostrożniejszy.

---

## 4. Wyniki modelu (out-of-sample, realne dane)

Ocena na chronologicznym splicie 80/20: **46 sesji treningowych, 11 testowych**
(2026-06-05 … 2026-06-19), których model nie widział na treningu.

### 4.1 Najważniejszy wykres: jak blisko górki sprzedajesz

![Regret peak](assets/14_peak_regret.svg)

„Regret” = o ile procent **poniżej** dziennego maksimum sprzedałeś, idąc za
alertem. To liczba, która realnie Cię interesuje:

| Strategia | Średnio poniżej szczytu |
|---|---|
| **MODEL (peak)** | **1.72%** |
| sprzedaż na otwarciu 09:00 | 2.32% |
| sprzedaż w losowym momencie | 2.78% |
| trzymanie do zamknięcia | 3.58% |
| idealnie (sufit) | 0.00% |

Model sprzedaje **bliżej górki niż każda prosta strategia** — w tym niż reguła
„sprzedaj na otwarciu”, która z EDA (§1.4) wydawała się mocna.

### 4.2 Gdzie był szczyt, a gdzie alert (każdy dzień testowy)

![Timeline peak](assets/15_peak_timeline.svg)

Czerwona kropka = faktyczny szczyt, zielony trójkąt = alert modelu. Część dni
trafia idealnie (lag 0), w innych alarmuje za wcześnie. Trafień „co do świecy”
(±1) jest **~36%**, a alert odpalił w **9 z 11 dni**.

> **Uczciwie:** model nie wskazuje górki z chirurgiczną precyzją co do świecy.
> Ale *cenowo* jesteś średnio bardzo blisko szczytu (1.72% poniżej), bo wokół
> maksimum cena jest dość płaska. To realnie użyteczne jako „sprzedaj teraz”.

### 4.3 Jak weryfikujemy, że strata faktycznie znajduje maksimum

**(a) Sweep kary** — zmieniamy `daily_high_fn_penalty` i patrzymy, czy zgodnie
z teorią większa kara → mniej przegapień:

![Wpływ kary](assets/16_peak_kara.svg)

- **kara = 1** (symetryczna): model tchórzliwy — **nie alarmuje wcale**,
  regret = 3.58% (tyle tracisz trzymając do zamknięcia).
- **kara = 12** (domyślna): alarmuje w 9/11 dni, regret **1.72%**.
- **kara = 30**: alarmuje codziennie, trafia 45% dni, regret **1.24%**.

To dokładnie zamierzony efekt: mocniejsza kara za przegapienie → model częściej
i celniej łapie szczyt. `daily_high_fn_penalty` stroisz sam w `config.yaml`.

**(b) Test permutacyjny** — tasujemy 1000× przypisanie prawdopodobieństw do
świec w obrębie dnia (model traci wiedzę, gdzie jest szczyt) i liczymy regret.
Wynik: średni regret modelu **1.72%** vs **3.06%** przy losowym wyborze świecy,
**p-value = 0.003**. Czyli model wybiera moment sprzedaży **istotnie lepiej niż
przypadek** — to nie jest szczęśliwy traf.

---

## 5. Dlaczego regresja logistyczna i czym jest ROC AUC

### 5.1 Wybór rodziny modelu (porównanie na zadaniu pomocniczym)

Rodzinę modelu wybraliśmy, porównując 7 kandydatów walidacją chronologiczną
(walk-forward) na pomocniczym zadaniu klasyfikacji świec. Kryterium = **ROC AUC**
(nie F1, bo przy niezbalansowanych klasach F1 faworyzuje model, który zawsze
typuje klasę większościową):

![Porównanie modeli](assets/05_porownanie_modeli.svg)
![Ranking ROC AUC](assets/06_ranking_auc.svg)

| Pozycja | Model | ROC AUC |
|---|---|---|
| 🥇 | **logistic_regression** | **0.642** |
| 🥈 | random_forest | 0.630 |
| 🥉 | extra_trees | 0.611 |
| 4 | gradient_boosting | 0.605 |
| 5 | svm_rbf | 0.599 |
| 6 | knn | 0.580 |
| – | baseline_most_frequent | 0.500 |

`logistic_regression` wygrała i ma trzy zalety, które przeważyły także dla
trybu peak: najlepsza rozdzielczość, **pełna interpretowalność** (jawna formuła,
§3.4) i mało parametrów → mniejsze ryzyko przeuczenia na 57 sesjach. Dlatego
tryb peak używa tej samej rodziny, dokładając asymetryczną stratę (§3.2).

### 5.2 Co dokładnie mierzy ROC AUC

![Rozkłady wyników ROC](assets/09_roc_rozklady.svg)
![Krzywa ROC](assets/10_roc_krzywa.svg)

ROC AUC odpowiada na pytanie: *jeśli wylosuję jedną świecę „szczyt” i jedną
„nie-szczyt”, jaka jest szansa, że model da tej pierwszej WYŻSZE
prawdopodobieństwo?* 0.5 = rzut monetą, 1.0 = ideał. Zalety: jest **niezależne
od progu** (a próg `alert_probability_threshold` można zmieniać) i **odporne na
niezbalansowanie**. Na wykresach: im mniej nakładają się rozkłady wyników obu
klas, tym wyższy AUC; krzywa ROC im wyżej-lewiej, tym lepiej.

---

## 6. Ograniczenia i następne kroki

**Ograniczenia (wszystkie realne):**
- tylko **57 przykładów szczytu** (1/sesję) — mała próba, duża wariancja;
- **brak kosztów transakcyjnych** i poślizgu w ocenie;
- dane live z yfinance mają **~15 min opóźnienia** dla GPW → alert na żywo bywa
  spóźniony o 15–30 min względem realnego szczytu;
- trafienia „co do świecy” ±1 to ~36% — precyzja czasowa jest umiarkowana
  (choć cenowo regret jest mały).

**Następne kroki:**
1. **Strojenie `daily_high_fn_penalty`** — sweep (§4.3) pokazuje, że wyższa kara
   (np. 30) daje mniejszy regret kosztem częstszych alertów; dobierz pod swój styl.
2. **Lepsze kodowanie `dow_num`** — sin/cos zamiast liniowego 0–4.
3. **Więcej danych intraday** — 57 sesji to mało; zbieranie 1-min co dzień da
   gęstszy zbiór (patrz niżej).
4. **`vol_zscore` per-dzień** z `min_periods` — kosmetyczna poprawka z §2.3.

---

## Uwaga o danych o mniejszym opóźnieniu

Dla *dokładniejszego momentu* dziennego maksimum kluczowe jest opóźnienie danych:

- **yfinance/Yahoo (obecne źródło):** 1 min dla ~7 dni, 15 min do ~60 dni — i
  tak **opóźnione ~15 min** dla GPW (Yahoo nie ma realtime dla Warszawy).
- **Stooq:** intraday 5-min, też opóźnione i niepełne dla małych spółek.
- **Realtime <15 min wymaga źródła płatnego/brokerskiego** — np. **XTB xStation
  API** albo **DM BOŚ/Bossa API** dają notowania ~realtime dla posiadaczy
  rachunku. To jedyna realna droga do wczesnego alertu o szczycie.
- **Granularność 1-min z Yahoo** (ostatnie ~7 dni), zbierana codziennie, nie
  zmniejszy opóźnienia live, ale da **dokładniejszy target** (precyzyjniejszy
  moment maksimum) i więcej próbek do treningu.

**Konkluzja:** wczesny, precyzyjny alert o dziennym szczycie da realnie tylko
źródło realtime od brokera GPW. Darmowe źródła nie zejdą poniżej ~15 min
opóźnienia dla SCW.WA — można za to poprawić granularność historyczną.
