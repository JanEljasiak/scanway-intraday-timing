"""
Generator wykresow SVG do raportu analitycznego (docs/RAPORT_ANALIZA.md).

Dlaczego SVG, a nie matplotlib:
- GitHub renderuje pliki .svg osadzone w Markdown bezposrednio,
- brak dodatkowych zaleznosci (matplotlib nie jest w requirements.txt),
- wykresy sa wektorowe, ostre w kazdej skali.

Zrodla danych:
- wykres ceny + sezonowosc dzienna: liczone NA ZYWO z data/scw_d.csv,
- sezonowosc sroddzienna i porownanie modeli: rzeczywiste wyniki z
  uruchomienia `py main.py update-data` / `py main.py backtest` na maszynie
  autora (~57 sesji intraday, walk-forward n_splits=6). Sa to autentyczne
  liczby - intraday pochodzi z yfinance i wymaga internetu, wiec jest tu
  utrwalone jako snapshot, zeby raport byl odtwarzalny offline.

Uruchomienie:  py tools/make_report_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # by importowac pakiet src.* w generatorach wykresow
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# --- paleta (spojna z ciemnym/jasnym tlem GitHuba) ---
C_BG = "none"
C_AXIS = "#8b949e"
C_TEXT = "#57606a"
C_GRID = "#d0d7de"
C_BLUE = "#0969da"
C_GREEN = "#1a7f37"
C_RED = "#cf222e"
C_ORANGE = "#bc4c00"
C_PURPLE = "#8250df"
C_GRAY = "#afb8c1"


def _svg_open(w: int, h: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">']


def _text(x, y, s, size=12, anchor="middle", color=C_TEXT, weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}" font-weight="{weight}">{s}</text>')


def save(name: str, parts: list[str]):
    parts.append("</svg>")
    (ASSETS / name).write_text("\n".join(parts), encoding="utf-8")
    print(f"  zapisano docs/assets/{name}")


# ----------------------------------------------------------------------------
# 1. WYKRES CENY (linia, weekly close) + tlo logarytmiczne nie jest potrzebne,
#    ale skala jest ogromna (30 -> 462), wiec uzywamy skali liniowej z siatka.
# ----------------------------------------------------------------------------
def chart_price():
    df = pd.read_csv(ROOT / "data" / "scw_d.csv")
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    w = df["close"].resample("W").last().dropna()

    W, H = 860, 360
    ml, mr, mt, mb = 55, 20, 30, 40
    pw, ph = W - ml - mr, H - mt - mb
    ymax = float(w.max()) * 1.05
    ymin = 0
    n = len(w)

    def X(i): return ml + pw * i / (n - 1)
    def Y(v): return mt + ph * (1 - (v - ymin) / (ymax - ymin))

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "SCANWAY (SCW.WA) - cena tygodniowa (close), 2023-2026",
                   14, "middle", C_TEXT, "bold"))
    # siatka pozioma
    for gv in [0, 100, 200, 300, 400]:
        y = Y(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" '
                 f'stroke="{C_GRID}" stroke-width="1"/>')
        p.append(_text(ml - 8, y + 4, f"{gv}", 11, "end", C_TEXT))
    # os X - lata
    years = {}
    for i, d in enumerate(w.index):
        years.setdefault(d.year, i)
    for yr, i in years.items():
        p.append(_text(X(i), H - 12, str(yr), 11, "middle", C_TEXT))
    # linia ceny
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(w.values))
    p.append(f'<polyline fill="none" stroke="{C_BLUE}" stroke-width="2" points="{pts}"/>')
    # ostatni punkt
    p.append(f'<circle cx="{X(n-1):.1f}" cy="{Y(w.iloc[-1]):.1f}" r="3.5" fill="{C_BLUE}"/>')
    p.append(_text(X(n-1), Y(w.iloc[-1]) - 8, f"{w.iloc[-1]:.0f} PLN", 11, "end", C_BLUE, "bold"))
    save("01_cena.svg", p)


# ----------------------------------------------------------------------------
# 2. SEZONOWOSC DZIENNA - sredni zwrot wg dnia tygodnia (z CSV)
# ----------------------------------------------------------------------------
def chart_dow():
    df = pd.read_csv(ROOT / "data" / "scw_d.csv")
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["ret"] = df["close"].pct_change() * 100
    df["dow"] = df.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pl = {"Monday": "Pon", "Tuesday": "Wt", "Wednesday": "Sr",
          "Thursday": "Czw", "Friday": "Pt"}
    g = df.groupby("dow")["ret"].mean().reindex(order)

    W, H = 560, 320
    ml, mr, mt, mb = 50, 20, 36, 36
    pw, ph = W - ml - mr, H - mt - mb
    vmax = 0.6
    bw = pw / len(g) * 0.6

    def Y(v): return mt + ph * (1 - v / vmax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "Sredni zwrot dzienny wg dnia tygodnia (669 sesji)",
                   13, "middle", C_TEXT, "bold"))
    p.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}" stroke="{C_AXIS}" stroke-width="1.5"/>')
    for gv in [0.2, 0.4, 0.6]:
        y = Y(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
        p.append(_text(ml - 8, y + 4, f"{gv:.1f}%", 10, "end", C_TEXT))
    for i, (d, v) in enumerate(g.items()):
        cx = ml + pw * (i + 0.5) / len(g)
        x = cx - bw / 2
        y = Y(v)
        col = C_GREEN if d == "Friday" else C_BLUE
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{Y(0)-y:.1f}" fill="{col}" rx="2"/>')
        p.append(_text(cx, y - 6, f"{v:.2f}", 10, "middle", C_TEXT, "bold"))
        p.append(_text(cx, H - 14, pl[d], 11, "middle", C_TEXT))
    save("02_sezonowosc_dzienna.svg", p)


# ----------------------------------------------------------------------------
# 3. SEZONOWOSC SRODDZIENNA - % dni, gdy dana godzina = szczyt dnia
#    (rzeczywiste dane z update-data; 09:00 = 41% -> kluczowy sygnal)
# ----------------------------------------------------------------------------
INTRADAY = [  # (godzina, pct_of_days_this_is_daily_high)  -- snapshot z update-data
    ("09:00", 41.07), ("10:00", 1.79), ("10:15", 7.14), ("10:30", 7.27),
    ("10:45", 7.14), ("11:00", 8.93), ("11:15", 3.51), ("11:30", 1.75),
    ("12:30", 1.75), ("12:45", 7.02),
]


def chart_intraday():
    data = INTRADAY
    W, H = 720, 320
    ml, mr, mt, mb = 45, 20, 40, 40
    pw, ph = W - ml - mr, H - mt - mb
    vmax = 45
    bw = pw / len(data) * 0.62

    def Y(v): return mt + ph * (1 - v / vmax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 16, "Jak czesto dana godzina jest SZCZYTEM dnia (% sesji, 57 dni)",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 32, "otwarcie 09:00 jest dziennym maksimum w 41% sesji - najsilniejszy sygnal sprzedazy",
                   10, "middle", C_RED))
    p.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}" stroke="{C_AXIS}" stroke-width="1.5"/>')
    for gv in [10, 20, 30, 40]:
        y = Y(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
        p.append(_text(ml - 8, y + 4, f"{gv}%", 10, "end", C_TEXT))
    for i, (t, v) in enumerate(data):
        cx = ml + pw * (i + 0.5) / len(data)
        x = cx - bw / 2
        y = Y(v)
        col = C_RED if t == "09:00" else C_GRAY
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{Y(0)-y:.1f}" fill="{col}" rx="2"/>')
        if v > 5:
            p.append(_text(cx, y - 5, f"{v:.0f}", 9, "middle", C_TEXT, "bold"))
        p.append(f'<text x="{cx:.1f}" y="{H-12:.1f}" font-size="9" text-anchor="end" '
                 f'fill="{C_TEXT}" transform="rotate(-45 {cx:.1f} {H-12:.1f})">{t}</text>')
    save("03_sezonowosc_sroddzienna.svg", p)


# ----------------------------------------------------------------------------
# 4. BILANS KLAS targetu (z backtestu: precision baseline 0.774 -> 77/23)
# ----------------------------------------------------------------------------
def chart_balance():
    W, H = 420, 200
    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Bilans klas targetu (1847 probek)", 13, "middle", C_TEXT, "bold"))
    total = 360
    ml = 40
    y0 = 70
    bh = 46
    neg = 0.774
    wn = total * neg
    p.append(f'<rect x="{ml}" y="{y0}" width="{wn:.1f}" height="{bh}" fill="{C_GRAY}" rx="3"/>')
    p.append(f'<rect x="{ml+wn:.1f}" y="{y0}" width="{total-wn:.1f}" height="{bh}" fill="{C_RED}" rx="3"/>')
    p.append(_text(ml + wn / 2, y0 + bh / 2 + 4, "0: brak szczytu  77%", 11, "middle", "#ffffff", "bold"))
    p.append(_text(ml + wn + (total - wn) / 2, y0 + bh / 2 + 4, "23%", 10, "middle", "#ffffff", "bold"))
    p.append(_text(W / 2, 150, "Niezbalansowanie 77/23 sprawia, ze F1 faworyzuje baseline,", 10, "middle", C_TEXT))
    p.append(_text(W / 2, 165, "ktory zawsze typuje klase wiekszosciowa (klasa 0).", 10, "middle", C_TEXT))
    save("04_bilans_klas.svg", p)


# ----------------------------------------------------------------------------
# 5. POROWNANIE MODELI: F1 vs ROC AUC (rzeczywiste wyniki backtestu)
# ----------------------------------------------------------------------------
MODELS = [  # (nazwa, avg_f1, avg_roc_auc) -- snapshot z `py main.py backtest`
    ("baseline_most_frequent", 0.871, 0.500),
    ("gradient_boosting",      0.818, 0.605),
    ("knn",                    0.803, 0.580),
    ("svm_rbf",                0.766, 0.599),
    ("logistic_regression",   0.688, 0.642),
    ("random_forest",         0.671, 0.630),
    ("extra_trees",           0.658, 0.611),
]


def chart_models():
    data = MODELS
    W, H = 820, 380
    ml, mr, mt, mb = 160, 20, 60, 30
    pw, ph = W - ml - mr, H - mt - mb
    rowh = ph / len(data)
    barh = rowh * 0.34

    def Xf1(v): return ml + pw * v          # F1 skala 0..1
    def Xauc(v): return ml + pw * v         # AUC skala 0..1

    p = _svg_open(W, H)
    p.append(_text(W / 2, 22, "Porownanie modeli: F1 (zwodniczy) vs ROC AUC (wlasciwy)",
                   14, "middle", C_TEXT, "bold"))
    # legenda
    p.append(f'<rect x="{ml}" y="36" width="14" height="10" fill="{C_GRAY}" rx="2"/>')
    p.append(_text(ml + 20, 45, "avg F1", 11, "start", C_TEXT))
    p.append(f'<rect x="{ml+90}" y="36" width="14" height="10" fill="{C_BLUE}" rx="2"/>')
    p.append(_text(ml + 110, 45, "avg ROC AUC", 11, "start", C_TEXT))
    # linia AUC=0.5 (losowy)
    x05 = Xauc(0.5)
    p.append(f'<line x1="{x05:.1f}" y1="{mt}" x2="{x05:.1f}" y2="{mt+ph}" stroke="{C_RED}" stroke-width="1" stroke-dasharray="4 3"/>')
    p.append(_text(x05, mt - 4, "AUC 0.5 = losowy", 9, "middle", C_RED))
    for i, (name, f1, auc) in enumerate(data):
        cy = mt + rowh * (i + 0.5)
        is_best = name == "logistic_regression"
        is_base = name == "baseline_most_frequent"
        lab_col = C_GREEN if is_best else (C_RED if is_base else C_TEXT)
        lab_w = "bold" if (is_best or is_base) else "normal"
        p.append(_text(ml - 8, cy + 3, name, 10, "end", lab_col, lab_w))
        # F1 bar
        p.append(f'<rect x="{ml}" y="{cy-barh-1:.1f}" width="{pw*f1:.1f}" height="{barh:.1f}" fill="{C_GRAY}" rx="2"/>')
        p.append(_text(Xf1(f1) + 4, cy - 3, f"{f1:.3f}", 9, "start", C_TEXT))
        # AUC bar
        aucol = C_GREEN if is_best else (C_RED if is_base else C_BLUE)
        p.append(f'<rect x="{ml}" y="{cy+1:.1f}" width="{pw*auc:.1f}" height="{barh:.1f}" fill="{aucol}" rx="2"/>')
        p.append(_text(Xauc(auc) + 4, cy + barh + 1, f"{auc:.3f}", 9, "start", C_TEXT))
    save("05_porownanie_modeli.svg", p)


# ----------------------------------------------------------------------------
# 6. RANKING wg ROC AUC (po wykluczeniu baseline) - co naprawde wybieramy
# ----------------------------------------------------------------------------
def chart_ranking():
    data = sorted([(n, a) for n, _, a in MODELS if n != "baseline_most_frequent"],
                  key=lambda t: t[1], reverse=True)
    W, H = 720, 300
    ml, mr, mt, mb = 160, 30, 50, 20
    pw, ph = W - ml - mr, H - mt - mb
    rowh = ph / len(data)
    barh = rowh * 0.55

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Ranking wg ROC AUC (baseline wykluczony) - wybor produkcyjny",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 38, "skala 0.50 (losowy) - 0.65", 10, "middle", C_TEXT))
    lo, hi = 0.50, 0.66

    def X(v): return ml + pw * (v - lo) / (hi - lo)
    for gv in [0.50, 0.55, 0.60, 0.65]:
        x = X(gv)
        p.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="{C_GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
        p.append(_text(x, mt + ph + 15, f"{gv:.2f}", 9, "middle", C_TEXT))
    for i, (name, auc) in enumerate(data):
        cy = mt + rowh * (i + 0.5)
        best = i == 0
        col = C_GREEN if best else C_BLUE
        p.append(_text(ml - 8, cy + 3, name, 10, "end",
                       C_GREEN if best else C_TEXT, "bold" if best else "normal"))
        p.append(f'<rect x="{ml}" y="{cy-barh/2:.1f}" width="{X(auc)-ml:.1f}" height="{barh:.1f}" fill="{col}" rx="2"/>')
        tag = f"{auc:.3f}" + ("  &#8592; WYBRANY" if best else "")
        p.append(_text(X(auc) + 5, cy + 3, tag, 10, "start", C_GREEN if best else C_TEXT,
                       "bold" if best else "normal"))
    save("06_ranking_auc.svg", p)


# ----------------------------------------------------------------------------
# 7. KOMPROMIS yfinance: interwal vs maksymalna historia (dlaczego 15m)
# ----------------------------------------------------------------------------
# (interwal, max dni wstecz, swiec na sesje GPW ~8h)  -- limity yfinance
INTERVALS = [
    ("1m", 7, 480),
    ("5m", 60, 96),
    ("15m", 60, 33),   # <- wybor projektu
    ("30m", 60, 16),
    ("1h", 730, 8),
]


def chart_intervals():
    data = INTERVALS
    W, H = 720, 320
    ml, mr, mt, mb = 60, 60, 50, 40
    pw, ph = W - ml - mr, H - mt - mb
    daymax = 730
    bw = pw / len(data) * 0.5

    def Y(v): return mt + ph * (1 - v / daymax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "yfinance: im drobniejszy interwal, tym krotsza historia",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 34, "15m to kompromis: ~60 dni (=~57 sesji) przy sensownej liczbie swiec/sesje",
                   10, "middle", C_TEXT))
    for gv in [7, 60, 365, 730]:
        y = Y(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1" stroke-dasharray="3 3"/>')
        p.append(_text(ml - 8, y + 4, f"{gv}d", 9, "end", C_TEXT))
    for i, (iv, days, bars) in enumerate(data):
        cx = ml + pw * (i + 0.5) / len(data)
        x = cx - bw / 2
        y = Y(days)
        chosen = iv == "15m"
        col = C_GREEN if chosen else C_BLUE
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{Y(0)-y:.1f}" fill="{col}" rx="2"/>')
        p.append(_text(cx, y - 6, f"{days}d", 10, "middle", C_TEXT, "bold"))
        lab = iv + ("  (wybor)" if chosen else "")
        p.append(_text(cx, H - 20, lab, 10, "middle",
                       C_GREEN if chosen else C_TEXT, "bold" if chosen else "normal"))
        p.append(_text(cx, H - 8, f"~{bars} swiec/sesje", 8, "middle", C_TEXT))
    save("07_interwaly_yfinance.svg", p)


# ----------------------------------------------------------------------------
# 8. REPLAY: odtworzenie sesji - cena + alerty + ground truth (2 panele)
# ----------------------------------------------------------------------------
def _load_real_or_synthetic():
    """Zwraca (intraday_df, daily_ctx, cfg, is_real, etykieta_zrodla).

    Preferuje PRAWDZIWE dane intraday (snapshot/live przez yfinance). Tylko gdy
    realnych danych brak (np. swiezy klon bez snapshotu i bez internetu),
    spada na sesje syntetyczna - wtedy wykresy oznaczamy jako ilustracyjne.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from src.config import load_config
    from src.data_sources import get_daily_history
    from src.features import build_daily_context_features
    from src.replay import load_intraday_for_replay, make_synthetic_session

    cfg = load_config()
    try:
        intraday, source = load_intraday_for_replay(cfg, synthetic=False)
        if intraday.empty:
            raise ValueError("pusto")
        daily = get_daily_history(cfg, refresh_live=False)
        daily_ctx = build_daily_context_features(daily)
        return intraday, daily_ctx, cfg, True, source
    except Exception as e:
        print(f"  [uwaga] brak realnych danych intraday ({e}); uzywam syntetycznych")
        return make_synthetic_session(cfg, n_sessions=40, seed=42), None, cfg, False, "syntetyczne"


def _compute_replay():
    """Liczy replay out-of-sample na realnej (lub awaryjnie syntetycznej)
    ostatniej sesji, tak jak komenda `replay`. Zwraca (rep, cfg, is_real)."""
    from src.features import build_features
    from src.replay import replay_session, train_excluding_session

    intraday, daily_ctx, cfg, is_real, _ = _load_real_or_synthetic()
    feat_df, cols = build_features(intraday, daily_ctx, cfg)
    holdout = sorted(feat_df["date"].unique())[-1]
    model, _, _ = train_excluding_session(feat_df, cols, cfg, "logistic_regression", holdout)
    rep = replay_session(intraday, daily_ctx, cfg, model, cols, session_date=holdout)
    return rep, cfg, is_real, holdout


def chart_replay():
    rep, cfg, is_real, holdout = _compute_replay()
    n = len(rep)
    W, H = 860, 460
    ml, mr = 55, 20
    # panel ceny
    p1t, p1b = 50, 250
    # panel proby
    p2t, p2b = 300, 420
    pw = W - ml - mr

    prices = rep["close"].values
    pmin, pmax = prices.min(), prices.max()
    pad = (pmax - pmin) * 0.15 or 1
    pmin -= pad; pmax += pad

    def X(i): return ml + pw * i / (n - 1)
    def Yp(v): return p1t + (p1b - p1t) * (1 - (v - pmin) / (pmax - pmin))
    def Yq(v): return p2t + (p2b - p2t) * (1 - v)  # proba 0..1

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Replay sesji: gdzie model wysyla ALERT vs realne szczyty",
                   14, "middle", C_TEXT, "bold"))
    sub = (f"PRAWDZIWA sesja {holdout} (out-of-sample) - model jej nie widzial na treningu"
           if is_real else "sesja ILUSTRACYJNA (syntetyczna) - brak realnych danych")
    p.append(_text(W / 2, 36, sub, 9, "middle", (C_TEXT if is_real else C_RED)))

    # --- panel 1: cena ---
    p.append(_text(ml, p1t - 6, "Cena (PLN)", 10, "start", C_TEXT, "bold"))
    # pasma ground-truth (target=1) jako jasne tlo
    for i, tg in enumerate(rep["target"].values):
        if tg == 1:
            x = X(i) - (pw / n / 2)
            p.append(f'<rect x="{x:.1f}" y="{p1t}" width="{pw/n:.1f}" height="{p1b-p1t:.1f}" fill="#fff1e5"/>')
    # linia ceny
    pts = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(prices))
    p.append(f'<polyline fill="none" stroke="{C_BLUE}" stroke-width="2" points="{pts}"/>')
    # szczyt dnia
    hi_idx = int(rep["is_session_high"].values.argmax())
    p.append(f'<circle cx="{X(hi_idx):.1f}" cy="{Yp(prices[hi_idx]):.1f}" r="5" fill="{C_RED}"/>')
    p.append(_text(X(hi_idx), Yp(prices[hi_idx]) - 10, "szczyt dnia", 9, "middle", C_RED, "bold"))
    # alerty (zielone trojkaty nad cena)
    for i, al in enumerate(rep["alert"].values):
        if al:
            x, y = X(i), Yp(prices[i]) - 4
            p.append(f'<polygon points="{x-4:.1f},{y-7:.1f} {x+4:.1f},{y-7:.1f} {x:.1f},{y:.1f}" fill="{C_GREEN}"/>')
    # os czasu (co 4 swiece)
    for i in range(0, n, 4):
        p.append(_text(X(i), p1b + 14, rep["time"].iloc[i], 8, "middle", C_TEXT))

    # --- panel 2: prawdopodobienstwo ---
    p.append(_text(ml, p2t - 6, "Prawdopodobienstwo szczytu", 10, "start", C_TEXT, "bold"))
    for gv in [0.0, 0.5, 1.0]:
        y = Yq(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        p.append(_text(ml - 6, y + 3, f"{gv:.0%}", 8, "end", C_TEXT))
    # prog alertu
    yth = Yq(0.6)
    p.append(f'<line x1="{ml}" y1="{yth:.1f}" x2="{ml+pw}" y2="{yth:.1f}" stroke="{C_RED}" stroke-width="1.5" stroke-dasharray="5 3"/>')
    p.append(_text(ml + pw, yth - 4, "prog 60%", 9, "end", C_RED))
    qpts = " ".join(f"{X(i):.1f},{Yq(v):.1f}" for i, v in enumerate(rep["proba"].values))
    p.append(f'<polyline fill="none" stroke="{C_PURPLE}" stroke-width="2" points="{qpts}"/>')
    for i, v in enumerate(rep["proba"].values):
        col = C_GREEN if rep["alert"].iloc[i] else C_GRAY
        p.append(f'<circle cx="{X(i):.1f}" cy="{Yq(v):.1f}" r="2.5" fill="{col}"/>')

    # legenda
    ly = H - 8
    p.append(f'<rect x="{ml}" y="{ly-9}" width="14" height="9" fill="#fff1e5" stroke="{C_ORANGE}" stroke-width="0.5"/>')
    p.append(_text(ml + 18, ly - 1, "realny moment sprzedazy (target=1)", 9, "start", C_TEXT))
    p.append(f'<polygon points="{ml+250},{ly-9} {ml+258},{ly-9} {ml+254},{ly-2}" fill="{C_GREEN}"/>')
    p.append(_text(ml + 264, ly - 1, "alert modelu", 9, "start", C_TEXT))
    p.append(f'<circle cx="{ml+360}" cy="{ly-4}" r="4" fill="{C_RED}"/>')
    p.append(_text(ml + 370, ly - 1, "szczyt dnia", 9, "start", C_TEXT))
    save("08_replay_sesja.svg", p)

    # zwroc statystyki sesji do raportu
    n_alert = int(rep["alert"].sum())
    n_target = int(rep["target"].sum())
    hits = int(((rep["alert"]) & (rep["target"] == 1)).sum())
    print(f"  [replay] alerty={n_alert} target={n_target} trafione={hits} "
          f"high@{rep['time'].iloc[hi_idx]}")


# ----------------------------------------------------------------------------
# 9. CO MIERZY ROC AUC: rozklad wynikow modelu dla obu klas (na sesji testowej)
# ----------------------------------------------------------------------------
def chart_roc_distributions():
    rep, cfg, is_real, holdout = _compute_replay()
    pos = rep.loc[rep["target"] == 1, "proba"].values  # realne momenty sprzedazy
    neg = rep.loc[rep["target"] == 0, "proba"].values  # reszta

    W, H = 720, 340
    ml, mr, mt, mb = 50, 20, 70, 40
    pw, ph = W - ml - mr, H - mt - mb
    nbins = 10
    edges = [i / nbins for i in range(nbins + 1)]

    def hist(vals):
        import numpy as np
        h, _ = np.histogram(vals, bins=edges)
        return h
    hp, hn = hist(pos), hist(neg)
    hmax = max(hp.max(), hn.max(), 1)

    def X(b): return ml + pw * b / nbins
    def Y(c): return mt + ph * (1 - c / hmax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Co mierzy ROC AUC: rozdzielenie wynikow modelu dla obu klas",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 38, "AUC = szansa, ze losowy 'moment sprzedazy' dostaje WYZSZY wynik niz losowy 'nie-szczyt'",
                   9, "middle", C_TEXT))
    p.append(_text(W / 2, 52, "Im mniejsze nakladanie sie slupkow, tym wyzszy AUC.", 9, "middle", C_TEXT))
    bw = pw / nbins
    for b in range(nbins):
        # neg (szary) i pos (pomaranczowy) obok siebie w binie
        xb = X(b)
        p.append(f'<rect x="{xb+1:.1f}" y="{Y(hn[b]):.1f}" width="{bw/2-1:.1f}" height="{Y(0)-Y(hn[b]):.1f}" fill="{C_GRAY}" opacity="0.9"/>')
        p.append(f'<rect x="{xb+bw/2:.1f}" y="{Y(hp[b]):.1f}" width="{bw/2-1:.1f}" height="{Y(0)-Y(hp[b]):.1f}" fill="{C_ORANGE}" opacity="0.9"/>')
    p.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}" stroke="{C_AXIS}" stroke-width="1"/>')
    for b in range(nbins + 1):
        p.append(_text(X(b), Y(0) + 14, f"{edges[b]:.1f}", 8, "middle", C_TEXT))
    p.append(_text(W / 2, H - 8, "wynik modelu = prawdopodobienstwo szczytu", 9, "middle", C_TEXT))
    # legenda
    p.append(f'<rect x="{ml}" y="{mt-2}" width="12" height="9" fill="{C_ORANGE}"/>')
    p.append(_text(ml + 16, mt + 6, "realny moment sprzedazy (target=1)", 9, "start", C_TEXT))
    p.append(f'<rect x="{ml+230}" y="{mt-2}" width="12" height="9" fill="{C_GRAY}"/>')
    p.append(_text(ml + 246, mt + 6, "nie-szczyt (target=0)", 9, "start", C_TEXT))
    save("09_roc_rozklady.svg", p)


# ----------------------------------------------------------------------------
# 10. KRZYWA ROC: sesja testowa + odniesienie do realnego backtestu (AUC 0.642)
# ----------------------------------------------------------------------------
def chart_roc_curve():
    from sklearn.metrics import roc_auc_score, roc_curve
    rep, cfg, is_real, holdout = _compute_replay()
    fpr, tpr, _ = roc_curve(rep["target"], rep["proba"])
    auc_sess = roc_auc_score(rep["target"], rep["proba"])

    # krzywa orientacyjna dla AUC=0.642 (realny backtest): tpr = fpr**k, AUC=1/(k+1)
    auc_real = 0.642
    k = 1 / auc_real - 1
    ref = [(i / 50, (i / 50) ** k) for i in range(51)]

    W, H = 460, 440
    ml, mr, mt, mb = 55, 20, 50, 50
    pw, ph = W - ml - mr, H - mt - mb

    def X(v): return ml + pw * v
    def Y(v): return mt + ph * (1 - v)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Krzywa ROC", 14, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 36, "im wyzej i lewiej, tym lepiej (wieksze pole pod krzywa = AUC)",
                   9, "middle", C_TEXT))
    # ramka + siatka
    for g in [0, 0.25, 0.5, 0.75, 1.0]:
        p.append(f'<line x1="{X(g):.1f}" y1="{Y(0):.1f}" x2="{X(g):.1f}" y2="{Y(1):.1f}" stroke="{C_GRID}" stroke-width="0.7"/>')
        p.append(f'<line x1="{X(0):.1f}" y1="{Y(g):.1f}" x2="{X(1):.1f}" y2="{Y(g):.1f}" stroke="{C_GRID}" stroke-width="0.7"/>')
        p.append(_text(X(g), Y(0) + 14, f"{g:.2f}", 8, "middle", C_TEXT))
        p.append(_text(X(0) - 6, Y(g) + 3, f"{g:.2f}", 8, "end", C_TEXT))
    # przekatna = losowy (AUC 0.5)
    p.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}" stroke="{C_RED}" stroke-width="1.5" stroke-dasharray="5 3"/>')
    p.append(_text(X(0.72), Y(0.62), "losowy (0.50)", 9, "start", C_RED))
    # krzywa orientacyjna realnego backtestu
    rpts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in ref)
    p.append(f'<polyline fill="none" stroke="{C_BLUE}" stroke-width="2" stroke-dasharray="2 2" points="{rpts}"/>')
    p.append(_text(X(0.55), Y(0.74), "realny backtest ~0.642", 9, "start", C_BLUE, "bold"))
    # krzywa sesji testowej
    spts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(fpr, tpr))
    p.append(f'<polyline fill="none" stroke="{C_GREEN}" stroke-width="2.5" points="{spts}"/>')
    p.append(_text(X(0.30), Y(0.96), f"sesja testowa (AUC {auc_sess:.2f})", 9, "start", C_GREEN, "bold"))
    p.append(_text(W / 2, H - 26, "FPR (falszywe alarmy / nie-szczyty)", 10, "middle", C_TEXT))
    p.append(f'<text x="16" y="{(mt+ph/2):.1f}" font-size="10" text-anchor="middle" fill="{C_TEXT}" transform="rotate(-90 16 {(mt+ph/2):.1f})">TPR (zlapane momenty sprzedazy)</text>')
    save("10_roc_krzywa.svg", p)


def _compute_evaluation():
    """Liczy ocene dzien-po-dniu na realnych (lub awaryjnie syntetycznych)
    danych, split 80/20, tak jak komenda `evaluate`."""
    from src.evaluate import daywise_evaluation, logistic_formula
    from src.features import build_features

    intraday, daily_ctx, cfg, is_real, _ = _load_real_or_synthetic()
    feat_df, cols = build_features(intraday, daily_ctx, cfg)
    per_day, pooled, model = daywise_evaluation(feat_df, cols, cfg, "logistic_regression", 0.8)
    fml = logistic_formula(model, cols)
    return per_day, pooled, fml, is_real


# ----------------------------------------------------------------------------
# 11. OCENA DZIEN-PO-DNIU: ROC AUC kazdego dnia testowego vs linia 0.5
# ----------------------------------------------------------------------------
def chart_daywise_auc():
    per_day, pooled, _, is_real = _compute_evaluation()
    aucs = per_day["auc"].fillna(0.5).tolist()
    labels = [str(d)[5:] for d in per_day["date"]]  # MM-DD

    W, H = 720, 340
    ml, mr, mt, mb = 50, 20, 56, 50
    pw, ph = W - ml - mr, H - mt - mb
    bw = pw / len(aucs) * 0.6

    def Y(v): return mt + ph * (1 - v)  # AUC 0..1

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "ROC AUC dla KAZDEGO dnia testowego (split 80/20)",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 36, f"{pooled['days_beat_random']}/{pooled['days_total']} dni powyzej 0.50 (losowego)",
                   10, "middle", C_GREEN, "bold"))
    for gv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = Y(gv)
        col = C_RED if gv == 0.5 else C_GRID
        wdt = "1.5" if gv == 0.5 else "1"
        dash = ' stroke-dasharray="5 3"' if gv == 0.5 else ""
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{col}" stroke-width="{wdt}"{dash}/>')
        p.append(_text(ml - 8, y + 4, f"{gv:.2f}", 9, "end", C_TEXT))
    p.append(_text(ml + pw, Y(0.5) - 4, "losowy 0.50", 9, "end", C_RED))
    for i, v in enumerate(aucs):
        cx = ml + pw * (i + 0.5) / len(aucs)
        x = cx - bw / 2
        y = Y(v)
        col = C_GREEN if v > 0.5 else C_RED
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{Y(0)-y:.1f}" fill="{col}" rx="2"/>')
        p.append(_text(cx, y - 5, f"{v:.2f}", 8, "middle", C_TEXT))
        p.append(f'<text x="{cx:.1f}" y="{Y(0)+14:.1f}" font-size="8" text-anchor="end" '
                 f'fill="{C_TEXT}" transform="rotate(-45 {cx:.1f} {Y(0)+14:.1f})">{labels[i]}</text>')
    save("11_ocena_dzienna.svg", p)


# ----------------------------------------------------------------------------
# 12. TEST PERMUTACYJNY: rozklad AUC modelu losowego vs nasz wynik
# ----------------------------------------------------------------------------
def chart_permutation():
    import numpy as np
    _, pooled, _, is_real = _compute_evaluation()
    null = pooled["perm_null"]
    obs = pooled["pooled_auc"]

    W, H = 720, 320
    ml, mr, mt, mb = 50, 20, 70, 45
    pw, ph = W - ml - mr, H - mt - mb
    lo, hi = 0.35, max(0.72, obs + 0.03)
    nbins = 28
    counts, edges = np.histogram(null, bins=nbins, range=(lo, hi))
    cmax = max(counts.max(), 1)

    def X(v): return ml + pw * (v - lo) / (hi - lo)
    def Y(c): return mt + ph * (1 - c / cmax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "Dowod 'lepiej niz losowo': test permutacyjny",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 36, "szare = 1000 modeli LOSOWYCH (przetasowane etykiety); zielona linia = nasz model",
                   9, "middle", C_TEXT))
    p.append(_text(W / 2, 50, f"p-value = {pooled['p_value']:.4f}  (zaden z 1000 losowych nie pobil naszego wyniku)",
                   10, "middle", C_GREEN, "bold"))
    bw = pw / nbins
    for i, c in enumerate(counts):
        if c == 0:
            continue
        x = X(edges[i])
        p.append(f'<rect x="{x:.1f}" y="{Y(c):.1f}" width="{bw-1:.1f}" height="{Y(0)-Y(c):.1f}" fill="{C_GRAY}"/>')
    p.append(f'<line x1="{ml}" y1="{Y(0):.1f}" x2="{ml+pw}" y2="{Y(0):.1f}" stroke="{C_AXIS}" stroke-width="1"/>')
    # linia 0.5 (srodek losowego)
    p.append(f'<line x1="{X(0.5):.1f}" y1="{mt}" x2="{X(0.5):.1f}" y2="{Y(0):.1f}" stroke="{C_RED}" stroke-width="1" stroke-dasharray="3 3"/>')
    p.append(_text(X(0.5), mt - 2, "0.50", 9, "middle", C_RED))
    # nasz obserwowany AUC
    p.append(f'<line x1="{X(obs):.1f}" y1="{mt}" x2="{X(obs):.1f}" y2="{Y(0):.1f}" stroke="{C_GREEN}" stroke-width="2.5"/>')
    p.append(_text(X(obs), mt - 2, f"nasz model {obs:.3f}", 10, "middle", C_GREEN, "bold"))
    for gv in [0.4, 0.5, 0.6, 0.7]:
        if lo <= gv <= hi:
            p.append(_text(X(gv), Y(0) + 14, f"{gv:.1f}", 9, "middle", C_TEXT))
    p.append(_text(W / 2, H - 8, "ROC AUC", 9, "middle", C_TEXT))
    save("12_permutacja.svg", p)


# ----------------------------------------------------------------------------
# 13. FORMULA: wagi (wspolczynniki) regresji logistycznej
# ----------------------------------------------------------------------------
def chart_formula():
    # wspolczynniki MODELU PEAK (dzienne maksimum) - trenowany na calym zbiorze
    from src.features import build_features
    from src.peak import add_daily_high_target, train_peak_model
    intraday, daily_ctx, cfg, is_real, _ = _load_real_or_synthetic()
    feat_df, cols = build_features(intraday, daily_ctx, cfg)
    feat_df = add_daily_high_target(feat_df)
    model = train_peak_model(feat_df, cols, cfg)
    clf = model.named_steps["clf"]
    import pandas as pd
    terms = pd.DataFrame({"feature": cols, "coef_std": clf.coef_.ravel()}) \
        .sort_values("coef_std", key=abs, ascending=False).reset_index(drop=True)
    feats = terms["feature"].tolist()
    coefs = terms["coef_std"].tolist()

    W, H = 720, 300
    ml, mr, mt, mb = 160, 80, 50, 20
    pw, ph = W - ml - mr, H - mt - mb
    rowh = ph / len(feats)
    barh = rowh * 0.55
    amax = max(abs(min(coefs)), abs(max(coefs))) * 1.15 or 1
    x0 = ml + pw / 2  # zero w srodku

    def X(v): return x0 + (pw / 2) * v / amax

    p = _svg_open(W, H)
    p.append(_text(W / 2, 18, "Formula modelu PEAK: wplyw cech na p(dzienne maksimum)",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 36, "waga + (zielona) podnosi prawd. szczytu, - (czerwona) obniza; dlugosc = sila",
                   9, "middle", C_TEXT))
    p.append(f'<line x1="{x0:.1f}" y1="{mt}" x2="{x0:.1f}" y2="{mt+ph}" stroke="{C_AXIS}" stroke-width="1"/>')
    for i, (f, c) in enumerate(zip(feats, coefs)):
        cy = mt + rowh * (i + 0.5)
        p.append(_text(ml - 8, cy + 3, f, 10, "end", C_TEXT))
        col = C_GREEN if c >= 0 else C_RED
        x1, x2 = (x0, X(c)) if c >= 0 else (X(c), x0)
        p.append(f'<rect x="{x1:.1f}" y="{cy-barh/2:.1f}" width="{abs(x2-x1):.1f}" height="{barh:.1f}" fill="{col}" rx="2"/>')
        tx = X(c) + (6 if c >= 0 else -6)
        anc = "start" if c >= 0 else "end"
        p.append(_text(tx, cy + 3, f"{c:+.3f}", 9, anc, C_TEXT, "bold"))
    save("13_formula.svg", p)


def _compute_peak():
    """Liczy ocene trybu peak na realnych (lub awaryjnie syntetycznych) danych,
    tak jak komenda `peak`. Zwraca (per_day, summary, sweep, is_real)."""
    from src.features import build_features
    from src.peak import daily_peak_evaluation, penalty_sweep

    intraday, daily_ctx, cfg, is_real, _ = _load_real_or_synthetic()
    feat_df, cols = build_features(intraday, daily_ctx, cfg)
    per_day, summary, _ = daily_peak_evaluation(feat_df, cols, cfg, train_frac=0.8)
    sweep = penalty_sweep(feat_df, cols, cfg, train_frac=0.8)
    return per_day, summary, sweep, is_real


# ----------------------------------------------------------------------------
# 14. PEAK - jak blisko szczytu sprzedajesz (regret vs strategie odniesienia)
# ----------------------------------------------------------------------------
def chart_peak_regret():
    _, summary, _, is_real = _compute_peak()
    data = [
        ("MODEL (peak)", summary["mean_regret_model"], C_GREEN),
        ("otwarcie 09:00", summary["mean_regret_open"], C_GRAY),
        ("losowo", summary["mean_regret_random"], C_GRAY),
        ("do zamkniecia", summary["mean_regret_close"], C_GRAY),
        ("idealnie", 0.0, C_BLUE),
    ]
    W, H = 720, 340
    ml, mr, mt, mb = 150, 60, 56, 30
    pw, ph = W - ml - mr, H - mt - mb
    rowh = ph / len(data)
    barh = rowh * 0.55
    vmax = max(v for _, v, _ in data) * 1.25 or 1

    def X(v): return ml + pw * v / vmax

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Jak blisko dziennego szczytu sprzedajesz? (mniej = lepiej)",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 38, "regret = sredni % PONIZEJ dziennego maksimum; "
                   f"p-value vs losowy = {summary['p_value']:.4f}",
                   9, "middle", C_GREEN if summary["p_value"] < 0.05 else C_TEXT))
    for i, (lab, v, col) in enumerate(data):
        cy = mt + rowh * (i + 0.5)
        best = i == 0
        p.append(_text(ml - 8, cy + 3, lab, 10, "end",
                       C_GREEN if best else C_TEXT, "bold" if best else "normal"))
        p.append(f'<rect x="{ml}" y="{cy-barh/2:.1f}" width="{max(X(v)-ml,1):.1f}" height="{barh:.1f}" fill="{col}" rx="2"/>')
        p.append(_text(X(v) + 6, cy + 3, f"{v:.2f}%", 10, "start", C_TEXT, "bold" if best else "normal"))
    src = "realne dane" if is_real else "dane syntetyczne (ilustracja)"
    p.append(_text(W / 2, H - 8, f"sredni regret na sesjach testowych ({src})", 9, "middle", C_TEXT))
    save("14_peak_regret.svg", p)


# ----------------------------------------------------------------------------
# 15. PEAK - os czasu: gdzie byl szczyt vs gdzie alert (kazdy dzien testowy)
# ----------------------------------------------------------------------------
def chart_peak_timeline():
    per_day, summary, _, is_real = _compute_peak()
    W, H = 720, 360
    ml, mr, mt, mb = 90, 20, 56, 40
    pw, ph = W - ml - mr, H - mt - mb
    rows = len(per_day)
    rowh = ph / rows
    # os czasu 09:00 - 17:00 w minutach
    t0, t1 = 9 * 60, 17 * 60

    def to_min(s):
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    def X(mins): return ml + pw * (mins - t0) / (t1 - t0)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Tryb peak: faktyczny szczyt vs alert modelu (kazdy dzien)",
                   13, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 38, f"trafienia w +/-{summary['tolerance_bars']} swiec: "
                   f"{summary['hit_rate']:.0%} dni | alert odpalil w "
                   f"{summary['days_fired']}/{summary['n_test_sessions']} dni",
                   9, "middle", C_TEXT))
    # siatka godzin
    for hh in range(9, 18):
        x = X(hh * 60)
        p.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="{C_GRID}" stroke-width="0.7"/>')
        p.append(_text(x, mt + ph + 14, f"{hh}:00", 8, "middle", C_TEXT))
    for i, r in per_day.reset_index(drop=True).iterrows():
        cy = mt + rowh * (i + 0.5)
        p.append(_text(ml - 8, cy + 3, str(r["date"])[5:], 9, "end", C_TEXT))
        # faktyczny szczyt (czerwony)
        xs = X(to_min(r["szczyt_o"]))
        # linia laczaca alert-szczyt
        if r["alert_o"] != "(brak)":
            xa = X(to_min(r["alert_o"]))
            hit = r["trafiony"]
            lc = C_GREEN if hit else C_ORANGE
            p.append(f'<line x1="{xs:.1f}" y1="{cy:.1f}" x2="{xa:.1f}" y2="{cy:.1f}" stroke="{lc}" stroke-width="1.5"/>')
            p.append(f'<polygon points="{xa-4:.1f},{cy-6:.1f} {xa+4:.1f},{cy-6:.1f} {xa:.1f},{cy+1:.1f}" fill="{C_GREEN}"/>')
        else:
            p.append(_text(X(t1) + 0, cy + 3, "brak alertu", 8, "start", C_RED))
        p.append(f'<circle cx="{xs:.1f}" cy="{cy:.1f}" r="4" fill="{C_RED}"/>')
    # legenda
    ly = 50
    p.append(f'<circle cx="{ml+250}" cy="{ly-4}" r="4" fill="{C_RED}"/>')
    p.append(_text(ml + 260, ly - 1, "szczyt", 9, "start", C_TEXT))
    p.append(f'<polygon points="{ml+320},{ly-8} {ml+328},{ly-8} {ml+324},{ly-1}" fill="{C_GREEN}"/>')
    p.append(_text(ml + 334, ly - 1, "alert", 9, "start", C_TEXT))
    save("15_peak_timeline.svg", p)


# ----------------------------------------------------------------------------
# 16. PEAK - wplyw asymetrycznej kary (funkcji straty) na lapanie szczytu
# ----------------------------------------------------------------------------
def chart_peak_penalty():
    _, _, sweep, is_real = _compute_peak()
    pens = [str(x) for x in sweep["kara_FN"].tolist()]
    hits = sweep["trafione_%"].tolist()
    regrets = sweep["sredni_regret_%"].tolist()

    W, H = 720, 340
    ml, mr, mt, mb = 55, 55, 56, 45
    pw, ph = W - ml - mr, H - mt - mb
    n = len(pens)
    bw = pw / n * 0.45
    hmax = max(max(hits), 50)
    rmax = max(max(regrets), 4)

    def Xc(i): return ml + pw * (i + 0.5) / n
    def Yh(v): return mt + ph * (1 - v / hmax)
    def Yr(v): return mt + ph * (1 - v / rmax)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Asymetryczna kara: mocniej karzesz przegapienie -> lepiej lapiesz szczyt",
                   12, "middle", C_TEXT, "bold"))
    p.append(_text(W / 2, 38, "slupki = % trafionych dni (wiecej lepiej); linia = sredni regret (mniej lepiej)",
                   9, "middle", C_TEXT))
    for i, h in enumerate(hits):
        x = Xc(i) - bw / 2
        p.append(f'<rect x="{x:.1f}" y="{Yh(h):.1f}" width="{bw:.1f}" height="{Yh(0)-Yh(h):.1f}" fill="{C_GREEN}" rx="2"/>')
        p.append(_text(Xc(i), Yh(h) - 5, f"{h:.0f}%", 9, "middle", C_TEXT, "bold"))
        p.append(_text(Xc(i), Yh(0) + 14, f"kara {pens[i]}", 9, "middle", C_TEXT))
    # linia regret (skala prawa)
    rpts = " ".join(f"{Xc(i):.1f},{Yr(v):.1f}" for i, v in enumerate(regrets))
    p.append(f'<polyline fill="none" stroke="{C_RED}" stroke-width="2" points="{rpts}"/>')
    for i, v in enumerate(regrets):
        p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yr(v):.1f}" r="3" fill="{C_RED}"/>')
        p.append(_text(Xc(i), Yr(v) - 8, f"{v:.1f}%", 8, "middle", C_RED))
    p.append(_text(ml - 6, mt + 4, "trafione", 9, "end", C_GREEN))
    p.append(_text(ml + pw + 6, mt + 4, "regret", 9, "start", C_RED))
    p.append(_text(W / 2, H - 8, "kara za przegapienie szczytu (false negative)", 9, "middle", C_TEXT))
    save("16_peak_kara.svg", p)


# ----------------------------------------------------------------------------
# 17. ANATOMIA JEDNEJ SESJI (model peak): 33 swiece -> p per swieca -> 1 alert
# ----------------------------------------------------------------------------
def chart_peak_anatomy():
    """Najwazniejszy wykres: jak z 33 swiec model wybiera JEDNA = sygnal sprzedazy.
    Liczony na realnej sesji testowej (out-of-sample) modelem peak."""
    import numpy as np

    from src.features import build_features
    from src.peak import add_daily_high_target, train_peak_model

    intraday, daily_ctx, cfg, is_real, _ = _load_real_or_synthetic()
    feat_df, cols = build_features(intraday, daily_ctx, cfg)
    feat_df = add_daily_high_target(feat_df)
    days = sorted(feat_df["date"].unique())
    hold = days[-1]
    model = train_peak_model(feat_df[feat_df["date"] != hold], cols, cfg)
    g = feat_df[feat_df["date"] == hold].sort_index()
    proba = model.predict_proba(g[cols])[:, 1]
    close = g["Close"].values
    times = [t.strftime("%H:%M") for t in g["time"]]
    n = len(close)
    peak_i = int(close.argmax())
    alert_i = int(proba.argmax())
    thr = cfg.alert_probability_threshold
    regret = (close[peak_i] - close[alert_i]) / close[peak_i] * 100

    W, H = 860, 470
    ml, mr = 55, 20
    p1t, p1b = 56, 250
    p2t, p2b = 300, 430
    pw = W - ml - mr
    pmin, pmax = close.min(), close.max()
    pad = (pmax - pmin) * 0.18 or 1
    pmin -= pad; pmax += pad

    def X(i): return ml + pw * i / (n - 1)
    def Yp(v): return p1t + (p1b - p1t) * (1 - (v - pmin) / (pmax - pmin))
    def Yq(v): return p2t + (p2b - p2t) * (1 - v)

    p = _svg_open(W, H)
    p.append(_text(W / 2, 20, "Anatomia decyzji: 33 swiece -> prawdopodobienstwo -> JEDEN sygnal sprzedazy",
                   14, "middle", C_TEXT, "bold"))
    src = f"realna sesja {hold} (out-of-sample)" if is_real else "sesja syntetyczna"
    p.append(_text(W / 2, 38, f"{src}: model wskazal {times[alert_i]}, szczyt byl o {times[peak_i]} "
                   f"-> sprzedaz {regret:.2f}% ponizej maksimum", 10, "middle", C_GREEN))

    # panel 1: cena
    p.append(_text(ml, p1t - 6, "Cena (PLN)", 10, "start", C_TEXT, "bold"))
    pts = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(close))
    p.append(f'<polyline fill="none" stroke="{C_BLUE}" stroke-width="2" points="{pts}"/>')
    # faktyczny szczyt
    p.append(f'<circle cx="{X(peak_i):.1f}" cy="{Yp(close[peak_i]):.1f}" r="5" fill="{C_RED}"/>')
    p.append(_text(X(peak_i), Yp(close[peak_i]) - 10, f"faktyczny szczyt {times[peak_i]}", 9, "middle", C_RED, "bold"))
    # sygnal modelu (gwiazdka/pion)
    xa = X(alert_i)
    p.append(f'<line x1="{xa:.1f}" y1="{p1t}" x2="{xa:.1f}" y2="{p1b}" stroke="{C_GREEN}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    p.append(f'<circle cx="{xa:.1f}" cy="{Yp(close[alert_i]):.1f}" r="6" fill="none" stroke="{C_GREEN}" stroke-width="2.5"/>')
    p.append(_text(xa, p1t - 2, f"SPRZEDAJ {times[alert_i]}", 10, "middle", C_GREEN, "bold"))
    for i in range(0, n, 4):
        p.append(_text(X(i), p1b + 14, times[i], 8, "middle", C_TEXT))

    # panel 2: prawdopodobienstwo per swieca (slupki)
    p.append(_text(ml, p2t - 6, "p(dzienne maksimum) dla KAZDEJ swiecy - model bierze najwyzszy slupek",
                   10, "start", C_TEXT, "bold"))
    for gv in [0.0, 0.5, 1.0]:
        y = Yq(gv)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="{C_GRID}" stroke-width="1"/>')
        p.append(_text(ml - 6, y + 3, f"{gv:.0%}", 8, "end", C_TEXT))
    yth = Yq(thr)
    p.append(f'<line x1="{ml}" y1="{yth:.1f}" x2="{ml+pw}" y2="{yth:.1f}" stroke="{C_RED}" stroke-width="1.3" stroke-dasharray="5 3"/>')
    p.append(_text(ml + pw, yth - 4, f"prog {thr:.0%}", 9, "end", C_RED))
    bw = pw / n * 0.6
    for i, v in enumerate(proba):
        cx = X(i)
        col = C_GREEN if i == alert_i else C_GRAY
        p.append(f'<rect x="{cx-bw/2:.1f}" y="{Yq(v):.1f}" width="{bw:.1f}" height="{Yq(0)-Yq(v):.1f}" fill="{col}" rx="1"/>')
    p.append(_text(xa, Yq(proba[alert_i]) - 5, f"max = {proba[alert_i]:.0%}", 9, "middle", C_GREEN, "bold"))
    save("17_peak_anatomia.svg", p)


# ----------------------------------------------------------------------------
# 18. PIPELINE: jak dziala model peak, krok po kroku (mapa na kod)
# ----------------------------------------------------------------------------
def chart_pipeline():
    steps = [
        ("1. Dane 15-min", "33 swiece/sesje (OHLCV)", "src/data_sources.py", "fetch_intraday_yf"),
        ("2. Cechy per swieca", "VWAP, RSI, momentum, pora dnia, kontekst dzienny", "src/features.py", "build_features"),
        ("3. Etykieta (target)", "1 swieca/dzien = najwyzszy Close", "src/peak.py", "add_daily_high_target"),
        ("4. Model + strata", "regresja logistyczna, asymetryczna kara za przegapienie", "src/peak.py", "train_peak_model"),
        ("5. Decyzja", "p dla kazdej swiecy -> argmax dnia -> 1 alert (jesli p>=prog)", "src/peak.py", "_sell_metrics_for_day"),
        ("6. SPRZEDAJESZ", "jeden sygnal -> sprzedaz; ocena: regret + permutacja", "src/peak.py", "daily_peak_evaluation"),
    ]
    W = 860
    bh, gap, mt = 58, 26, 50
    H = mt + len(steps) * (bh + gap)
    bw = 560
    bx = (W - bw) / 2

    p = _svg_open(W, H)
    p.append(_text(W / 2, 24, "Jak dziala model dziennego maksimum - krok po kroku (mapa na kod)",
                   14, "middle", C_TEXT, "bold"))
    for i, (title, desc, f, fn) in enumerate(steps):
        y = mt + i * (bh + gap)
        last = i == len(steps) - 1
        fill = "#eaf6ec" if last else "#f0f6ff"
        stroke = C_GREEN if last else C_BLUE
        p.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        p.append(_text(bx + 14, y + 23, title, 13, "start", (C_GREEN if last else C_BLUE), "bold"))
        p.append(_text(bx + 14, y + 41, desc, 10, "start", C_TEXT))
        p.append(f'<text x="{bx+bw-12:.1f}" y="{y+bh-8:.1f}" font-size="9" text-anchor="end" '
                 f'fill="{C_AXIS}" font-family="monospace">{f} : {fn}()</text>')
        if not last:
            ay = y + bh + gap / 2
            p.append(f'<line x1="{W/2:.1f}" y1="{y+bh:.1f}" x2="{W/2:.1f}" y2="{ay+4:.1f}" stroke="{C_AXIS}" stroke-width="1.5"/>')
            p.append(f'<polygon points="{W/2-5:.1f},{ay:.1f} {W/2+5:.1f},{ay:.1f} {W/2:.1f},{ay+6:.1f}" fill="{C_AXIS}"/>')
    save("18_pipeline.svg", p)


if __name__ == "__main__":
    print("Generuje wykresy SVG do docs/assets/ ...")
    chart_price()
    chart_dow()
    chart_intraday()
    chart_balance()
    chart_models()
    chart_ranking()
    chart_intervals()
    chart_replay()
    chart_roc_distributions()
    chart_roc_curve()
    chart_daywise_auc()
    chart_permutation()
    chart_formula()
    chart_peak_regret()
    chart_peak_timeline()
    chart_peak_penalty()
    chart_peak_anatomy()
    chart_pipeline()
    print("Gotowe.")
