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

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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
def chart_replay():
    """Liczy replay na tej samej syntetycznej sesji co `replay --synthetic`
    (seed=42), zeby wizualizacja w raporcie zgadzala sie z wynikiem komendy."""
    import sys
    sys.path.insert(0, str(ROOT))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.config import load_config
    from src.features import build_features
    from src.replay import make_synthetic_session, replay_session

    cfg = load_config()
    intraday = make_synthetic_session(cfg, n_sessions=40, seed=42)
    feat_df, cols = build_features(intraday, None, cfg)
    sessions = sorted(feat_df["date"].unique())
    train = feat_df[feat_df["date"].isin(sessions[:-1])]
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(train[cols], train["target_local_top"])
    rep = replay_session(intraday, None, cfg, model, cols)

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
    p.append(_text(W / 2, 36, "sesja ILUSTRACYJNA (syntetyczna, seed=42) - pokazuje mechanizm, nie realne notowania",
                   9, "middle", C_RED))

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
    print("Gotowe.")
