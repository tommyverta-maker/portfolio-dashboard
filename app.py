"""University Investment Challenge Portfolio Optimizer - the whole dashboard in one Python file.

How it works
------------
1. This file is a small web server (Flask). When someone opens the site, it sends
   them the dashboard page (the HTML at the bottom of this file).
2. The page then asks this server for its numbers (/data/data.json and /data/prices.json).
   The first time, the server downloads 10 years of prices with yfinance and runs the
   same maths as Portfolio_Optimizer_M1.xlsx (build() below).
3. The "Pull latest data" button on the page calls /api/refresh: the server downloads
   fresh prices from Yahoo Finance, recalculates everything and the page redraws.
   To be polite to Yahoo, a new download happens at most once every 60 seconds.

Run it on your computer:   pip install -r requirements.txt   then   python app.py
then open http://localhost:5000 in your browser.
Put it online: see README.md (Render, free).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, Response

# ---------------------------------------------------------------- settings
CONFIG = {
    "owner": "Tommy",
    "title": "University Investment Challenge Portfolio Optimizer",
    "capital": 10000,
    "deposit_date": "2026-10-13",
    "purchase_date": "2026-11-13",
    "risk_free_rate": 0.0492,
    "trading_days": 252,
    "weeks_per_year": 52,
    "history_years": 10,
    "min_weight": 0.05,
    "max_weight": 0.2,
    "holdings": {
        "EOG": {
            "shares": 12.7055,
            "price": 157.26,
            "commission": 2.0,
            "paid": 2000.0
        },
        "BLK": {
            "shares": 1.2214,
            "price": 1226.88,
            "commission": 1.5,
            "paid": 1500.0
        },
        "SO": {
            "shares": 10.8035,
            "price": 92.47,
            "commission": 1.0,
            "paid": 1000.0
        },
        "COST": {
            "shares": 0.9127,
            "price": 1094.59,
            "commission": 1.0,
            "paid": 1000.0
        },
        "MCD": {
            "shares": 3.651,
            "price": 273.62,
            "commission": 1.0,
            "paid": 1000.0
        },
        "META": {
            "shares": 1.5928,
            "price": 627.21,
            "commission": 1.0,
            "paid": 1000.0
        },
        "ORCL": {
            "shares": 7.8353,
            "price": 127.5,
            "commission": 1.0,
            "paid": 1000.0
        },
        "PG": {
            "shares": 2.9561,
            "price": 168.97,
            "commission": 0.5,
            "paid": 500.0
        }
    },
    "dividends": [
        {
            "date": "2026-11-16",
            "ticker": "SO",
            "amount": 8.21
        },
        {
            "date": "2026-11-24",
            "ticker": "COST",
            "amount": 27.38
        }
    ],
    "names": {
        "EOG": [
            "EOG Resources",
            "Energy"
        ],
        "BLK": [
            "BlackRock",
            "Financials"
        ],
        "SO": [
            "Southern Company",
            "Utilities"
        ],
        "COST": [
            "Costco Wholesale",
            "Consumer Staples"
        ],
        "MCD": [
            "McDonald's",
            "Consumer Discretionary"
        ],
        "META": [
            "Meta Platforms",
            "Communication Services"
        ],
        "ORCL": [
            "Oracle",
            "Information Technology"
        ],
        "PG": [
            "Procter & Gamble",
            "Consumer Staples"
        ]
    }
}


# ---------------------------------------------------------------- calculations
def _r(x, n=6):
    """Round for JSON; NaN/inf -> None."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, n)


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build(close: pd.DataFrame, adj: pd.DataFrame, cfg: dict, source: str):
    """close/adj: DataFrames indexed by date, one column per ticker."""
    tickers = list(cfg["holdings"].keys())
    close = close[tickers].sort_index().dropna(how="all")
    adj = adj[tickers].sort_index().loc[close.index]
    T, W = cfg["trading_days"], cfg["weeks_per_year"]
    rf, capital = cfg["risk_free_rate"], cfg["capital"]
    lo, hi = cfg["min_weight"], cfg["max_weight"]
    H = cfg["holdings"]

    # ---- Daily statistics (Excel model) --------------------------------------
    logr = np.log(adj / adj.shift(1)).iloc[1:]             # LN(P_t / P_t-1)
    N = len(logr)
    mean_log = logr.mean()
    exp_daily = np.exp(mean_log) - 1                        # EXP(AVERAGE)-1
    exp_annual = np.exp(mean_log * T) - 1                   # EXP(AVERAGE*252)-1
    std_daily = logr.std(ddof=1)                            # SQRT(VAR.S)
    std_annual = np.sqrt(logr.var(ddof=1) * T)              # SQRT(VAR.S*252)
    variance = std_daily ** 2
    mod_sharpe = exp_annual / std_annual
    # Covariance: computed in Python with pandas, population (ddof=0) = Excel VARP/COVARIANCE.P
    cov_py = logr.cov(ddof=0)
    # VARCOVAR: the Excel formula MMULT(TRANSPOSE(X),X)/(N-1), X = log return - expected daily return
    X = logr - exp_daily
    varcovar = (X.T @ X) / (N - 1)
    corr = varcovar / np.outer(std_daily, std_daily)        # VARCOVAR / (sd_i * sd_j)

    # Holdings tab: Invested ($) = amount paid - commission; Weight = Invested / capital
    invested = np.array([H[t]["paid"] - H[t]["commission"] for t in tickers])
    act_w = invested / capital
    mu = exp_annual.values

    # Portfolio figures exactly as the spreadsheet computes them:
    # N3 = SUMPRODUCT(annual returns, actual weights)
    # N4 = SQRT(MMULT(MMULT(max-weight row, covariance matrix), TRANSPOSE(max-weight row)))
    # N5 = same x 252 inside the root;  N8 = (N3 - rf) / N5
    w_sheet = np.full(len(tickers), hi)
    p_ret = float(act_w @ mu)
    p_std_d = float(math.sqrt(w_sheet @ cov_py.values @ w_sheet))
    p_std_a = float(math.sqrt(w_sheet @ cov_py.values @ w_sheet * T))
    p_sharpe = (p_ret - rf) / p_std_a

    # ---- Weekly statistics -----------------------------------------------------
    wk = adj.resample("W-FRI").last().dropna(how="all")     # last trading day of each week
    wr = wk.pct_change().iloc[1:]
    wk_stats = {}
    for t in tickers:
        m, v = wr[t].mean(), wr[t].var(ddof=1)
        wk_stats[t] = dict(avg_weekly=_r(m), weekly_var=_r(v, 8), weekly_std=_r(math.sqrt(v)),
                           ann_return=_r(m * W), ann_std=_r(math.sqrt(v * W)), weeks=int(wr[t].count()))

    # ---- Calendar-year statistics ---------------------------------------------
    last = close.index[-1]
    yr = last.year
    cy = close[close.index.year == yr]
    wry = wr[wr.index.year == yr]
    year_stats = {}
    for t in tickers:
        s = cy[t]
        hi_v, lo_v, lst = s.max(), s.min(), s.iloc[-1]
        year_stats[t] = dict(
            high=_r(hi_v, 2), high_date=_d(s.idxmax()), low=_r(lo_v, 2), low_date=_d(s.idxmin()),
            latest=_r(lst, 2), pct_below_high=_r(lst / hi_v - 1), pct_above_low=_r(lst / lo_v - 1),
            best_weeks=[[_d(d), _r(v)] for d, v in wry[t].nlargest(5).items()],
            worst_weeks=[[_d(d), _r(v)] for d, v in wry[t].nsmallest(5).items()])

    # ---- Positions from the actual trades --------------------------------------
    pdate = pd.Timestamp(cfg["purchase_date"])
    all_divs = cfg.get("dividends", [])
    divs = [x for x in all_divs if pd.Timestamp(x["date"]) <= last]   # received by the price date
    div_by = {t: round(sum(x["amount"] for x in divs if x["ticker"] == t), 2) for t in tickers}
    cash = capital - sum(H[t]["paid"] for t in tickers) + sum(x["amount"] for x in divs)
    simple = adj.pct_change()

    def ret_since(t, days=None, ytd=False):
        s = adj[t]
        if ytd:
            base = s[s.index < pd.Timestamp(yr, 1, 1)]
            b = base.iloc[-1] if len(base) else s.iloc[0]
        else:
            b = s[s.index <= last - pd.Timedelta(days=days)].iloc[-1]
        return s.iloc[-1] / b - 1

    rows, tot_val, tot_prev = [], 0.0, 0.0
    for i, t in enumerate(tickers):
        h = H[t]
        px, prev = float(close[t].iloc[-1]), float(close[t].iloc[-2])
        val = h["shares"] * px
        # a position bought on the latest trading day moves from its fill price, not yesterday's close
        pval = h["shares"] * (prev if pdate < last else h["price"])
        tot_val += val
        tot_prev += pval
        rows.append(dict(
            ticker=t, name=cfg["names"][t][0], sector=cfg["names"][t][1],
            shares=h["shares"], fill=h["price"], commission=h["commission"], paid=h["paid"],
            invested=_r(invested[i], 2), weight=_r(act_w[i]),
            price=_r(px, 2), prev_close=_r(prev, 2), day_change=_r(px - prev, 2), day_pct=_r(px / prev - 1),
            value=_r(val, 2), pl=_r(val - invested[i], 2), pl_pct=_r(val / invested[i] - 1),
            dividends=div_by[t],
            ret_1m=_r(ret_since(t, 30)), ret_ytd=_r(ret_since(t, ytd=True)), ret_1y=_r(ret_since(t, 365)),
            ret_5y=_r(ret_since(t, 365 * 5)),
            vol_30d=_r(simple[t].iloc[-30:].std(ddof=1) * math.sqrt(T)),
            vol_1y=_r(simple[t].iloc[-252:].std(ddof=1) * math.sqrt(T))))

    p_val, p_prev = tot_val + cash, tot_prev + cash
    port_daily = (simple.fillna(0) * act_w).sum(axis=1)
    pvol30 = port_daily.iloc[-30:].std(ddof=1) * math.sqrt(T)

    ledger = [dict(date=cfg["deposit_date"], kind="Deposit", ticker="", detail="Starting capital", amount=capital)]
    ledger += [dict(date=cfg["purchase_date"], kind="Buy", ticker=t,
                    detail=f'{H[t]["shares"]} × ${H[t]["price"]:,.2f} + ${H[t]["commission"]:.2f} commission',
                    amount=-H[t]["paid"]) for t in tickers]
    ledger += [dict(date=x["date"], kind="Dividend", ticker=x["ticker"], detail="Cash dividend",
                    amount=x["amount"]) for x in all_divs]
    ledger.sort(key=lambda x: x["date"], reverse=True)

    calc = {t: dict(exp_daily=_r(exp_daily[t], 8), exp_annual=_r(exp_annual[t]),
                    std_daily=_r(std_daily[t]), std_annual=_r(std_annual[t]),
                    min_weight=lo, max_weight=hi, act_weight=_r(act_w[i]),
                    variance=_r(variance[t], 8), mod_sharpe=_r(mod_sharpe[t]), n=N)
            for i, t in enumerate(tickers)}

    def mat(df, n=8):
        return [[_r(df.loc[a, b], n) for b in tickers] for a in tickers]

    data = dict(
        meta=dict(owner=cfg["owner"], title=cfg["title"], source=source,
                  generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                  as_of=_d(last), data_start=_d(close.index[0]), purchase_date=cfg["purchase_date"],
                  deposit_date=cfg["deposit_date"], capital=capital, rf=rf, trading_days=T, n_returns=N,
                  year=yr, min_weight=lo, max_weight=hi),
        tickers=tickers,
        holdings=rows,
        cash=_r(cash, 2),
        # Holdings tab CASH = capital - money in stocks; its P/L = commissions paid and dividends received
        cash_invested=_r(capital - invested.sum(), 2), cash_weight=_r(1 - act_w.sum()),
        cash_pl=_r(cash - (capital - invested.sum()), 2),
        ledger=ledger,
        portfolio=dict(
            value=_r(p_val, 2), prev_value=_r(p_prev, 2), day_change=_r(p_val - p_prev, 2),
            day_pct=_r(p_val / p_prev - 1), stocks_value=_r(tot_val, 2),
            paid=_r(sum(H[t]["paid"] for t in tickers), 2), invested=_r(invested.sum(), 2),
            commissions=_r(sum(H[t]["commission"] for t in tickers), 2),
            dividends=_r(sum(x["amount"] for x in divs), 2),
            pl=_r(p_val - capital, 2), pl_pct=_r(p_val / capital - 1), stock_weight=_r(act_w.sum()),
            exp_return=_r(p_ret), std_daily=_r(p_std_d), std_annual=_r(p_std_a), sharpe=_r(p_sharpe),
            vol_30d=_r(pvol30)),
        calc=calc,
        cov=mat(cov_py, 10), varcovar=mat(varcovar, 10), corr=mat(corr, 6),
        weekly=dict(stats=wk_stats, weeks=int(len(wr))),
        year_stats=year_stats,
    )

    roll = simple.rolling(30).std(ddof=1) * math.sqrt(T)
    prices = dict(
        dates=[_d(d) for d in close.index],
        close={t: [_r(v, 2) for v in close[t].values] for t in tickers},
        adj={t: [_r(v, 3) for v in adj[t].values] for t in tickers},
        vol30={t: [_r(v, 4) for v in roll[t].values] for t in tickers},
    )
    return data, prices


# ---------------------------------------------------------------- data download
def download_prices(cfg: dict):
    """10 years of daily Close and Adj Close for every holding, from Yahoo Finance."""
    tickers = list(cfg["holdings"])
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=cfg["history_years"])
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    close = raw["Close"].dropna(how="any")
    adj = raw["Adj Close"].loc[close.index]
    if len(close) < 300:
        raise RuntimeError(f"only {len(close)} days of prices came back")
    return close, adj


# ---------------------------------------------------------------- web server
app = Flask(__name__)
COOLDOWN_SECONDS = 60
_lock = threading.Lock()
_state = {"data": None, "prices": None, "pulled": 0.0}


def refresh(force: bool = False):
    """Download prices and rebuild every figure. Returns (did_download, seconds_since_last)."""
    with _lock:
        age = time.time() - _state["pulled"]
        if _state["data"] is not None and not force and age < COOLDOWN_SECONDS:
            return False, age
        close, adj = download_prices(CONFIG)
        data, prices = build(close, adj, CONFIG, source="Yahoo Finance via yfinance")
        _state.update(data=data, prices=prices, pulled=time.time())
        return True, 0.0


def _json(obj, status=200):
    return Response(json.dumps(obj, separators=(",", ":")), status=status, mimetype="application/json")


def _ensure_data():
    if _state["data"] is None:
        refresh()


@app.get("/")
def page():
    return Response(HTML, mimetype="text/html")


@app.get("/data/data.json")
def data_json():
    try:
        _ensure_data()
    except Exception as e:  # download failed: tell the page
        return _json({"error": str(e)}, 502)
    return _json(_state["data"])


@app.get("/data/prices.json")
def prices_json():
    try:
        _ensure_data()
    except Exception as e:
        return _json({"error": str(e)}, 502)
    return _json(_state["prices"])


@app.get("/api/status")
def status():
    return _json({"live": True})


@app.post("/api/refresh")
def api_refresh():
    try:
        downloaded, age = refresh()
    except Exception as e:
        return _json({"error": f"Yahoo Finance download failed ({e})"}, 502)
    as_of = datetime.strptime(_state["data"]["meta"]["as_of"], "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")
    if downloaded:
        msg = f"Updated from Yahoo Finance: prices as of the {as_of} close."
    else:
        msg = f"Already up to date: prices were pulled {int(age)} seconds ago (as of the {as_of} close)."
    return _json({"ok": True, "downloaded": downloaded, "message": msg})



# ---------------------------------------------------------------- the dashboard page
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>University Investment Challenge Portfolio Optimizer</title>
<meta name="description" content="Long-only portfolio of eight US equities: live holdings, P/L, risk and per-stock analytics.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* VOICE: a fund factsheet - calm, exact, numbers first.
   IDEA: the printed portfolio statement: ruled tables, one deep teal, figures that line up.
   PICTURE: holdings, allocation and each stock's price line carry the screens.
   COLOUR and TYPE: white, ink #14161a, deep teal #0b5d63, red for losses; Inter throughout.
   LAYOUT: finding as title, figures strip, holdings, then allocation and risk tables. */
:root{color-scheme:light;--bg:#fff;--ink:#14161a;--mute:#5b6168;--rule:#dfe2e5;--soft:#f3f6f6;--teal:#0b5d63;--neg:#b42318;--pos:#0b5d63}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--bg:#0d1416;--ink:#e8eced;--mute:#97a3a6;--rule:#25333a;--soft:#13201f;--teal:#5fb8bd;--neg:#f0796c;--pos:#5fb8bd}}
:root[data-theme=dark]{color-scheme:dark;--bg:#0d1416;--ink:#e8eced;--mute:#97a3a6;--rule:#25333a;--soft:#13201f;--teal:#5fb8bd;--neg:#f0796c;--pos:#5fb8bd}
*{box-sizing:border-box;min-width:0}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow-x:clip;font-variant-numeric:tabular-nums}
a{color:var(--teal)}
.wrap{max-width:1240px;margin:0 auto;padding:0 max(4vw,16px)}
.band{padding:clamp(24px,3.5vw,48px) 0;border-top:1px solid var(--rule)}
header{padding-top:22px}
.mast{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:6px 24px;border-bottom:3px solid var(--teal);padding-bottom:10px}
.mast b{font-weight:700;font-size:18px;letter-spacing:-.01em}
.mast span{color:var(--mute);font-size:13px}
.upd{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px}
.upd button{min-height:36px}
.upd button[disabled]{opacity:.6;cursor:progress}
#pullmsg{flex-basis:100%;margin:0}#pullmsg:empty{display:none}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:12px 0}
nav a{min-height:40px;display:inline-flex;align-items:center;padding:0 14px;border:1px solid var(--rule);border-radius:20px;text-decoration:none;color:var(--ink);font-weight:500;font-size:14px}
nav a.on{background:var(--teal);border-color:var(--teal);color:var(--bg)}
nav a:focus-visible,button:focus-visible,tr:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
h1{font-weight:600;font-size:clamp(28px,3.6vw,40px);line-height:1.12;letter-spacing:-.02em;margin:18px 0 10px;text-wrap:balance}
h2{font-weight:600;font-size:22px;line-height:1.2;letter-spacing:-.01em;margin:0 0 6px}
h3{font-weight:600;font-size:14px;margin:14px 0 6px}
.lede{color:var(--mute);max-width:72ch;margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));border-top:1px solid var(--ink);border-bottom:1px solid var(--rule);margin-top:22px}
.kpi{padding:14px 16px 14px 0}
.kpi small{display:block;color:var(--mute);font-size:13px}
.kpi strong{display:block;font-weight:600;font-size:clamp(22px,2.4vw,28px);line-height:1.2;letter-spacing:-.01em;white-space:nowrap}
.kpi em{font-style:normal;font-size:13px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.split{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:40px}
@media(max-width:860px){.split{grid-template-columns:minmax(0,1fr)}}
.fig{position:relative;min-height:340px}.fig.sm{min-height:200px}
.fig>canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:0 0 10px}
.bar .grow{flex:1}
button{font:500 14px Inter,system-ui,sans-serif;min-height:36px;padding:0 12px;border:1px solid var(--rule);background:var(--bg);color:var(--ink);border-radius:6px;cursor:pointer}
button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.cap{color:var(--mute);font-size:13px;margin:8px 0 0}
.ruled{border-top:1px solid var(--ink)}
.row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid var(--rule);font-size:15px}
.row span{color:var(--mute)}.row b{font-weight:500;white-space:nowrap}
.alloc{display:grid;grid-template-columns:54px minmax(0,1fr) 150px;gap:8px 12px;align-items:center;font-size:14px;margin-top:12px}
.alloc .t{font-weight:600}.alloc .v{text-align:right;white-space:nowrap}.alloc .v small{color:var(--mute)}
.track{height:22px;position:relative;background:var(--soft)}
.track i{position:absolute;left:0;top:0;height:14px;background:var(--teal)}
.track u{position:absolute;left:0;bottom:0;height:5px;background:var(--mute);opacity:.6}
.key{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin-top:4px}
.key i{display:inline-block;width:14px;height:10px;background:var(--teal);margin-right:6px;vertical-align:-1px}
.key u{display:inline-block;width:14px;height:4px;background:var(--mute);opacity:.6;margin-right:6px;vertical-align:2px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 8px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
th{font-weight:600;color:var(--mute);font-size:13px;border-bottom:1px solid var(--ink)}
th:first-child,td:first-child,th.l,td.l{text-align:left}
tbody tr.link{cursor:pointer}tbody tr.link:hover{background:var(--soft)}
td .nm{display:block;color:var(--mute);font-size:12px}
tr.tot td{border-top:1px solid var(--ink);font-weight:600}
.heat td{text-align:center;min-width:64px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:28px 40px}
.expl{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px 28px;margin:0 0 18px}
.expl div{background:var(--soft);padding:14px 16px;font-size:14px}
.expl b{display:block;font-size:15px;margin-bottom:4px}
.expl code{font-size:13px}
.hero{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 18px}
.hero strong{font-weight:600;font-size:clamp(34px,4.6vw,48px);line-height:1;letter-spacing:-.02em}
.pager{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.pager a{min-height:44px;display:inline-flex;align-items:center}
footer{color:var(--mute);font-size:13px;padding:28px 0 48px;border-top:1px solid var(--rule)}
#err{display:none;padding:40px 0}
@media(max-width:600px){.bar .grow{flex-basis:100%}table{font-size:13px}th,td{padding:7px 6px}.alloc{grid-template-columns:44px minmax(0,1fr) 130px}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<header class="wrap">
  <div class="mast"><b>University Investment Challenge Portfolio Optimizer</b><div class="upd"><span id="stamp">Loading data…</span><button id="pull" type="button" hidden>Pull latest data</button><p id="pullmsg" class="cap" role="status" aria-live="polite"></p></div></div>
  <nav id="nav" aria-label="Sections"></nav>
</header>
<main class="wrap" id="app"></main>
<div class="wrap" id="err"><h2>Data could not be loaded</h2><p class="lede">This page reads <code>data/data.json</code> and <code>data/prices.json</code> next to it. Open it through a web server (GitHub Pages, or <code>python -m http.server</code>) rather than as a local file.</p></div>
<footer class="wrap" id="foot"></footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
{
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=(x,d=2)=>x==null?'–':(x<0?'−$':'$')+Math.abs(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const smoney=x=>x==null?'–':(x>0?'+':'')+money(x);
const pct=(x,d=2,sign=false)=>x==null?'–':(sign&&x>0?'+':x<0?'−':'')+Math.abs(x*100).toFixed(d)+'%';
const num=(x,d=4)=>x==null?'–':(+x).toFixed(d);
const sci=x=>x==null?'–':(+x).toExponential(3);
const cls=x=>x>0?'pos':x<0?'neg':'';
const wpct=x=>x==null?'–':+(x*100).toFixed(3)+'%';
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const fdate=s=>{const [y,m,d]=String(s).split('-');return `${+d} ${MON[+m-1]} ${y}`};
let D,P,charts=[];
const dark=()=>{const t=document.documentElement.dataset.theme;return t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches)};
const pal=()=>dark()?{bg:'#0d1416',ink:'#e8eced',mute:'#97a3a6',rule:'#25333a',teal:'#5fb8bd',neg:'#f0796c',fill:'rgba(95,184,189,.12)',nfill:'rgba(240,121,108,.10)'}:{bg:'#ffffff',ink:'#14161a',mute:'#5b6168',rule:'#dfe2e5',teal:'#0b5d63',neg:'#b42318',fill:'rgba(11,93,99,.08)',nfill:'rgba(180,35,24,.07)'};
const kill=()=>{charts.forEach(c=>c.destroy());charts=[]};

function baseOpts(c,{xTicks=7,yFmt=v=>v}={}){return{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},
  plugins:{legend:{display:false},tooltip:{backgroundColor:c.ink,titleColor:c.bg,bodyColor:c.bg,displayColors:false}},
  scales:{x:{ticks:{color:c.mute,maxTicksLimit:innerWidth<520?4:xTicks,maxRotation:0,autoSkip:true,callback(v){const l=this.getLabelForValue(v);return innerWidth<520&&typeof l==='string'?l.slice(0,7):l}},grid:{display:false},border:{color:c.rule}},
          y:{position:'right',ticks:{color:c.mute,callback:yFmt,maxTicksLimit:6},grid:{color:c.rule},border:{display:false}}}}}

function nav(active){
  $('#nav').innerHTML=`<a href="#overview" class="${active==='overview'?'on':''}">Overview</a>`+
    D.tickers.map(t=>`<a href="#${t.toLowerCase()}" class="${active===t?'on':''}">${esc(t)}</a>`).join('');
}

const RISK_ROWS=[
 ['exp_daily','Expected daily return',v=>pct(v,4)],
 ['exp_annual','Expected annual return',v=>pct(v,2)],
 ['std_daily','Std dev, daily',v=>pct(v,3)],
 ['std_annual','Std dev, annual',v=>pct(v,2)],
 ['min_weight','Min weight',v=>pct(v,0)],
 ['max_weight','Max weight',v=>pct(v,0)],
 ['act_weight','Weight (invested ÷ capital)',wpct],
 ['variance','Variance, daily',sci],
 ['mod_sharpe','Modified Sharpe (return ÷ σ)',v=>num(v,3)],
 ['n','Observations (N)',v=>v.toLocaleString()]];

/* ------------------------------------------------------------ OVERVIEW */
function overview(){
  const p=D.portfolio,m=D.meta,H=D.holdings;
  const best=[...H].sort((a,b)=>b.day_pct-a.day_pct)[0],worst=[...H].sort((a,b)=>a.day_pct-b.day_pct)[0];
  const scale=Math.max(...H.map(h=>h.weight),D.cash_weight)*1.05;
  $('#app').innerHTML=`
  <h1>University Investment Challenge Portfolio Optimizer</h1>
  <p class="lede">Eight US large caps chosen with a long-only mean-variance optimiser on ${m.n_returns.toLocaleString()} daily log returns (${fdate(m.data_start)} to ${fdate(m.as_of)}). ${money(m.capital,0)} deposited on ${fdate(m.deposit_date)}, invested on ${fdate(m.purchase_date)}. Prices refresh every trading day from Yahoo Finance. Select any holding for its full figures and price history.</p>
  <div class="kpis">
    <div class="kpi"><small>Portfolio value</small><strong>${money(p.value)}</strong><em class="${cls(p.day_change)}">${smoney(p.day_change)} (${pct(p.day_pct,2,true)}) on the day</em></div>
    <div class="kpi"><small>Total P/L since ${fdate(m.purchase_date)}</small><strong class="${cls(p.pl)}">${smoney(p.pl)}</strong><em class="${cls(p.pl)}">${pct(p.pl_pct,2,true)} on ${money(m.capital,0)}</em><em style="display:block;color:var(--mute)">incl. ${money(p.dividends)} dividends, ${money(p.commissions)} commissions</em></div>
    <div class="kpi"><small>Expected annual return</small><strong>${pct(p.exp_return)}</strong><em>weights × geometric means</em></div>
    <div class="kpi"><small>Annual volatility</small><strong>${pct(p.std_annual)}</strong><em>30-day realised ${pct(p.vol_30d,1)}</em></div>
    <div class="kpi"><small>Sharpe ratio</small><strong>${num(p.sharpe,4)}</strong><em>risk-free ${pct(m.rf)}</em></div>
  </div>

  <section style="margin-top:34px">
    <h2>Holdings</h2><p class="lede" style="margin-bottom:12px">Bought ${fdate(m.purchase_date)}. Prices at the ${fdate(m.as_of)} close. Invested is the amount put into each stock, excluding commission; commissions and dividends go through cash. Select a row for the stock page.</p>
    <div class="scroll"><table id="hold"><thead><tr><th>Holding</th><th>Price</th><th>Shares</th><th>Invested</th><th>Value</th><th>P/L</th><th>Dividends</th><th>Weight</th><th>1M</th><th>YTD</th><th>1Y</th><th>30D vol</th><th>1Y vol</th></tr></thead><tbody>
    ${H.map(h=>`<tr class="link" data-t="${esc(h.ticker)}" tabindex="0"><td><b>${esc(h.ticker)}</b><span class="nm">${esc(h.name)}</span></td><td>${money(h.price)}<span class="nm ${cls(h.day_pct)}">${pct(h.day_pct,2,true)} 1D</span></td><td>${num(h.shares,4)}<span class="nm">@ ${money(h.fill)}</span></td><td>${money(h.invested)}</td><td>${money(h.value)}</td><td class="${cls(h.pl)}">${smoney(h.pl)}<span class="nm ${cls(h.pl)}">${pct(h.pl_pct,2,true)}</span></td><td>${h.dividends?money(h.dividends):'–'}</td><td>${wpct(h.weight)}</td><td class="${cls(h.ret_1m)}">${pct(h.ret_1m,1,true)}</td><td class="${cls(h.ret_ytd)}">${pct(h.ret_ytd,1,true)}</td><td class="${cls(h.ret_1y)}">${pct(h.ret_1y,1,true)}</td><td>${pct(h.vol_30d,1)}</td><td>${pct(h.vol_1y,1)}</td></tr>`).join('')}
    <tr><td><b>Cash</b><span class="nm">earns 0%</span></td><td></td><td></td><td>${money(D.cash_invested)}</td><td>${money(D.cash)}</td><td class="${cls(D.cash_pl)}">${smoney(D.cash_pl)}<span class="nm">−${money(p.commissions)} fees, +${money(p.dividends)} div.</span></td><td></td><td>${wpct(D.cash_weight)}</td><td colspan="5"></td></tr>
    <tr class="tot"><td>Total</td><td class="${cls(p.day_pct)}">${pct(p.day_pct,2,true)}<span class="nm">1D</span></td><td></td><td>${money(m.capital)}</td><td>${money(p.value)}</td><td class="${cls(p.pl)}">${smoney(p.pl)}<span class="nm ${cls(p.pl)}">${pct(p.pl_pct,2,true)}</span></td><td>${money(p.dividends)}</td><td>100%</td><td colspan="3"></td><td>${pct(p.vol_30d,1)}</td><td></td></tr>
    </tbody></table></div>
    <p class="cap">Total P/L = value of the shares + cash − ${money(m.capital,0)} deposited. Weight = invested ÷ ${money(m.capital,0)} capital; cash is the capital not invested in stocks.</p>
  </section>

  <section class="band" style="margin-top:34px"><div class="split">
    <div><h2>Allocation</h2>
      <p class="cap" style="margin:0">Invested ÷ ${money(m.capital,0)} capital.</p>
      <div class="alloc">${H.map(h=>`<span class="t">${esc(h.ticker)}</span><div class="track"><i style="width:${h.weight/scale*100}%"></i></div><span class="v">${wpct(h.weight)} <small>${money(h.invested)}</small></span>`).join('')}
      <span class="t">Cash</span><div class="track"><i style="width:${D.cash_weight/scale*100}%;background:var(--mute)"></i></div><span class="v">${wpct(D.cash_weight)} <small>${money(D.cash_invested)}</small></span></div>
    </div>
    <aside class="ruled">
      <h3>Portfolio</h3>
      <div class="row"><span>Expected return</span><b>${pct(p.exp_return)}</b></div>
      <div class="row"><span>Std dev, daily</span><b>${pct(p.std_daily,3)}</b></div>
      <div class="row"><span>Std dev, annual</span><b>${pct(p.std_annual)}</b></div>
      <div class="row"><span>Risk-free rate</span><b>${pct(m.rf)}</b></div>
      <div class="row"><span>Total weight in stocks</span><b>${pct(p.stock_weight)}</b></div>
      <div class="row"><span>Sharpe ratio</span><b>${num(p.sharpe,4)}</b></div>
      <h3>Latest session</h3>
      <div class="row"><span>Best · ${esc(best.ticker)}</span><b class="${cls(best.day_pct)}">${pct(best.day_pct,2,true)}</b></div>
      <div class="row"><span>Worst · ${esc(worst.ticker)}</span><b class="${cls(worst.day_pct)}">${pct(worst.day_pct,2,true)}</b></div>
    </aside>
  </div></section>

  <section class="band">
    <h2>Account activity</h2><p class="lede" style="margin-bottom:12px">Every cash movement since the account opened.</p>
    <div class="scroll"><table><thead><tr><th>Date</th><th class="l">Movement</th><th class="l">Detail</th><th>Amount</th></tr></thead><tbody>
    ${D.ledger.map(l=>`<tr><td>${fdate(l.date)}</td><td class="l">${esc(l.kind)}${l.ticker?' · <a href="#'+esc(l.ticker.toLowerCase())+'">'+esc(l.ticker)+'</a>':''}</td><td class="l">${esc(l.detail)}</td><td class="${l.amount>0&&l.kind==='Dividend'?'pos':''}">${smoney(l.amount)}</td></tr>`).join('')}
    <tr class="tot"><td>Cash balance</td><td></td><td class="l" style="font-weight:400;color:var(--mute);white-space:normal">at the ${fdate(m.as_of)} close${D.ledger.some(l=>l.date>m.as_of)?'; movements after that date are added once prices for that day load':''}</td><td>${money(D.cash)}</td></tr>
    </tbody></table></div>
  </section>

  <section class="band">
    <h2>Correlation</h2><p class="lede" style="margin-bottom:12px">Correlation = VARCOVAR ÷ (σᵢ × σⱼ), from daily log returns.</p>
    <div class="scroll"><table class="heat" id="corrT"></table></div>
  </section>

  <section class="band">
    <div class="bar"><h2 class="grow">Covariance</h2><span id="mx"></span></div>
    <div class="expl">
      <div><b>VARCOVAR: calculated in Excel</b>X is the matrix of daily log returns minus each stock's expected daily return (e<sup>mean</sup> − 1). VARCOVAR = <code>MMULT(TRANSPOSE(X), X) / (N − 1)</code>, N = ${m.n_returns.toLocaleString()}. This is the matrix behind the correlations above. The daily refresh re-runs the same formula so it stays current.</div>
      <div><b>Covariance: calculated in Python</b>pandas <code>DataFrame.cov(ddof=0)</code> on the same daily log returns: each return minus its average, products summed and divided by N (population covariance, the same as Excel's VARP). It agrees with VARCOVAR to about 0.04%; the gap comes from dividing by N instead of N − 1. This matrix is used for the portfolio's standard deviation.</div>
    </div>
    <div class="scroll"><table class="heat" id="covT"></table></div>
    <p class="cap" id="covcap"></p>
  </section>

  <section class="band">
    <h2>Return and risk by stock</h2><p class="lede" style="margin-bottom:12px">From ${m.n_returns.toLocaleString()} daily log returns, ln(P<sub>t</sub> ÷ P<sub>t−1</sub>), of the dividend-adjusted close.</p>
    <div class="scroll"><table><thead><tr><th>Measure</th>${D.tickers.map(t=>`<th><a href="#${t.toLowerCase()}">${esc(t)}</a></th>`).join('')}</tr></thead><tbody>
    ${RISK_ROWS.map(([k,l,f])=>`<tr><td>${l}</td>${D.tickers.map(t=>`<td>${f(D.calc[t][k])}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
  </section>
  ${methodHTML()}`;

  $('#hold').addEventListener('click',e=>{const r=e.target.closest('tr[data-t]');if(r)location.hash=r.dataset.t.toLowerCase()});
  $('#hold').addEventListener('keydown',e=>{const r=e.target.closest('tr[data-t]');if(r&&e.key==='Enter')location.hash=r.dataset.t.toLowerCase()});
  heat($('#corrT'),D.corr,true);covariance('varcov');
}

function heat(el,src,isCorr){
  const T=D.tickers,max=isCorr?1:Math.max(...src.flat().map(Math.abs));
  el.innerHTML=`<thead><tr><th></th>${T.map(t=>`<th>${esc(t)}</th>`).join('')}</tr></thead><tbody>${src.map((r,i)=>`<tr><th>${esc(T[i])}</th>${r.map(v=>{const a=Math.abs(v)/max;return `<td style="background:rgba(${dark()?'95,184,189':'11,93,99'},${(a*0.85).toFixed(2)});color:${a>0.55&&!dark()?'#fff':'inherit'}">${isCorr?num(v,2):sci(v)}</td>`}).join('')}</tr>`).join('')}</tbody>`;
}
function covariance(kind){
  heat($('#covT'),kind==='cov'?D.cov:D.varcovar,false);
  $('#covcap').textContent={varcov:'VARCOVAR (Excel): daily log returns, sample, divided by N − 1.',cov:'Covariance (Python): daily log returns, population, divided by N.'}[kind];
  const map={varcov:'VARCOVAR (Excel)',cov:'Covariance (Python)'};
  const el=$('#mx');el.innerHTML=Object.entries(map).map(([k,l])=>`<button data-k="${k}" class="${k===kind?'on':''}">${l}</button>`).join(' ');
  el.onclick=e=>{const b=e.target.closest('button');if(b)covariance(b.dataset.k)};
}

function methodHTML(){const m=D.meta;return `<section class="band"><h2>Method</h2><div class="grid2"><div>
  <p>Prices come from Yahoo Finance through the <code>yfinance</code> Python library: ${fdate(m.data_start)} to ${fdate(m.as_of)}, ${m.n_returns.toLocaleString()} daily returns. Returns are daily log returns of the dividend-adjusted close. Expected daily return is e<sup>mean</sup> − 1; expected annual return is e<sup>mean × ${m.trading_days}</sup> − 1; annual σ is √(sample variance × ${m.trading_days}).</p>
  <p>Portfolio σ is √(wᵀΣw × ${m.trading_days}), with Σ the covariance matrix and w the max-weight row (${pct(m.max_weight,0)} in each stock), as in the Excel model. Expected return uses the actual weights, with cash at 0%. Sharpe = (expected return − ${pct(m.rf)}) ÷ annual σ.</p></div><div>
  <p>P/L uses the actual fills of ${fdate(m.purchase_date)} including commissions, plus dividends received in cash. Past returns do not predict future returns; this is a student project, not investment advice.</p></div></div></section>`}

/* ------------------------------------------------------------ STOCK PAGE */
function rangeButtons(el,opts,cur,cb){
  el.innerHTML=opts.map(o=>`<button data-k="${o}" class="${o===cur?'on':''}">${o}</button>`).join(' ');
  el.onclick=e=>{const b=e.target.closest('button');if(!b)return;el.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));cb(b.dataset.k)};
}
function startIndex(dates,k){
  if(k==='10Y')return 0;
  const last=new Date(dates[dates.length-1]);let d=new Date(last);
  if(k==='YTD')d=new Date(Date.UTC(last.getUTCFullYear(),0,1));
  else{const n=parseInt(k);k.endsWith('M')?d.setUTCMonth(d.getUTCMonth()-n):d.setUTCFullYear(d.getUTCFullYear()-n)}
  const iso=d.toISOString().slice(0,10);const i=dates.findIndex(x=>x>=iso);return i<0?0:i;
}

function stock(t){
  const i=D.tickers.indexOf(t),h=D.holdings[i],k=D.calc[t],w=D.weekly.stats[t],y=D.year_stats[t],m=D.meta;
  const prev=D.tickers[(i+D.tickers.length-1)%D.tickers.length],next=D.tickers[(i+1)%D.tickers.length];
  const others=D.tickers.map((u,j)=>({u,corr:D.corr[i][j],vc:D.varcovar[i][j],cov:D.cov[i][j]})).filter(o=>o.u!==t);
  $('#app').innerHTML=`
  <h1>${esc(h.name)} <span style="color:var(--mute);font-weight:400">(${esc(t)})</span></h1>
  <p class="lede">${esc(h.sector)} · ${wpct(h.weight)} of the portfolio (${money(h.invested)} invested)</p>
  <div class="hero" style="margin-top:16px"><strong>${money(h.price)}</strong><span class="${cls(h.day_pct)}">${smoney(h.day_change)} (${pct(h.day_pct,2,true)})</span><span class="cap" style="margin:0">close ${fdate(m.as_of)}</span></div>
  <div class="kpis">
    <div class="kpi"><small>Position value</small><strong>${money(h.value)}</strong><em>${num(h.shares,4)} shares bought ${fdate(m.purchase_date)} at ${money(h.fill)}</em></div>
    <div class="kpi"><small>P/L</small><strong class="${cls(h.pl)}">${smoney(h.pl)}</strong><em class="${cls(h.pl)}">${pct(h.pl_pct,2,true)} on ${money(h.invested)} invested</em><em style="display:block;color:var(--mute)">${money(h.commission)} commission paid from cash</em>${h.dividends?`<em style="display:block;color:var(--mute)">plus ${money(h.dividends)} dividends received</em>`:''}</div>
    <div class="kpi"><small>Expected annual return</small><strong>${pct(k.exp_annual)}</strong><em>σ ${pct(k.std_annual)} annual</em></div>
    <div class="kpi"><small>30-day volatility</small><strong>${pct(h.vol_30d,1)}</strong><em>1-year ${pct(h.vol_1y,1)}</em></div>
    <div class="kpi"><small>Year to date</small><strong class="${cls(h.ret_ytd)}">${pct(h.ret_ytd,1,true)}</strong><em>1M ${pct(h.ret_1m,1,true)} · 1Y ${pct(h.ret_1y,1,true)} · 5Y ${pct(h.ret_5y,0,true)}</em></div>
  </div>
  <section style="margin-top:26px">
    <div class="bar"><h2 class="grow">Price history</h2><span id="ser"></span><span id="rng"></span></div>
    <div class="fig"><canvas id="px"></canvas></div>
    <p class="cap" id="pxcap"></p>
    <h3 style="margin:22px 0 6px">Rolling 30-day volatility, annualised</h3>
    <div class="fig sm"><canvas id="vol"></canvas></div>
  </section>
  <section class="band" style="margin-top:30px"><div class="grid2">
    <div class="ruled"><h3>Return and risk (daily log returns)</h3>${RISK_ROWS.map(([key,l,f])=>`<div class="row"><span>${l}</span><b>${f(k[key])}</b></div>`).join('')}</div>
    <div class="ruled"><h3>Weekly returns</h3>
      <div class="row"><span>Average weekly return</span><b>${pct(w.avg_weekly,3)}</b></div>
      <div class="row"><span>Weekly variance</span><b>${sci(w.weekly_var)}</b></div>
      <div class="row"><span>Weekly std dev</span><b>${pct(w.weekly_std,2)}</b></div>
      <div class="row"><span>Annualised return (× 52)</span><b>${pct(w.ann_return)}</b></div>
      <div class="row"><span>Annualised std dev</span><b>${pct(w.ann_std)}</b></div>
      <div class="row"><span>Weeks of data</span><b>${w.weeks}</b></div>
      <h3>${m.year} high and low (closing prices)</h3>
      <div class="row"><span>High · ${fdate(y.high_date)}</span><b>${money(y.high)}</b></div>
      <div class="row"><span>Low · ${fdate(y.low_date)}</span><b>${money(y.low)}</b></div>
      <div class="row"><span>Latest close</span><b>${money(y.latest)}</b></div>
      <div class="row"><span>From high / from low</span><b><span class="neg">${pct(y.pct_below_high,1)}</span> / <span class="pos">${pct(y.pct_above_low,1,true)}</span></b></div>
    </div></div></section>
  <section class="band"><div class="grid2">
    <div><h2>Co-movement with the other holdings</h2>
      <div class="scroll"><table><thead><tr><th>Stock</th><th>Correlation</th><th>VARCOVAR (Excel)</th><th>Covariance (Python)</th></tr></thead><tbody>
      ${others.map(o=>`<tr><td><a href="#${o.u.toLowerCase()}">${esc(o.u)}</a></td><td>${num(o.corr,3)}</td><td>${sci(o.vc)}</td><td>${sci(o.cov)}</td></tr>`).join('')}
      </tbody></table></div><p class="cap">Variance of ${esc(t)} itself: ${sci(D.varcovar[i][i])} (VARCOVAR, Excel), ${sci(D.cov[i][i])} (covariance, Python).</p></div>
    <div><h2>${m.year}: best and worst weeks</h2><div class="grid2" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:20px">
      <table><thead><tr><th>Week ending</th><th>Best</th></tr></thead><tbody>${y.best_weeks.map(([d,v])=>`<tr><td>${fdate(d)}</td><td class="${cls(v)}">${pct(v,2,true)}</td></tr>`).join('')}</tbody></table>
      <table><thead><tr><th>Week ending</th><th>Worst</th></tr></thead><tbody>${y.worst_weeks.map(([d,v])=>`<tr><td>${fdate(d)}</td><td class="${cls(v)}">${pct(v,2,true)}</td></tr>`).join('')}</tbody></table>
    </div></div></div></section>
  <section class="band pager"><a href="#${prev.toLowerCase()}">← ${esc(prev)}</a><a href="#overview">Back to the portfolio</a><a href="#${next.toLowerCase()}">${esc(next)} →</a></section>`;
  priceChart(t,pal());
}

function priceChart(t,c){
  let ser='Close',rng='1Y',ch,vch;
  const draw=()=>{
    const i0=startIndex(P.dates,rng),lab=P.dates.slice(i0);
    const data=(ser==='Close'?P.close[t]:P.adj[t]).slice(i0);
    const first=data.find(v=>v!=null),last=data[data.length-1],chg=last/first-1,up=chg>=0;
    [ch,vch].forEach(x=>x&&x.destroy());
    const o=baseOpts(c,{yFmt:v=>'$'+v.toLocaleString()});
    o.plugins.tooltip.callbacks={label:x=>`${ser}: ${money(x.parsed.y)}`};
    ch=new Chart($('#px'),{type:'line',data:{labels:lab,datasets:[{data,borderColor:up?c.teal:c.neg,backgroundColor:up?c.fill:c.nfill,fill:true,borderWidth:1.6,pointRadius:0,tension:0}]},options:o});
    const vo=baseOpts(c,{yFmt:v=>(v*100).toFixed(0)+'%'});vo.plugins.tooltip.callbacks={label:x=>'30D vol: '+pct(x.parsed.y,1)};
    vch=new Chart($('#vol'),{type:'line',data:{labels:lab,datasets:[{data:P.vol30[t].slice(i0),borderColor:c.mute,borderWidth:1.3,pointRadius:0}]},options:vo});
    charts.push(ch,vch);
    const vals=data.filter(v=>v!=null);
    $('#pxcap').innerHTML=`${fdate(lab[0])} to ${fdate(lab[lab.length-1])}: <b class="${cls(chg)}">${pct(chg,1,true)}</b> on ${ser==='Close'?'the actual closing price':'the dividend- and split-adjusted close (total return)'}. High ${money(Math.max(...vals))}, low ${money(Math.min(...vals))}.`;
  };
  rangeButtons($('#ser'),['Close','Adj Close'],ser,k=>{ser=k;draw()});
  rangeButtons($('#rng'),['1M','3M','6M','YTD','1Y','5Y','10Y'],rng,k=>{rng=k;draw()});
  draw();
}

/* ------------------------------------------------------------ ROUTER */
function route(){
  kill();
  const h=location.hash.slice(1).toUpperCase();
  const t=D.tickers.includes(h)?h:null;
  nav(t||'overview');
  t?stock(t):overview();
  if(t||location.hash==='#overview')scrollTo({top:0});
}

async function getJSON(u,opt){const r=await fetch(u,opt);let j=null;try{j=await r.json()}catch(e){}if(!r.ok)throw new Error((j&&j.error)||('HTTP '+r.status));return j}
async function load(){
  const [d,p]=await Promise.all([getJSON('data/data.json'),getJSON('data/prices.json')]);
  D=d;P=p;
  $('#stamp').textContent=`Prices as of the ${fdate(D.meta.as_of)} close · pulled from Yahoo Finance ${D.meta.generated}`;
  $('#foot').textContent=`${D.meta.title} by ${D.meta.owner} · Data: ${D.meta.source} · Built with Python (pandas, NumPy, yfinance, Flask) and Chart.js. Not investment advice.`;
}
/* "Pull latest data": only shown when the page is served by the Python app (app.py), which runs yfinance */
async function pull(){
  const b=$('#pull'),msg=$('#pullmsg');
  b.disabled=true;b.textContent='Pulling from Yahoo Finance…';msg.textContent='';
  try{
    const r=await getJSON('api/refresh',{method:'POST'});
    const y=scrollY;await load();route();scrollTo({top:y});
    msg.textContent=r.message||'Updated.';
  }catch(e){msg.textContent='Could not pull new data: '+e.message+'. The figures shown are unchanged.'}
  finally{b.disabled=false;b.textContent='Pull latest data'}
}
load().then(()=>{
  route();
  addEventListener('hashchange',route);
  new MutationObserver(route).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',route);
  getJSON('api/status').then(st=>{if(st&&st.live){const b=$('#pull');b.hidden=false;b.addEventListener('click',pull)}}).catch(()=>{});
})
.catch(e=>{console.error(e);$('#err').style.display='block';$('#stamp').textContent='';});
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
