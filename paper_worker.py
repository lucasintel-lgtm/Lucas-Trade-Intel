#!/usr/bin/env python3
"""
Lucas Trade Intel — Paper Trading Worker
Runs every 15 min during market hours via GitHub Actions.
Fetches data (keys live in Actions secrets, never in the site),
scores 3 horizons with FIXED indicators, opens/closes paper positions,
and writes atomic JSON logs the dashboard reads.

Design rules (agreed July 27, 2026):
- Atomic facts only in the log. All slicing/overlays computed at view time.
- Up to 3 positions per ticker: one per horizon (intraday / swing / position).
- Every non-neutral signal is logged as a position, including WAIT and SKIP
  verdicts (shadow trades) so "followed recs vs took everything" is a filter.
- P&L is simulated on the UNDERLYING (shares), long or short. Spread labels
  and spread-quality metrics are recorded for analysis but not priced.
- Excess-vs-SPY is stored per trade: alpha = dir * (r_stock - r_SPY).
"""

import json, os, sys, time, math
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import urllib.request, urllib.parse

ET = ZoneInfo("America/New_York")
TRADIER_KEY = os.environ.get("TRADIER_API_KEY", "").strip()
FINNHUB_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
TRADIER = "https://api.tradier.com/v1"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_F      = os.path.join(DATA_DIR, "log.json")       # every trade, open+closed
SNAP_F     = os.path.join(DATA_DIR, "snapshot.json")   # latest scan, all tickers
META_F     = os.path.join(DATA_DIR, "meta.json")       # run info, equity curve

POS_DOLLARS = 500.0          # virtual $ per position
BASELINE_DATE = None         # set on first run, stored in meta

WATCHLIST = {
  "AI & Chips":   ["NVDA","AMD","AVGO","TSM","ARM","ASML","AMAT","MU","SMCI","QCOM","INTC","MRVL","KLAC","LRCX"],
  "AI Software":  ["MSFT","GOOGL","META","AMZN","PLTR","CRM","NOW","ORCL","SNOW","DDOG","NET","SOUN","AI","BBAI"],
  "Quantum":      ["IONQ","RGTI","QUBT","QBTS","IBM"],
  "Space & LEO":  ["RKLB","ASTS","LUNR"],
  "Nuclear":      ["CEG","CCJ","OKLO","SMR","NNE","VST","UEC"],
  "Energy":       ["XOM","CVX","OXY","FSLR","NEE","SLB","DVN","GEV","PWR"],
  "EV & Battery": ["TSLA","RIVN","QS","ENVX"],
  "Commodities":  ["GLD","SLV","USO","GDX","COPX","MP","NEM","WPM"],
  "Macro ETFs":   ["SPY","QQQ","IWM","TLT","SOXX","SMH","ARKK","XLE"],
  "Crypto":       ["MSTR","COIN","HOOD"],
  "Robotics":     ["ISRG","PATH","TER"],
  "Photonics":    ["LITE","COHR","AAOI"],
}
TICKERS = sorted({t for ts in WATCHLIST.values() for t in ts})
SECTOR = {t: s for s, ts in WATCHLIST.items() for t in ts}

# Exit rules per horizon (target %, stop %, max hold)
HORIZONS = {
    "intraday": {"target": 1.2, "stop": -0.8, "max_days": 0},   # closes 15:55 ET same day
    "swing":    {"target": 5.0, "stop": -3.0, "max_days": 10},  # trading days
    "position": {"target": 10.0, "stop": -6.0, "max_days": 45}, # calendar days
}

# ── HTTP ──────────────────────────────────────────────────────────────────────
def http_json(url, headers=None, timeout=15):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  http fail {url.split('?')[0]}: {e}", file=sys.stderr)
        return None

def tradier(path, params):
    q = urllib.parse.urlencode(params)
    return http_json(f"{TRADIER}{path}?{q}",
                     {"Authorization": f"Bearer {TRADIER_KEY}", "Accept": "application/json"})

def finnhub(path, params):
    params = dict(params, token=FINNHUB_KEY)
    q = urllib.parse.urlencode(params)
    return http_json(f"https://finnhub.io/api/v1{path}?{q}")

# ── Indicators (FIXED — real MACD with a real 9-EMA signal line) ─────────────
def ema_series(vals, period):
    if len(vals) < period: return None
    k = 2 / (period + 1)
    out = [sum(vals[:period]) / period]
    for v in vals[period:]:
        out.append(v * k + out[-1] * (1 - k))
    return out  # aligned to vals[period-1:]

def ema_last(vals, period):
    s = ema_series(vals, period)
    return s[-1] if s else None

def macd_full(closes):
    """Real MACD: line = EMA12-EMA26, signal = EMA9 of the MACD line."""
    if len(closes) < 26 + 9: return None
    e12, e26 = ema_series(closes, 12), ema_series(closes, 26)
    line = [a - b for a, b in zip(e12[len(e12)-len(e26):], e26)]
    sig = ema_series(line, 9)
    if not sig: return None
    hist = line[-1] - sig[-1]
    prev_hist = line[-2] - sig[-2] if len(line) > 1 and len(sig) > 1 else hist
    return {"line": line[-1], "signal": sig[-1], "hist": hist,
            "cross_up": prev_hist <= 0 < hist, "cross_dn": prev_hist >= 0 > hist}

def rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    gains = losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        d = closes[i] - closes[i-1]
        if d > 0: gains += d
        else: losses -= d
    if losses == 0: return 100.0
    rs = (gains / period) / (losses / period)
    return round(100 - 100 / (1 + rs), 1)

def realized_vol(closes, n=20):
    """Annualized realized vol from daily closes, %."""
    if len(closes) < n + 1: return None
    rets = [math.log(closes[i] / closes[i-1]) for i in range(len(closes)-n, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 1)

def support_resistance(highs, lows, n=20):
    if len(highs) < n: return None, None
    rh, rl = sorted(highs[-n:], reverse=True)[:3], sorted(lows[-n:])[:3]
    return round(sum(rl)/3, 2), round(sum(rh)/3, 2)

# ── Scoring (direction can only come from coherent votes; ties = NEUTRAL) ────
def score_swing(px, chg, closes, highs, lows, vols):
    """1–4 week horizon. Daily bars. Fixed logic: no branch votes both ways."""
    reasons, warnings = [], []
    bull = bear = 0
    e20, e50 = ema_last(closes, 20), ema_last(closes, 50)
    m = macd_full(closes)
    r = rsi(closes)
    score = 0

    # Trend (max 30): full alignment scores; mixed structure votes NOTHING
    if e20 and e50:
        if px > e20 > e50:   bull += 3; score += 30; reasons.append("Uptrend: price > 20d > 50d")
        elif px < e20 < e50: bear += 3; score += 30; reasons.append("Downtrend: price < 20d < 50d")
        elif px > e20 and px > e50: bull += 2; score += 20; reasons.append("Price above both averages")
        elif px < e20 and px < e50: bear += 2; score += 20; reasons.append("Price below both averages")
        else: score += 8; warnings.append("Mixed trend structure — no directional vote")

    # Momentum (max 25): real MACD histogram + crossovers
    if m:
        if m["cross_up"]:   bull += 2; score += 25; reasons.append("MACD crossed up (fresh momentum turn)")
        elif m["cross_dn"]: bear += 2; score += 25; reasons.append("MACD crossed down (fresh momentum turn)")
        elif m["hist"] > 0: bull += 1; score += 15; reasons.append("MACD momentum positive")
        else:               bear += 1; score += 15; reasons.append("MACD momentum negative")

    # RSI regime (max 20): confirms, never leads
    lean_bull = bull >= bear
    if lean_bull:
        if 45 <= r <= 65: score += 20; reasons.append(f"RSI {r:.0f} healthy for longs")
        elif r > 70: score += 4; warnings.append(f"RSI {r:.0f} overbought — chasing risk")
        else: score += 10
    else:
        if 35 <= r <= 55: score += 20; reasons.append(f"RSI {r:.0f} healthy for shorts")
        elif r < 30: score += 4; warnings.append(f"RSI {r:.0f} oversold — bounce risk")
        else: score += 10

    # Volume confirmation (max 10)
    if vols and len(vols) >= 21:
        v_ratio = vols[-1] / (sum(vols[-21:-1]) / 20) if sum(vols[-21:-1]) else 1
        if v_ratio > 1.5: score += 10; reasons.append(f"Volume {v_ratio:.1f}x average — conviction behind move")
        elif v_ratio > 1.0: score += 6
        else: score += 3

    # Same-day chase guard (max 15 → penalties)
    a = abs(chg or 0)
    if a > 8: score -= 12; warnings.append(f"Already moved {chg:+.1f}% today — do not chase")
    elif a > 5: score -= 6; warnings.append(f"Big move today ({chg:+.1f}%) — wait for consolidation")
    else: score += 8

    # S/R context (±5)
    sup, res = support_resistance(highs, lows)
    if sup and res and px:
        if lean_bull and (px - sup)/px*100 < 3: score += 5; reasons.append("Near support — favorable entry")
        if lean_bull and (res - px)/px*100 < 2: score -= 5; warnings.append("Right under resistance")
        if not lean_bull and (res - px)/px*100 < 3: score += 5; reasons.append("Near resistance — favorable short entry")
        if not lean_bull and (px - sup)/px*100 < 2: score -= 5; warnings.append("Right above support")

    direction = "BULL" if bull > bear + 1 else "BEAR" if bear > bull + 1 else "NEUTRAL"
    score = max(0, min(100, round(score)))
    return {"score": score, "direction": direction, "reasons": reasons[:4],
            "warnings": warnings[:3], "rsi": r, "ema20": e20 and round(e20,2),
            "ema50": e50 and round(e50,2), "macd_hist": m and round(m["hist"], 3),
            "support": sup, "resistance": res}

def score_position(px, closes, earn_days):
    """1 month+ horizon. Longer averages, longer returns, earnings cycle."""
    reasons, warnings = [], []
    bull = bear = 0
    score = 0
    e50, e100 = ema_last(closes, 50), ema_last(closes, 100)
    if e50 and e100:
        if px > e50 > e100:   bull += 3; score += 35; reasons.append("Long-term uptrend: price > 50d > 100d")
        elif px < e50 < e100: bear += 3; score += 35; reasons.append("Long-term downtrend: price < 50d < 100d")
        else: score += 12; warnings.append("Long-term trend mixed")
    if len(closes) >= 61:
        r60 = (closes[-1]/closes[-61] - 1) * 100
        r20 = (closes[-1]/closes[-21] - 1) * 100
        if r60 > 10 and r20 > 0: bull += 2; score += 30; reasons.append(f"Strong 3-month trend ({r60:+.0f}%) still advancing")
        elif r60 < -10 and r20 < 0: bear += 2; score += 30; reasons.append(f"Persistent 3-month decline ({r60:+.0f}%)")
        elif abs(r60) > 25: score += 8; warnings.append(f"Extended 3-month move ({r60:+.0f}%) — mean-reversion risk")
        else: score += 15
    if earn_days < 45:
        score += 5; warnings.append(f"Earnings in {earn_days}d sits inside this holding window")
    else:
        score += 15; reasons.append("No earnings inside the holding window")
    direction = "BULL" if bull > bear + 1 else "BEAR" if bear > bull + 1 else "NEUTRAL"
    return {"score": max(0, min(100, round(score))), "direction": direction,
            "reasons": reasons[:4], "warnings": warnings[:3]}

def score_intraday(bars, day_open):
    """Same-day horizon from 5-min bars: VWAP, opening range, volume."""
    if not bars or len(bars) < 8: return None
    closes = [b["close"] for b in bars]; vols = [b["volume"] for b in bars]
    px = closes[-1]
    cum_pv = cum_v = 0.0
    for b in bars:
        typ = (b["high"] + b["low"] + b["close"]) / 3
        cum_pv += typ * b["volume"]; cum_v += b["volume"]
    vwap = cum_pv / cum_v if cum_v else px
    orb = bars[:6]  # first 30 min
    or_hi, or_lo = max(b["high"] for b in orb), min(b["low"] for b in orb)
    recent_v = sum(vols[-3:]) / 3
    early_v = sum(vols[:6]) / 6 if len(vols) >= 6 else recent_v
    v_ratio = recent_v / early_v if early_v else 1.0

    reasons, warnings = [], []
    bull = bear = 0; score = 0
    if px > vwap * 1.001: bull += 2; score += 30; reasons.append("Trading above VWAP")
    elif px < vwap * 0.999: bear += 2; score += 30; reasons.append("Trading below VWAP")
    else: score += 10; warnings.append("Pinned to VWAP — no intraday edge")
    if px > or_hi: bull += 2; score += 30; reasons.append("Broke above opening range")
    elif px < or_lo: bear += 2; score += 30; reasons.append("Broke below opening range")
    else: score += 10
    if v_ratio > 1.3: score += 20; reasons.append(f"Volume expanding ({v_ratio:.1f}x open)")
    elif v_ratio > 0.8: score += 12
    else: score += 5; warnings.append("Volume fading — moves less reliable")
    if day_open and abs(px/day_open - 1) * 100 > 6:
        score -= 10; warnings.append("Large move already in — late entry risk")
    else:
        score += 20
    direction = "BULL" if bull > bear + 1 else "BEAR" if bear > bull + 1 else "NEUTRAL"
    return {"score": max(0, min(100, round(score))), "direction": direction,
            "reasons": reasons[:3], "warnings": warnings[:2],
            "vwap": round(vwap, 2), "or_high": round(or_hi,2), "or_low": round(or_lo,2)}

def decision_from(score, direction):
    if direction == "NEUTRAL": return "SKIP"
    if score >= 70: return "TRADE"
    if score >= 55: return "WAIT"
    return "SKIP"

# ── Options spread quality (breakeven vs paid probability) ───────────────────
def spread_quality(chain, direction, px):
    if not chain: return None
    typ = "call" if direction == "BULL" else "put"
    cs = [o for o in chain if o.get("option_type") == typ and o.get("greeks")
          and o["greeks"].get("delta") is not None and (o.get("bid") or 0) > 0 and (o.get("ask") or 0) > 0]
    long = next((o for o in cs if 0.40 <= abs(o["greeks"]["delta"]) <= 0.60), None)
    if not long: return None
    short = next((o for o in cs if 0.20 <= abs(o["greeks"]["delta"]) <= 0.38 and
                  (o["strike"] > long["strike"] if direction == "BULL" else o["strike"] < long["strike"])), None)
    if not short: return None
    debit = round(long["ask"] - short["bid"], 2)
    width = abs(long["strike"] - short["strike"])
    if width <= 0 or debit <= 0: return None
    breakeven_prob = round(debit / width, 3)          # win rate you NEED
    market_prob = round(abs(short["greeks"]["delta"]), 3)  # rough odds market SELLS you
    return {"long_strike": long["strike"], "short_strike": short["strike"],
            "debit": debit, "width": width,
            "breakeven_prob": breakeven_prob, "market_prob": market_prob,
            "edge": round(market_prob - breakeven_prob, 3),   # positive = fair or better
            "min_oi": min(long.get("open_interest") or 0, short.get("open_interest") or 0),
            "iv_avg": round(sum(o["greeks"].get("mid_iv", 0) for o in (long, short)) / 2 * 100, 1)}

# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_quotes():
    d = tradier("/markets/quotes", {"symbols": ",".join(TICKERS), "greeks": "false"})
    q = (d or {}).get("quotes", {}).get("quote")
    if not q: return {}
    if isinstance(q, dict): q = [q]
    return {x["symbol"]: x for x in q if x.get("last")}

def fetch_history(t):
    end = date.today(); start = end - timedelta(days=320)
    d = tradier("/markets/history", {"symbol": t, "interval": "daily",
                                     "start": start.isoformat(), "end": end.isoformat()})
    day = (d or {}).get("history", {}).get("day") if d and d.get("history") else None
    if not day: return None
    if isinstance(day, dict): day = [day]
    if len(day) < 30: return None
    return {"closes": [x["close"] for x in day], "highs": [x["high"] for x in day],
            "lows": [x["low"] for x in day], "vols": [x["volume"] for x in day]}

def fetch_timesales(t, now_et):
    start = now_et.replace(hour=9, minute=30, second=0).strftime("%Y-%m-%d %H:%M")
    end = now_et.strftime("%Y-%m-%d %H:%M")
    d = tradier("/markets/timesales", {"symbol": t, "interval": "5min",
                                       "start": start, "end": end, "session_filter": "open"})
    data = (d or {}).get("series", {}).get("data") if d and d.get("series") else None
    if not data: return None
    if isinstance(data, dict): data = [data]
    return data

def fetch_chain(t, min_dte, max_dte):
    d = tradier("/markets/options/expirations", {"symbol": t, "includeAllRoots": "true"})
    exps = (d or {}).get("expirations", {}).get("date") if d and d.get("expirations") else None
    if not exps: return None
    if isinstance(exps, str): exps = [exps]
    today = date.today()
    pick = next((e for e in exps if min_dte <= (date.fromisoformat(e) - today).days <= max_dte), None)
    if not pick: return None
    c = tradier("/markets/options/chains", {"symbol": t, "expiration": pick, "greeks": "true"})
    return (c or {}).get("options", {}).get("option") if c and c.get("options") else None

def fetch_earnings_days(t):
    d = finnhub("/calendar/earnings", {"symbol": t})
    cal = (d or {}).get("earningsCalendar") or []
    for x in cal:
        try:
            dd = (date.fromisoformat(x["date"]) - date.today()).days
            if dd >= 0: return dd
        except Exception: pass
    return 999

# ── State ─────────────────────────────────────────────────────────────────────
def load(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def save(path, obj):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, separators=(",", ":"))

def trading_days_between(d1, d2):
    n, d = 0, d1
    while d < d2:
        d += timedelta(days=1)
        if d.weekday() < 5: n += 1
    return n

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TRADIER_KEY:
        print("TRADIER_API_KEY missing"); sys.exit(1)
    now = datetime.now(ET)
    is_open = now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.hour < 16
    print(f"run {now.isoformat()} open={is_open}")

    log = load(LOG_F, [])
    meta = load(META_F, {"baseline": now.date().isoformat(), "runs": 0, "equity": []})
    quotes = fetch_quotes()
    if not quotes:
        print("no quotes — market data unavailable, exiting cleanly"); return
    spy_px = quotes.get("SPY", {}).get("last")

    # ── 1. EXITS: check every open position against live prices ──────────────
    closed_now = 0
    for tr in log:
        if tr["status"] != "open": continue
        q = quotes.get(tr["ticker"])
        if not q: continue
        px = q["last"]
        sgn = 1 if tr["direction"] == "BULL" else -1
        pnl = sgn * (px / tr["entry"] - 1) * 100
        h = HORIZONS[tr["horizon"]]
        opened = datetime.fromisoformat(tr["ts_open"])
        reason = None
        if pnl >= h["target"]: reason = "target"
        elif pnl <= h["stop"]: reason = "stop"
        elif tr["horizon"] == "intraday":
            if opened.date() < now.date() or (now.hour, now.minute) >= (15, 55): reason = "eod"
        elif tr["horizon"] == "swing":
            if trading_days_between(opened.date(), now.date()) >= h["max_days"]: reason = "time"
        elif (now.date() - opened.date()).days >= h["max_days"]: reason = "time"
        if reason:
            tr.update(status="closed", ts_close=now.isoformat(), exit=px,
                      spy_exit=spy_px, exit_reason=reason, pnl_pct=round(pnl, 2))
            if spy_px and tr.get("spy_entry"):
                r_spy = (spy_px / tr["spy_entry"] - 1) * 100
                # alpha = direction * (stock return - SPY return)
                tr["excess_pct"] = round(sgn * ((px / tr["entry"] - 1) * 100 - r_spy), 2)
            closed_now += 1

    # ── 2. SCAN & SCORE all tickers ───────────────────────────────────────────
    open_keys = {(t["ticker"], t["horizon"]) for t in log if t["status"] == "open"}
    snapshot, new_trades = [], 0
    movers = sorted(quotes.items(), key=lambda kv: abs(kv[1].get("change_percentage") or 0), reverse=True)
    intraday_set = {t for t, _ in movers[:20]} if is_open and (now.hour, now.minute) >= (10, 5) and now.hour < 15 else set()

    for i, t in enumerate(TICKERS):
        q = quotes.get(t)
        if not q: continue
        px, chg = q["last"], q.get("change_percentage") or 0
        hist = fetch_history(t)
        if not hist: continue
        earn = fetch_earnings_days(t) if i % 3 == 0 or t in intraday_set else 999  # throttle finnhub
        rv = realized_vol(hist["closes"])

        sw = score_swing(px, chg, hist["closes"], hist["highs"], hist["lows"], hist["vols"])
        po = score_position(px, hist["closes"], earn)
        it = None
        if t in intraday_set:
            bars = fetch_timesales(t, now)
            it = score_intraday(bars, q.get("open"))

        row = {"ticker": t, "sector": SECTOR.get(t, ""), "price": px, "change": round(chg, 2),
               "rv20": rv, "earnings_days": earn,
               "swing": {**sw, "decision": decision_from(sw["score"], sw["direction"])},
               "position": {**po, "decision": decision_from(po["score"], po["direction"])}}
        if it: row["intraday"] = {**it, "decision": decision_from(it["score"], it["direction"])}
        snapshot.append(row)

        # ── 3. OPEN positions: every non-neutral signal, one per (ticker,horizon)
        for hz, sc in (("intraday", it), ("swing", sw), ("position", po)):
            if not sc or sc["direction"] == "NEUTRAL": continue
            if (t, hz) in open_keys: continue
            if not is_open: continue
            if hz == "intraday" and now.hour >= 15: continue  # too late to open same-day
            dec = decision_from(sc["score"], sc["direction"])
            sq = None
            if hz in ("swing", "position") and dec == "TRADE" and new_trades < 8:
                dte = (7, 21) if hz == "swing" else (45, 90)
                sq = spread_quality(fetch_chain(t, *dte), sc["direction"], px)
            tr = {"id": f"{t}-{hz}-{int(now.timestamp())}", "ticker": t, "sector": SECTOR.get(t, ""),
                  "horizon": hz, "direction": sc["direction"], "decision": dec,
                  "score": sc["score"], "ts_open": now.isoformat(), "entry": px,
                  "spy_entry": spy_px, "day_change_at_entry": round(chg, 2),
                  "earnings_days": earn, "rv20": rv, "status": "open",
                  "reasons": sc["reasons"], "warnings": sc["warnings"]}
            if sq: tr["spread"] = sq
            log.append(tr)
            open_keys.add((t, hz))
            new_trades += 1
        time.sleep(0.12)

    # ── 4. Equity point + meta ────────────────────────────────────────────────
    closed = [t for t in log if t["status"] == "closed"]
    realized = sum(t["pnl_pct"] / 100 * POS_DOLLARS for t in closed)
    open_pnl = 0.0
    for t in log:
        if t["status"] == "open" and quotes.get(t["ticker"]):
            sgn = 1 if t["direction"] == "BULL" else -1
            open_pnl += sgn * (quotes[t["ticker"]]["last"] / t["entry"] - 1) * POS_DOLLARS
    meta["runs"] = meta.get("runs", 0) + 1
    meta["last_run"] = now.isoformat()
    meta["equity"].append({"ts": now.isoformat(), "realized": round(realized, 2),
                           "open": round(open_pnl, 2),
                           "n_open": sum(1 for t in log if t["status"] == "open"),
                           "n_closed": len(closed)})
    meta["equity"] = meta["equity"][-2000:]

    save(LOG_F, log)
    save(SNAP_F, {"ts": now.isoformat(), "rows": snapshot})
    save(META_F, meta)
    print(f"done: {len(snapshot)} scanned, {new_trades} opened, {closed_now} closed, "
          f"realized ${realized:+.0f}, open ${open_pnl:+.0f}")

if __name__ == "__main__":
    main()
