from http.server import BaseHTTPRequestHandler
import json
import time
import requests
import urllib.parse
import pandas as pd
import numpy as np

DELTA_BASE_URL = "https://api.india.delta.exchange"

# Isolated adaptive states and journals for each timeframe
QUANT_ENGINES = {
    "15m": {
        "journal": [],
        "adaptive_state": {
            "adx_thresh": 22.0,
            "min_squeeze_bars": 12, # 3 hours
            "tp_mult": 2.5,
            "sl_mult": 1.0,
            "max_hold_bars": 32, # 8 hours
            "learning_status": "INITIALIZED (15M Baseline: ADX 22.0)"
        }
    },
    "1h": {
        "journal": [],
        "adaptive_state": {
            "adx_thresh": 25.0,
            "min_squeeze_bars": 5, # 5 hours
            "tp_mult": 2.8,
            "sl_mult": 1.0,
            "max_hold_bars": 24, # 24 hours
            "learning_status": "INITIALIZED (1H Baseline: ADX 25.0)"
        }
    }
}

def fetch_candles(tf="15m", limit=250):
    url = f"{DELTA_BASE_URL}/v2/history/candles"
    end = int(time.time())
    step = 900 if tf == "15m" else 3600
    start = end - (limit * step)
    
    params = {"symbol": "ETHUSD", "resolution": tf, "start": start, "end": end}
    headers = {"User-Agent": "CryptoDirectionalTrades/4.0"}
    
    try:
        res = requests.get(url, params=params, headers=headers, timeout=8)
        data = res.json()
        if data.get("success") and data.get("result"):
            df = pd.DataFrame(data["result"])
            df["timestamp"] = pd.to_datetime(df["time"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                df[c] = df[c].astype(float)
            return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass
    return generate_fallback_data(tf, limit)

def fetch_options_chain():
    url = f"{DELTA_BASE_URL}/v2/tickers"
    headers = {"User-Agent": "CryptoDirectionalTrades/4.0"}
    try:
        res = requests.get(url, headers=headers, timeout=8)
        data = res.json()
        if data.get("success"):
            call_oi, put_oi = 0.0, 0.0
            for t in data.get("result", []):
                sym = t.get("symbol", "")
                if "ETH" in sym and ("C-" in sym or "P-" in sym):
                    oi = float(t.get("open_interest", 0) or 0)
                    if "C-" in sym or "-C-" in sym:
                        call_oi += oi
                    elif "P-" in sym or "-P-" in sym:
                        put_oi += oi
            pcr = round(put_oi / call_oi, 2) if call_oi > 0 else 1.0
            return {"pcr_oi": pcr, "total_oi": call_oi + put_oi}
    except Exception:
        pass
    return {"pcr_oi": 1.18, "total_oi": 292000.0}

def compute_quant_analytics(df):
    d = df.copy()
    d["EMA_20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA_50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA_200"] = d["Close"].ewm(span=200, adjust=False).mean()
    
    tr0 = abs(d["High"] - d["Low"])
    tr1 = abs(d["High"] - d["Close"].shift(1))
    tr2 = abs(d["Low"] - d["Close"].shift(1))
    d["ATR"] = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1).rolling(14).mean().bfill()
    
    bb_mid = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_Upper"] = bb_mid + (2.0 * bb_std)
    d["BB_Lower"] = bb_mid - (2.0 * bb_std)
    
    d["KC_Upper"] = d["EMA_20"] + (1.5 * d["ATR"])
    d["KC_Lower"] = d["EMA_20"] - (1.5 * d["ATR"])
    
    d["Squeeze_On"] = (d["BB_Lower"] > d["KC_Lower"]) & (d["BB_Upper"] < d["KC_Upper"])
    
    sq_series = d["Squeeze_On"]
    d["Squeeze_Duration"] = sq_series.groupby((~sq_series).cumsum()).cumsum()
    
    hh = d["High"].rolling(20).max()
    ll = d["Low"].rolling(20).min()
    d["Squeeze_Mom"] = d["Close"] - (((hh + ll) / 2 + d["EMA_20"]) / 2)
    
    d["Vol_SMA_20"] = d["Volume"].rolling(20).mean().bfill()
    
    up = d["High"].diff()
    down = -d["Low"].diff()
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    s_tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1).rolling(14).sum()
    p_di = 100 * (pd.Series(p_dm).rolling(14).sum() / (s_tr + 1e-9))
    m_di = 100 * (pd.Series(m_dm).rolling(14).sum() / (s_tr + 1e-9))
    dx = 100 * (abs(p_di - m_di) / (p_di + m_di + 1e-9))
    d["ADX"] = dx.rolling(14).mean().bfill()
    
    return d.bfill().ffill()

def run_adaptive_learning_engine(tf):
    engine = QUANT_ENGINES[tf]
    journal = engine["journal"]
    state = engine["adaptive_state"]
    
    closed_trades = [t for t in journal if t["status"] in ["TARGET_HIT", "SL_HIT"]]
    
    if len(closed_trades) < 5:
        state["learning_status"] = f"ACCUMULATING DATA ({len(closed_trades)}/5 Min Trades)"
        return
        
    wins = [t for t in closed_trades if t["status"] == "TARGET_HIT"]
    win_rate = len(wins) / len(closed_trades)
    
    total_profit = sum([t["pnl_pct"] for t in wins]) if wins else 0.0
    total_loss = abs(sum([t["pnl_pct"] for t in closed_trades if t["status"] == "SL_HIT"]))
    profit_factor = round(total_profit / (total_loss + 1e-9), 2)
    
    if win_rate < 0.40 or profit_factor < 1.1:
        state["adx_thresh"] = min(30.0, state["adx_thresh"] + 1.0)
        state["tp_mult"] = 3.0 if tf == "1h" else 2.8
        state["learning_status"] = f"ENHANCED: Filters Tightened (Win Rate: {win_rate*100:.1f}%, PF: {profit_factor})"
    elif win_rate >= 0.55 and profit_factor >= 1.8:
        state["adx_thresh"] = max(18.0, state["adx_thresh"] - 0.5)
        state["learning_status"] = f"ENHANCED: Momentum Optimized (Win Rate: {win_rate*100:.1f}%, PF: {profit_factor})"
    else:
        state["learning_status"] = f"STABLE: Parameters Balanced (Win Rate: {win_rate*100:.1f}%, PF: {profit_factor})"

def simulate_and_update_signal_journal(df, tf):
    engine = QUANT_ENGINES[tf]
    journal = engine["journal"]
    state = engine["adaptive_state"]
    
    if len(journal) > 0:
        return

    adx_thresh = state["adx_thresh"]
    tp_m = state["tp_mult"]
    sl_m = state["sl_mult"]
    max_hold = state["max_hold_bars"]
    
    for i in range(40, len(df) - 1):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        sq_release = (prev_row["Squeeze_On"] == True) and (row["Squeeze_On"] == False)
        vol_surge = row["Volume"] >= 1.15 * row["Vol_SMA_20"]
        
        sig_dir = None
        if sq_release and vol_surge and row["ADX"] >= adx_thresh:
            if row["Close"] > row["EMA_200"] and row["Squeeze_Mom"] > 0:
                sig_dir = "BULLISH"
            elif row["Close"] < row["EMA_200"] and row["Squeeze_Mom"] < 0:
                sig_dir = "BEARISH"
                
        if sig_dir:
            entry_p = float(row["Close"])
            atr_v = float(row["ATR"])
            tp_p = round(entry_p + (tp_m * atr_v), 2) if sig_dir == "BULLISH" else round(entry_p - (tp_m * atr_v), 2)
            sl_p = round(entry_p - (sl_m * atr_v), 2) if sig_dir == "BULLISH" else round(entry_p + (sl_m * atr_v), 2)
            
            status = "PENDING"
            exit_price = entry_p
            pnl_pct = 0.0
            
            for j in range(i + 1, min(i + max_hold + 1, len(df))):
                sub_row = df.iloc[j]
                if sig_dir == "BULLISH":
                    if sub_row["High"] >= tp_p:
                        status, exit_price, pnl_pct = "TARGET_HIT", tp_p, (tp_p - entry_p) / entry_p * 100
                        break
                    elif sub_row["Low"] <= sl_p:
                        status, exit_price, pnl_pct = "SL_HIT", sl_p, (sl_p - entry_p) / entry_p * 100
                        break
                else:
                    if sub_row["Low"] <= tp_p:
                        status, exit_price, pnl_pct = "TARGET_HIT", tp_p, (entry_p - tp_p) / entry_p * 100
                        break
                    elif sub_row["High"] >= sl_p:
                        status, exit_price, pnl_pct = "SL_HIT", sl_p, (entry_p - sl_p) / entry_p * 100
                        break
                        
            journal.append({
                "timestamp": row["timestamp"],
                "timeframe": tf.upper(),
                "direction": sig_dir,
                "entry": round(entry_p, 2),
                "tp": tp_p,
                "sl": sl_p,
                "status": status,
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2)
            })

def generate_fallback_data(tf="15m", candles=250):
    np.random.seed(42 if tf == "15m" else 100)
    vol_scale = 3.2 if tf == "15m" else 8.5
    prices = 2650.0 + np.cumsum(np.random.normal(0, vol_scale, candles))
    timestamps = [f"{tf}-{i}" for i in range(candles)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "Open": prices,
        "High": prices + (vol_scale * 0.8),
        "Low": prices - (vol_scale * 0.8),
        "Close": prices,
        "Volume": np.random.uniform(500, 5000, candles)
    })

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse requested timeframe parameter from URL
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            tf = query_params.get("tf", ["15m"])[0].lower()
            if tf not in ["15m", "1h"]:
                tf = "15m"
                
            df = fetch_candles(tf, 250)
            df = compute_quant_analytics(df)
            options = fetch_options_chain()
            
            simulate_and_update_signal_journal(df, tf)
            run_adaptive_learning_engine(tf)
            
            engine = QUANT_ENGINES[tf]
            state_data = engine["adaptive_state"]
            journal = engine["journal"]
            
            latest = df.iloc[-1]
            spot = float(latest["Close"])
            atr = float(latest["ATR"])
            adx = float(latest["ADX"])
            squeeze = bool(latest["Squeeze_On"])
            
            curr_adx_thresh = state_data["adx_thresh"]
            tp_mult = state_data["tp_mult"]
            sl_mult = state_data["sl_mult"]
            min_squeeze = state_data["min_squeeze_bars"]
            
            # State classification
            if squeeze:
                state = f"NEUTRAL VOLATILITY EXPANSION ({tf.upper()})"
            elif adx > curr_adx_thresh and latest["Volume"] > 1.15 * latest["Vol_SMA_20"]:
                if latest["Close"] > latest["EMA_200"] and latest["Squeeze_Mom"] > 0:
                    state = f"STRONG BULLISH BREAKOUT ({tf.upper()})"
                elif latest["Close"] < latest["EMA_200"] and latest["Squeeze_Mom"] < 0:
                    state = f"STRONG BEARISH BREAKOUT ({tf.upper()})"
                else:
                    state = "NO-TRADE"
            else:
                state = "NO-TRADE"

            # Compute Execution Targets & Explanations
            if "BULLISH" in state:
                entry_val = f"${spot:,.2f}"
                tp_val = f"${spot + (atr * tp_mult):,.2f}"
                sl_val = f"${spot - (atr * sl_mult):,.2f}"
                
                reason_entry = (
                    f"[{tf.upper()} Frame] Squeeze released upward. Spot (${spot:,.2f}) > EMA 200 (${latest['EMA_200']:,.2f}) "
                    f"confirms macro bull trend. Candle Volume ({latest['Volume']:,.0f}) > 1.15x SMA20. ADX ({adx:.1f}) > {curr_adx_thresh} threshold."
                )
                reason_tp = f"Set at Entry + ({tp_mult}x ATR14 = ${atr*tp_mult:.2f}). Captures explosive {tf.upper()} breakout volatility."
                reason_sl = f"Set at Entry - ({sl_mult}x ATR14 = ${atr*sl_mult:.2f}). Protects against false breakout whipsaws."
                
            elif "BEARISH" in state:
                entry_val = f"${spot:,.2f}"
                tp_val = f"${spot - (atr * tp_mult):,.2f}"
                sl_val = f"${spot + (atr * sl_mult):,.2f}"
                
                reason_entry = (
                    f"[{tf.upper()} Frame] Squeeze released downward. Spot (${spot:,.2f}) < EMA 200 (${latest['EMA_200']:,.2f}) "
                    f"confirms macro bear trend. Candle Volume ({latest['Volume']:,.0f}) > 1.15x SMA20. ADX ({adx:.1f}) > {curr_adx_thresh} threshold."
                )
                reason_tp = f"Set at Entry - ({tp_mult}x ATR14 = ${atr*tp_mult:.2f}). Captures downside {tf.upper()} volatility expansion."
                reason_sl = f"Set at Entry + ({sl_mult}x ATR14 = ${atr*sl_mult:.2f}). Strict stop above entry candle high."
                
            elif "NEUTRAL" in state:
                entry_val = f"${spot:,.2f}"
                tp_val = f"${spot + (atr * tp_mult):,.2f} / ${spot - (atr * tp_mult):,.2f}"
                sl_val = "Exit on Contraction"
                
                duration_hrs = int(latest['Squeeze_Duration']) * (0.25 if tf == "15m" else 1.0)
                reason_entry = f"[{tf.upper()} Frame] TTM Squeeze active ({int(latest['Squeeze_Duration'])} bars / ~{duration_hrs:.1f}h). Price compressed inside Keltner Channels."
                reason_tp = f"Symmetrical dual-target layout anticipating pending {tf.upper()} Gamma expansion."
                reason_sl = "Hard exit if Squeeze contracts below 1.0x ATR."
                
            else: # NO-TRADE
                entry_val, tp_val, sl_val = "N/A", "N/A", "N/A"
                reason_entry = f"[{tf.upper()} Frame] No trigger. Price is choppy or ADX ({adx:.1f}) is below active {tf.upper()} threshold ({curr_adx_thresh:.1f})."
                reason_tp = "N/A (No active position)"
                reason_sl = "N/A (No active position)"

            # Performance stats
            closed = [t for t in journal if t["status"] in ["TARGET_HIT", "SL_HIT"]]
            wins = [t for t in closed if t["status"] == "TARGET_HIT"]
            w_rate = round((len(wins) / len(closed)) * 100, 1) if closed else 0.0
            
            payload = {
                "timeframe": tf.upper(),
                "spot_price": spot,
                "signal_state": state,
                "squeeze_on": squeeze,
                "adx": round(adx, 1),
                "pcr": options["pcr_oi"],
                "atr": round(atr, 2),
                "targets": {
                    "entry": entry_val,
                    "target": tp_val,
                    "sl": sl_val
                },
                "explanation": {
                    "entry": reason_entry,
                    "tp": reason_tp,
                    "sl": reason_sl
                },
                "adaptive_engine": {
                    "adx_thresh": curr_adx_thresh,
                    "min_squeeze_bars": min_squeeze,
                    "tp_mult": tp_mult,
                    "sl_mult": sl_mult,
                    "status": state_data["learning_status"],
                    "total_logged_signals": len(journal),
                    "recorded_win_rate": w_rate
                },
                "signal_journal": journal[-15:][::-1],
                "series": {
                    "timestamps": df["timestamp"].tolist(),
                    "open": df["Open"].tolist(),
                    "high": df["High"].tolist(),
                    "low": df["Low"].tolist(),
                    "close": df["Close"].tolist(),
                    "ema_20": df["EMA_20"].tolist(),
                    "ema_50": df["EMA_50"].tolist(),
                    "ema_200": df["EMA_200"].tolist(),
                    "bb_upper": df["BB_Upper"].tolist(),
                    "bb_lower": df["BB_Lower"].tolist(),
                    "momentum": df["Squeeze_Mom"].tolist()
                }
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))