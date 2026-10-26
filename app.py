import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import concurrent.futures
import time
from datetime import datetime
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Institutional CPR & AI Screener", layout="wide")

@st.cache_data(ttl=86400) 
def get_nifty_100_universe():
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty100list.csv"
        df = pd.read_csv(url)
        tickers = {row['Company Name']: f"{row['Symbol']}.NS" for _, row in df.iterrows()}
        tickers = {"NIFTY 50": "^NSEI"} | tickers 
        return tickers
    except Exception as e:
        return {"NIFTY 50": "^NSEI", "RELIANCE": "RELIANCE.NS", "HDFCBANK": "HDFCBANK.NS", "TCS": "TCS.NS", "INFY": "INFY.NS"}

TICKERS = get_nifty_100_universe()

# ==========================================
# 2. DATA ACQUISITION ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def fetch_eod_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="1d")
        if data.empty: return None
        return data.iloc[-2] 
    except:
        return None

def fetch_intraday_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="5m")
        if data.empty: return None
        
        data.index = pd.to_datetime(data.index)
        data['VWAP'] = ta.vwap(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        has_volume = data['Volume'].sum() > 0
        data['VOL_SMA'] = ta.sma(data['Volume'], length=20) if has_volume else 0
        
        adx_df = ta.adx(data['High'], data['Low'], data['Close'], length=14)
        data['ADX'] = adx_df.iloc[:, 0] if adx_df is not None and not adx_df.empty else 0
            
        return data
    except:
        return None

# ==========================================
# 3. MATHEMATICAL ENGINES & INDEX FILTER
# ==========================================
def calculate_pivots(prev_day):
    high, low, close = float(np.squeeze(prev_day['High'])), float(np.squeeze(prev_day['Low'])), float(np.squeeze(prev_day['Close']))
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = (pivot - bc) + pivot
    range_hl = high - low
    
    return {
        "Pivot": pivot, "TC": tc, "BC": bc,
        "CPR_Top": max(tc, bc), "CPR_Bot": min(tc, bc),
        "R3": close + (range_hl * 1.1 / 4), "S3": close - (range_hl * 1.1 / 4)
    }

def generate_signals(ticker, live_data, pivots, params, nifty_bias):
    current_price = live_data['Close'].iloc[-1]
    current_vwap = live_data['VWAP'].iloc[-1]
    current_rsi = live_data['RSI'].iloc[-1]
    current_vol, vol_sma = live_data['Volume'].iloc[-1], live_data['VOL_SMA'].iloc[-1]
    current_adx = live_data['ADX'].iloc[-1]
    
    signal, entry, sl, target = "NEUTRAL", None, None, None
    cpr_top, cpr_bot, r3, s3, pivot = pivots["CPR_Top"], pivots["CPR_Bot"], pivots["R3"], pivots["S3"], pivots["Pivot"]
    
    prox = params['proximity'] / 100.0  
    short_prox_ok = (r3 * (1 - prox)) <= current_price <= (r3 * (1 + prox)) if params['use_proximity'] else True
    long_prox_ok = (s3 * (1 - prox)) <= current_price <= (s3 * (1 + prox)) if params['use_proximity'] else True

    short_vwap_ok, long_vwap_ok = current_price < current_vwap, current_price > current_vwap

    short_rsi_ok = current_rsi < params['rsi_short'] if params['use_rsi'] else True
    long_rsi_ok = current_rsi > params['rsi_long'] if params['use_rsi'] else True

    vol_ok = (current_vol > vol_sma) if (params['use_volume'] and current_vol > 0) else True
    adx_ok = current_adx < params['adx_limit'] if params['use_adx'] else True

    nifty_long_ok = nifty_bias != "BEARISH" if nifty_bias else True
    nifty_short_ok = nifty_bias != "BULLISH" if nifty_bias else True
    
    if (cpr_bot <= r3 <= cpr_top) and short_vwap_ok and short_prox_ok and short_rsi_ok and vol_ok and adx_ok and nifty_short_ok:          
        signal, entry, sl, target = "SHORT ENTRY", current_price, cpr_top * 1.002, pivot        
            
    elif (cpr_bot <= s3 <= cpr_top) and long_vwap_ok and long_prox_ok and long_rsi_ok and vol_ok and adx_ok and nifty_long_ok:          
        signal, entry, sl, target = "LONG ENTRY", current_price, cpr_bot * 0.998, pivot        
            
    if (signal == "LONG ENTRY" and target <= entry) or (signal == "SHORT ENTRY" and target >= entry): 
        signal = "NEUTRAL"
            
    return {
        "Symbol": ticker, "Live Price": round(current_price, 2), "VWAP": round(current_vwap, 2),
        "Signal": signal, "Entry": round(entry, 2) if signal != "NEUTRAL" else "-",
        "Stop Loss": round(sl, 2) if signal != "NEUTRAL" else "-", "Target": round(target, 2) if signal != "NEUTRAL" else "-"
    }

# ==========================================
# 4. USER INTERFACE & CONTROLS
# ==========================================
st.title("⚡ Nifty 100 Intraday CPR Screener & AI Edge")

st.sidebar.title("⚙️ Engine Controls")
defaults = {
    'prox_toggle': True, 'prox_slider': 0.3,
    'rsi_toggle': True, 'rsi_short_slider': 50, 'rsi_long_slider': 50,
    'vol_toggle': True, 'adx_toggle': True, 'adx_slider': 35
}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

if st.sidebar.button("🔄 Reset All Filters"):
    for key, val in defaults.items(): st.session_state[key] = val
    st.rerun()

params = {
    'use_proximity': st.sidebar.checkbox("Enable Proximity Filter", value=st.session_state.prox_toggle, key="prox_toggle"),
    'proximity': st.sidebar.slider("↳ Proximity Band (%)", 0.1, 1.0, value=st.session_state.prox_slider, step=0.05, key="prox_slider"),
    'use_rsi': st.sidebar.checkbox("Enable RSI Filter", value=st.session_state.rsi_toggle, key="rsi_toggle"),
    'rsi_short': st.sidebar.slider("↳ Short RSI (<)", 30, 60, value=st.session_state.rsi_short_slider, step=1, key="rsi_short_slider"),
    'rsi_long': st.sidebar.slider("↳ Long RSI (>)", 40, 70, value=st.session_state.rsi_long_slider, step=1, key="rsi_long_slider"),
    'use_volume': st.sidebar.checkbox("Enable Vol > SMA", value=st.session_state.vol_toggle, key="vol_toggle"),
    'use_adx': st.sidebar.checkbox("Enable ADX Limit", value=st.session_state.adx_toggle, key="adx_toggle"),
    'adx_limit': st.sidebar.slider("↳ Max ADX", 15, 50, value=st.session_state.adx_slider, step=1, key="adx_slider")
}

if st.sidebar.button("Force Data Refresh"): st.cache_data.clear()
auto_refresh = st.sidebar.checkbox("Enable 5-Min Auto-Refresh", value=False)
if 'previous_signals' not in st.session_state: st.session_state.previous_signals = set()

# ==========================================
# 5. MARKET BIAS & SCREENER EXECUTION
# ==========================================
@st.cache_data(ttl=300)
def get_nifty_bias():
    nifty_data = fetch_intraday_data("^NSEI")
    if nifty_data is not None and not nifty_data.empty:
        return "BULLISH" if nifty_data['Close'].iloc[-1] > nifty_data['VWAP'].iloc[-1] else "BEARISH"
    return "NEUTRAL"

nifty_bias = get_nifty_bias()

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)
is_market_hours = now_ist.weekday() < 5 and (now_ist.replace(hour=9, minute=15) <= now_ist <= now_ist.replace(hour=15, minute=30))
status_msg = "🟢 Market Live (IST)" if is_market_hours else "🟡 Market Closed (Snapshot Mode)"
st.markdown(f"**Status:** {status_msg} | **Nifty 50 Bias:** **{nifty_bias}**")

def process_ticker(name, symbol, current_params, market_bias):
    prev, live = fetch_eod_data(symbol), fetch_intraday_data(symbol)
    if prev is not None and live is not None:
        return generate_signals(name, live, calculate_pivots(prev), current_params, market_bias)
    return None

master_grid, actionable_signals = [], []
progress_bar, status_display = st.progress(0), st.empty()

with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    future_to_ticker = {executor.submit(process_ticker, n, s, params, nifty_bias): n for n, s in TICKERS.items()}
    completed, total = 0, len(future_to_ticker)
    for future in concurrent.futures.as_completed(future_to_ticker):
        try:
            result = future.result()
            if result:
                master_grid.append(result)
                if result["Signal"] != "NEUTRAL": actionable_signals.append(result)
        except: pass 
        completed += 1
        progress_bar.progress(completed / total)

progress_bar.empty()
status_display.success("✅ Nifty 100 Scan Complete.")

st.subheader("🔥 Active Trade Setups")
if actionable_signals:
    st.dataframe(pd.DataFrame(actionable_signals), use_container_width=True)
    new_signals = set([s['Symbol'] for s in actionable_signals]) - st.session_state.previous_signals
    if new_signals:
        st.markdown("""<audio autoplay="true"><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg"></audio>""", unsafe_allow_html=True)
        st.success(f"🔔 New Setup Detected: {', '.join(new_signals)}")
    st.session_state.previous_signals = set([s['Symbol'] for s in actionable_signals])
else:
    st.info("No active setups. Relaxing parameters may yield results.")

# ==========================================
# 6. AI DEEP DIVE (Tier 2: NLP + KNN)
# ==========================================
if actionable_signals:
    st.markdown("---")
    valid_symbols = sorted([sig['Symbol'] for sig in actionable_signals])
    selected_ticker = st.selectbox("Select Asset for AI Analysis", valid_symbols)
    target_symbol = TICKERS[selected_ticker]
    active_signal_dir = [s['Signal'] for s in actionable_signals if s['Symbol'] == selected_ticker][0]

    colA, colB = st.columns(2)
    
    # NLP Shield
    with colA:
        if st.button(f"🔍 Run NLP Catalyst Check"):
            with st.spinner("Analyzing headlines..."):
                analyzer = SentimentIntensityAnalyzer()
                news = yf.Ticker(target_symbol).news
                if not news:
                    st.info("No recent news. Pure technical environment.")
                else:
                    scores = []
                    for article in news[:5]:
                        title = article.get('title', '')
                        if title:
                            score = analyzer.polarity_scores(title)['compound']
                            scores.append(score)
                            tag = "🚨 BEAR" if score <= -0.2 else ("🚀 BULL" if score >= 0.2 else "⚖️ NEUT")
                            st.write(f"- {title[:50]}... | **{score:.2f} {tag}**")
                    avg = sum(scores) / len(scores) if scores else 0
                    if avg <= -0.25 and "LONG" in active_signal_dir:
                        st.error(f"**VETO ACTIVE:** Toxic catalyst (Score: {avg:.2f}). DO NOT GO LONG.")
                    elif avg >= 0.25 and "SHORT" in active_signal_dir:
                        st.error(f"**VETO ACTIVE:** Bullish catalyst (Score: {avg:.2f}). DO NOT SHORT.")
                    else:
                        st.success("**ALL CLEAR:** No conflicting news detected.")

    # KNN Probability Matrix
    with colB:
        if st.button(f"🧬 Run KNN Probability Matrix"):
            with st.spinner("Computing 60-day historical tick matrix..."):
                hist = yf.Ticker(target_symbol).history(period="60d", interval="5m")
                if len(hist) < 500:
                    st.warning("Insufficient data for KNN.")
                else:
                    hist['VWAP'] = ta.vwap(high=hist['High'], low=hist['Low'], close=hist['Close'], volume=hist['Volume'])
                    hist['RSI'] = ta.rsi(hist['Close'], length=14)
                    adx_df = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14)
                    hist['ADX'] = adx_df.iloc[:, 0] if adx_df is not None else 0
                    hist['VWAP_Dist'] = abs((hist['Close'] - hist['VWAP']) / hist['VWAP']) * 100
                    
                    forward_ret = (hist['Close'].shift(-12) - hist['Close']) / hist['Close']
                    hist['Result'] = np.where(forward_ret < -0.005, 1, 0) if "SHORT" in active_signal_dir else np.where(forward_ret > 0.005, 1, 0)
                    
                    pool = hist[hist['RSI'] > 60].dropna() if "SHORT" in active_signal_dir else hist[hist['RSI'] < 40].dropna()
                    
                    if len(pool) < 20:
                        st.warning("Not enough similar extremes historically.")
                    else:
                        X, y = pool[['RSI', 'ADX', 'VWAP_Dist']], pool['Result']
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        knn = KNeighborsClassifier(n_neighbors=9, weights='distance')
                        knn.fit(X_scaled, y)
                        
                        live_row = hist.iloc[-1]
                        live_scaled = scaler.transform([[live_row['RSI'], live_row['ADX'], live_row['VWAP_Dist']]])
                        prob = knn.predict_proba(live_scaled)[0][1] * 100
                        
                        st.metric("Historical Setup Win-Rate (1-Hour Forward)", f"{prob:.1f}%")
                        if prob >= 65: st.success("🟢 HIGH PROBABILITY ENTRY")
                        elif prob >= 45: st.info("🟡 NEUTRAL EDGE")
                        else: st.error("🔴 LOW PROBABILITY. AVOID.")

if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()