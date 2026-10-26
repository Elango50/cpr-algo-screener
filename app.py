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

# Import our custom Kotak feed
from data_feed import start_kotak_feed, live_market_data, NIFTY_TICKERS

# ==========================================
# 1. SETUP & BACKGROUND PROCESSES
# ==========================================
st.set_page_config(page_title="Institutional CPR & AI Screener", layout="wide")

# Start WebSocket once per session
if 'ws_started' not in st.session_state:
    start_kotak_feed()
    st.session_state.ws_started = True

# Scan the hard-coded Nifty 100 list and the Nifty 50 index.
TICKERS = {ticker.replace(".NS", ""): ticker for ticker in NIFTY_TICKERS}

# ==========================================
# 2. DATA ACQUISITION
# ==========================================
@st.cache_data(ttl=3600)
def fetch_eod_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="5d", interval="1d")
        if data.empty: return None
        return data.iloc[-2] 
    except: return None

def fetch_intraday_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="5m")
        if data.empty or len(data) < 2: return None
        
        data.index = pd.to_datetime(data.index)
        data['VWAP'] = ta.vwap(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'])
        data['RSI'] = ta.rsi(data['Close'], length=14)
        adx_df = ta.adx(data['High'], data['Low'], data['Close'], length=14)
        data['ADX'] = adx_df.iloc[:, 0] if adx_df is not None and not adx_df.empty else 0
            
        return data
    except: return None

# ==========================================
# 3. CORE LOGIC (FIXED REPAINTING)
# ==========================================
def calculate_pivots(prev_day):
    h, l, c = float(np.squeeze(prev_day['High'])), float(np.squeeze(prev_day['Low'])), float(np.squeeze(prev_day['Close']))
    p = (h + l + c) / 3
    bc, tc = (h + l) / 2, (p - (h + l) / 2) + p
    r_hl = h - l
    return {"Pivot": p, "TC": tc, "BC": bc, "CPR_Top": max(tc, bc), "CPR_Bot": min(tc, bc), "R3": c + (r_hl * 1.1 / 4), "S3": c - (r_hl * 1.1 / 4)}

def generate_signals(ticker, live_data, pivots, params, nifty_bias):
    # Use Kotak Live LTP if available, fallback to yfinance
    live_close = live_market_data.get(ticker, live_data['Close'].iloc[-1])
    
    # CLOSED CANDLE EVALUATION (For strict triggering)
    eval_c, eval_o = live_data['Close'].iloc[-2], live_data['Open'].iloc[-2]
    eval_h, eval_l = live_data['High'].iloc[-2], live_data['Low'].iloc[-2]
    eval_vwap, eval_rsi, eval_adx = live_data['VWAP'].iloc[-2], live_data['RSI'].iloc[-2], live_data['ADX'].iloc[-2]
    
    sig, entry, sl, target = "NEUTRAL", None, None, None
    cpr_t, cpr_b, r3, s3, pvt = pivots["CPR_Top"], pivots["CPR_Bot"], pivots["R3"], pivots["S3"], pivots["Pivot"]
    
    eval_time = live_data.index[-2]
    eval_time_ist = eval_time.tz_convert('Asia/Kolkata') if getattr(eval_time, 'tzinfo', None) else eval_time.tz_localize('UTC').tz_convert('Asia/Kolkata')
        
    if eval_time_ist.hour < 9 or (eval_time_ist.hour == 9 and eval_time_ist.minute < 45):
        return {"Symbol": ticker, "Live Price": round(live_close, 2), "Signal": "NEUTRAL", "Entry": "-", "Stop Loss": "-", "Target": "-"}

    prox = params['proximity'] / 100.0  
    short_prox_ok = (r3 * (1 - prox)) <= eval_h <= (r3 * (1 + prox)) if params['use_proximity'] else True
    long_prox_ok = (s3 * (1 - prox)) <= eval_l <= (s3 * (1 + prox)) if params['use_proximity'] else True

    short_vwap_ok = eval_c > eval_vwap 
    long_vwap_ok = eval_c < eval_vwap  

    short_rsi_ok = eval_rsi > params['rsi_overbought'] if params['use_rsi'] else True
    long_rsi_ok = eval_rsi < params['rsi_oversold'] if params['use_rsi'] else True

    short_rejection_ok = eval_c < eval_o
    long_rejection_ok = eval_c > eval_o

    adx_ok = eval_adx < params['adx_limit'] if params['use_adx'] else True
    nifty_long_ok = nifty_bias != "BEARISH" if nifty_bias else True
    nifty_short_ok = nifty_bias != "BULLISH" if nifty_bias else True
    
    if (cpr_b <= r3 <= cpr_t) and short_vwap_ok and short_prox_ok and short_rsi_ok and short_rejection_ok and adx_ok and nifty_short_ok:          
        sig, entry, sl, target = "SHORT ENTRY", eval_c, eval_h * 1.002, pvt        
            
    elif (cpr_b <= s3 <= cpr_t) and long_vwap_ok and long_prox_ok and long_rsi_ok and long_rejection_ok and adx_ok and nifty_long_ok:          
        sig, entry, sl, target = "LONG ENTRY", eval_c, eval_l * 0.998, pvt        
            
    return {"Symbol": ticker, "Live Price": round(live_close, 2), "Signal": sig, 
            "Entry": round(entry, 2) if sig != "NEUTRAL" else "-", "Stop Loss": round(sl, 2) if sig != "NEUTRAL" else "-", 
            "Target": round(target, 2) if sig != "NEUTRAL" else "-"}

# ==========================================
# 4. UI & STATE MANAGEMENT
# ==========================================
st.title("⚡ Institutional CPR & AI Screener")

if 'active_trades' not in st.session_state: st.session_state.active_trades = {}

st.sidebar.title("⚙️ Engine Controls")
filter_defaults = {
    'prox_toggle': True,
    'prox_slider': 0.5,
    'rsi_toggle': True,
    'rsi_oversold_slider': 35,
    'rsi_overbought_slider': 65,
    'adx_toggle': True,
    'adx_slider': 35,
}

for key, value in filter_defaults.items():
    st.session_state.setdefault(key, value)

def reset_all_filters():
    for key, value in filter_defaults.items():
        st.session_state[key] = value

st.sidebar.button("🔄 Reset All Filters", on_click=reset_all_filters)

st.sidebar.markdown("---")
params = {
    'use_proximity': st.sidebar.checkbox("Enable Proximity Band", key='prox_toggle'),
    'proximity': st.sidebar.slider("↳ Band (%)", 0.1, 1.5, step=0.1, key='prox_slider'),
    'use_rsi': st.sidebar.checkbox("Enable RSI Exhaustion", key='rsi_toggle'),
    'rsi_oversold': st.sidebar.slider("↳ Oversold (Longs <)", 15, 45, key='rsi_oversold_slider'),
    'rsi_overbought': st.sidebar.slider("↳ Overbought (Shorts >)", 55, 85, key='rsi_overbought_slider'),
    'use_adx': st.sidebar.checkbox("Enable Trend Exhaustion", key='adx_toggle'),
    'adx_limit': st.sidebar.slider("↳ Max ADX Limit", 15, 50, key='adx_slider')
}
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

# ==========================================
# 5. MARKET SCANNER & LIVE PERSISTENCE
# ==========================================
@st.cache_data(ttl=300)
def get_nifty_bias():
    data = fetch_intraday_data("^NSEI")
    if data is not None and not data.empty: return "BULLISH" if data['Close'].iloc[-2] > data['VWAP'].iloc[-2] else "BEARISH"
    return "NEUTRAL"

nifty_bias = get_nifty_bias()
st.markdown(f"**Nifty 50 Bias:** **{nifty_bias}**")

master_grid = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(generate_signals, s, fetch_intraday_data(s), calculate_pivots(fetch_eod_data(s)), params, nifty_bias): s for n, s in TICKERS.items() if s != "^NSEI"}
    for f in concurrent.futures.as_completed(futures):
        try:
            res = f.result()
            if res: master_grid.append(res)
        except: pass 

ist = pytz.timezone('Asia/Kolkata')
for res in master_grid:
    sym = res["Symbol"]
    # Update state entries
    if "ENTRY" in res["Signal"] and sym not in st.session_state.active_trades:
        st.session_state.active_trades[sym] = {
            "Time": datetime.now(ist).strftime("%H:%M:%S"), "Symbol": sym, "Signal": res["Signal"], "Entry": res["Entry"],
            "Stop Loss": res["Stop Loss"], "Target": res["Target"], "Status": "ACTIVE"
        }

# Update live prices instantly from Kotak WebSocket
for sym, trade in st.session_state.active_trades.items():
    if trade['Status'] == 'ACTIVE':
        # Instantly apply Kotak LTP from background thread
        live_price = live_market_data.get(sym, trade.get("Live Price", trade["Entry"]))
        trade['Live Price'] = round(live_price, 2)
        
        if trade['Signal'] == 'SHORT ENTRY':
            if live_price >= trade['Stop Loss']: trade['Status'] = 'STOPPED OUT'
            elif live_price <= trade['Target']: trade['Status'] = 'TARGET REACHED'
        elif trade['Signal'] == 'LONG ENTRY':
            if live_price <= trade['Stop Loss']: trade['Status'] = 'STOPPED OUT'
            elif live_price >= trade['Target']: trade['Status'] = 'TARGET REACHED'

st.subheader("🛡️ Trade Management Dashboard")
active_list = list(st.session_state.active_trades.values())

if active_list:
    df_trades = pd.DataFrame(active_list)
    def highlight_status(val):
        return 'color: green; font-weight: bold' if val == 'TARGET REACHED' else 'color: red; font-weight: bold' if val == 'STOPPED OUT' else 'color: orange; font-weight: bold'
    st.dataframe(df_trades.style.map(highlight_status, subset=['Status']), use_container_width=True)
else:
    st.info("No active setups logged. Scanner is running...")

# ==========================================
# 6. AI DEEP DIVE & CHARTING
# ==========================================
available_for_ai = [t['Symbol'] for t in active_list]
if available_for_ai:
    st.markdown("---")
    selected_ticker = st.selectbox("Select Active Asset to Visualize & Analyze", available_for_ai)
    
    # 6a. Visual Validation
    with st.spinner(f"Loading chart for {selected_ticker}..."):
        c_data, p_data = fetch_intraday_data(selected_ticker), fetch_eod_data(selected_ticker)
    
    if c_data is not None and p_data is not None:
        pivots = calculate_pivots(p_data)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=c_data.index, open=c_data['Open'], high=c_data['High'], low=c_data['Low'], close=c_data['Close'], name="Price"))
        fig.add_hline(y=pivots["Pivot"], line_dash="solid", line_color="purple", annotation_text="Pivot")
        
        px = params['proximity'] / 100.0
        fig.add_hrect(y0=pivots["R3"]*(1-px), y1=pivots["R3"]*(1+px), fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hline(y=pivots["R3"], line_dash="dash", line_color="red", annotation_text="Cam R3")
        fig.add_hrect(y0=pivots["S3"]*(1-px), y1=pivots["S3"]*(1+px), fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hline(y=pivots["S3"], line_dash="dash", line_color="green", annotation_text="Cam S3")
        fig.update_layout(height=400, title=f"5-Min Profile: {selected_ticker}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # 6b. NLP & KNN
    st.subheader("🧠 AI Catalyst & Statistics Deep Dive")
    act_sig = st.session_state.active_trades[selected_ticker]['Signal']
    colA, colB = st.columns(2)
    
    with colA:
        if st.button(f"🔍 Run NLP Catalyst Check"):
            news = yf.Ticker(selected_ticker).news
            if not news: st.info("Pure technical environment. No recent news.")
            else:
                scores, analyzer = [], SentimentIntensityAnalyzer()
                for article in news[:5]:
                    title = article.get('title', '')
                    if title:
                        s = analyzer.polarity_scores(title)['compound']
                        scores.append(s)
                        st.write(f"- {title[:50]}... | **{s:.2f}**")
                avg = sum(scores) / len(scores) if scores else 0
                if avg <= -0.25 and "LONG" in act_sig: st.error(f"VETO: Toxic catalyst (Score: {avg:.2f}). DO NOT GO LONG.")
                elif avg >= 0.25 and "SHORT" in act_sig: st.error(f"VETO: Bullish catalyst (Score: {avg:.2f}). DO NOT SHORT.")
                else: st.success("ALL CLEAR: No conflicting news detected.")

    with colB:
        if st.button(f"🧬 Run KNN Probability Matrix"):
            with st.spinner("Computing 60-day historical tick matrix..."):
                h = yf.Ticker(selected_ticker).history(period="60d", interval="5m")
                if len(h) < 500: st.warning("Insufficient data for KNN.")
                else:
                    h['VWAP'], h['RSI'] = ta.vwap(high=h['High'], low=h['Low'], close=h['Close'], volume=h['Volume']), ta.rsi(h['Close'], length=14)
                    h['Dist'] = abs((h['Close'] - h['VWAP']) / h['VWAP']) * 100
                    h['Res'] = np.where((h['Close'].shift(-12) - h['Close']) / h['Close'] < -0.005, 1, 0) if "SHORT" in act_sig else np.where((h['Close'].shift(-12) - h['Close']) / h['Close'] > 0.005, 1, 0)
                    p = h[h['RSI'] > 50].dropna() if "SHORT" in act_sig else h[h['RSI'] < 50].dropna()
                    
                    if len(p) < 20: st.warning("Not enough historical extremes.")
                    else:
                        scaler = StandardScaler()
                        knn = KNeighborsClassifier(n_neighbors=9, weights='distance').fit(scaler.fit_transform(p[['RSI', 'Dist']]), p['Res'])
                        prob = knn.predict_proba(scaler.transform([[h.iloc[-1]['RSI'], h.iloc[-1]['Dist']]]))[0][1] * 100
                        st.metric("Historical Win-Rate (1-Hr Forward)", f"{prob:.1f}%")
                        if prob >= 65: st.success("🟢 HIGH PROBABILITY")
                        else: st.error("🔴 LOW PROBABILITY. AVOID.")

if auto_refresh:
    time.sleep(300)
    st.rerun()