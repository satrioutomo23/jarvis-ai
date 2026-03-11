import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
import feedparser
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# =========================================================
# 1. CORE SYSTEM & CONFIG
# =========================================================
st.set_page_config(page_title="Jarvis v13.0 Omni-Sovereign", layout="wide", page_icon="🔱")
st_autorefresh(interval=300000, key="jarvis_heartbeat")

# Keamanan API Key melalui Streamlit Secrets
if "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
else:
    st.error("❌ API Key Gemini Missing! Masukkan GEMINI_KEY di Advanced Settings > Secrets.")
    st.stop()

# =========================================================
# 2. THE BRAIN: ANALYTICS & FUNCTIONS
# =========================================================
@st.cache_data(ttl=300)
def fetch_master_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return pd.DataFrame()

def run_backtest(df):
    """Fungsi Validator untuk menghitung Win Rate"""
    df = df.copy()
    df['Signal'] = np.where(df['Score'] >= 75, 1, 0)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    valid_trades = (df['Strategy_Returns'] != 0).sum()
    win_rate = (df['Strategy_Returns'] > 0).sum() / valid_trades * 100 if valid_trades > 0 else 0
    cum_profit = (df['Strategy_Returns'] + 1).prod() - 1
    return win_rate, cum_profit * 100

def detect_candle_patterns(df):
    """Mata AI: Pengenalan Pola Visual Candlestick"""
    patterns = []
    last, prev = df.iloc[-1], df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    range_total = last['High'] - last['Low']
    if range_total == 0: return "Neutral"
    
    # Hammer
    if body < (range_total * 0.3) and (last['Low'] < min(last['Open'], last['Close']) - body):
        patterns.append("🔨 Hammer")
    # Engulfing
    if last['Close'] > prev['Open'] and last['Open'] < prev['Close'] and prev['Close'] < prev['Open']:
        patterns.append("🔥 Engulfing")
    # Doji
    if body < (range_total * 0.1):
        patterns.append("⚖️ Doji")
    return ", ".join(patterns) if patterns else "Neutral"

def analyze_supreme_logic(df, ihsg_df=None):
    """Otak Analisis: Gabungan VPA, RS, MFI, dan Scoring"""
    if df.empty or len(df) < 50: return pd.DataFrame()
    
    # Technical Ribbon
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    # Money Flow Index (MFI)
    tp_idx = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp_idx * df['Volume']
    pos_mf = mf.where(tp_idx.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp_idx.diff() < 0, 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / neg_mf)))
    
    # VPA: Trap & Accumulation
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    price_chg = df['Close'].pct_change()
    df['VPA_Desc'] = "Sideways"
    df.loc[(df['Vol_Ratio'] > 1.5) & (price_chg > 0.02), 'VPA_Desc'] = "Accumulation"
    df.loc[(df['Vol_Ratio'] > 1.8) & (price_chg.abs() < 0.005), 'VPA_Desc'] = "🚨 BULL TRAP"
    
    # Relative Strength (RS) vs IHSG
    if ihsg_df is not None and not ihsg_df.empty:
        ih_c = ihsg_df['Close'].reindex(df.index, method='ffill')
        raw_rs = (df['Close'] / df['Close'].iloc[0]) / (ih_c / ih_c.iloc[0])
        df['RS'] = raw_rs.rolling(5).mean()
    else: df['RS'] = 1.0
    
    # Sovereign Scoring System
    df['Score'] = ((df['MFI'] > 55).astype(int) * 25) + \
                  ((df['Close'] > df['MA20']).astype(int) * 25) + \
                  ((df['RS'] > df['RS'].shift(1)).astype(int) * 25) + \
                  ((df['MA5'] > df['MA20']).astype(int) * 25)
    return df.dropna()

# =========================================================
# 3. INTERFACE: OMNI-COMMAND CENTER
# =========================================================
watchlist = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "ADRO.JK", "GOTO.JK", "ANTM.JK", "PTBA.JK", "MEDC.JK", "BRIS.JK", "TPIA.JK"]
st.title("🔱 Jarvis v13.0 Omni-Sovereign")

ihsg = fetch_master_data("^JKSE")

with st.sidebar:
    st.header("🛡️ Tactical Shield")
    alert_box = st.container()
    st.divider()
    cap = st.number_input("Modal Capital (Rp)", value=10000000, step=1000000)
    risk_pct = st.select_slider("Risk per Trade (%)", options=[0.5, 1.0, 2.0], value=1.0)
    sel = st.selectbox("🎯 Target Select", watchlist)
    st.divider()
    st.info("Status: Live Analytics Active")

tab_radar, tab_sniper, tab_validate, tab_oracle = st.tabs(["🚀 GLOBAL RADAR", "🎯 TACTICAL SNIPER", "📈 VALIDATOR", "🧠 OMNI-INTEL"])

# --- TAB 1: RADAR (Global Screener) ---
with tab_radar:
    if st.button("🛰️ EXECUTE SUPREME SCAN"):
        res = []
        for t in watchlist:
            d = analyze_supreme_logic(fetch_master_data(t), ihsg)
            if not d.empty:
                c_val = d['Close'].iloc[-1]
                atr_val = d['ATR'].iloc[-1]
                score = d['Score'].iloc[-1]
                res.append({
                    "Ticker": t, "Price": f"{c_val:,.0f}", "Score": score,
                    "TP1": f"{int(c_val + (atr_val * 2)):,.0f}",
                    "VPA": d['VPA_Desc'].iloc[-1], "Pattern": detect_candle_patterns(d)
                })
                if score >= 90: alert_box.warning(f"🚀 HIGH CONVICTION: {t}")
        
        df_res = pd.DataFrame(res)
        c1, c2 = st.columns([3, 1])
        with c1: st.dataframe(df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)
        with c2: st.bar_chart(df_res.set_index('Ticker')['Score'])

# --- TAB 2: TACTICAL SNIPER ---
with tab_sniper:
    df = analyze_supreme_logic(fetch_master_data(sel), ihsg)
    if not df.empty:
        c_val, s_val, atr = df['Close'].iloc[-1], df['Score'].iloc[-1], df['ATR'].iloc[-1]
        sl = int(c_val - (atr * 2))
        tp_1 = int(c_val + (atr * 2))
        tp_2 = int(c_val + (atr * 4))
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"Rp {c_val:,.0f}")
        m2.metric("Apex Score", f"{s_val}%")
        m3.metric("Pattern", detect_candle_patterns(df))
        m4.metric("VPA Status", df['VPA_Desc'].iloc[-1])

        l_col, r_col = st.columns([1.5, 2.5])
        with l_col:
            st.subheader("⚔️ Trading Plan")
            st.write(f"**Entry:** {int(c_val):,.0f}")
            st.error(f"**Stop Loss:** {sl:,.0f}")
            st.success(f"**Target TP 1:** {tp_1:,.0f}")
            st.success(f"**Target TP 2:** {tp_2:,.0f}")
            
            st.divider()
            risk_amt = cap * (risk_pct / 100)
            risk_per_share = c_val - sl
            lots = int(risk_amt / (risk_per_share * 100)) if risk_per_share > 0 else 0
            st.info(f"**Size:** {lots} Lots")
            
            try:
                m_prop = Prophet(daily_seasonality=True).fit(df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}))
                future = m_prop.predict(m_prop.make_future_dataframe(periods=2))
                st.write(f"🔮 **AI Forecast:** Rp {future['yhat'].iloc[-1]:,.0f}")
            except: pass

        with r_col:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index[-60:], open=df['Open'][-60:], high=df['High'][-60:], low=df['Low'][-60:], close=df['Close'][-60:], name="Price"), row=1, col=1)
            fig.add_hline(y=tp_1, line_dash="dash", line_color="green", annotation_text="TP 1")
            fig.add_hline(y=sl, line_dash="dash", line_color="red", annotation_text="SL")
            
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['MFI'][-60:], line=dict(color='cyan'), fill='tozeroy', name="MFI Flow"), row=2, col
