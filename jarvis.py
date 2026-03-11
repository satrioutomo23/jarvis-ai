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
# 1. CORE SYSTEM & CONFIG (Evolusi v10.0 - v13.0)
# =========================================================
st.set_page_config(page_title="Jarvis v13.0 Omni-Sovereign", layout="wide", page_icon="🔱")
st_autorefresh(interval=300000, key="jarvis_heartbeat")

# Keamanan API Key
if "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
else:
    st.error("❌ API Key Gemini Missing! Check secrets.toml.")
    st.stop()

# =========================================================
# 2. THE BRAIN: ADVANCED ANALYTICS (Synthesis v11.5 & v12.0)
# =========================================================
@st.cache_data(ttl=300)
def fetch_master_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return pd.DataFrame()

def detect_candle_patterns(df):
    """Mata AI: Pengenalan Pola Visual (v12.0)"""
    patterns = []
    last, prev = df.iloc[-1], df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    range_total = last['High'] - last['Low']
    # Hammer Detection
    if body < (range_total * 0.3) and (last['Low'] < min(last['Open'], last['Close']) - body):
        patterns.append("🔨 Hammer")
    # Engulfing Detection
    if last['Close'] > prev['Open'] and last['Open'] < prev['Close'] and prev['Close'] < prev['Open']:
        patterns.append("🔥 Engulfing")
    # Doji Detection
    if body < (range_total * 0.1):
        patterns.append("⚖️ Doji")
    return ", ".join(patterns) if patterns else "Neutral"

def analyze_supreme_logic(df, ihsg_df=None):
    """Otak Analisis: Gabungan VPA, RS, MFI, dan Scoring (v10.8 - v12.5)"""
    if df.empty or len(df) < 50: return pd.DataFrame()
    
    # --- Modul Technical (MA & ATR) ---
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['BB_Upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    
    # --- Modul Money Flow (MFI Precision v10.8) ---
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos_mf = mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp.diff() < 0, 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / neg_mf)))
    
    # --- Modul VPA: Trap & Accumulation (v12.0) ---
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    price_chg = df['Close'].pct_change()
    df['VPA_Desc'] = "Sideways"
    df.loc[(df['Vol_Ratio'] > 1.5) & (price_chg > 0.02), 'VPA_Desc'] = "Accumulation"
    df.loc[(df['Vol_Ratio'] > 1.8) & (price_chg.abs() < 0.005), 'VPA_Desc'] = "🚨 BULL TRAP"
    
    # --- Modul Relative Strength (RS Smoothing v11.5) ---
    if ihsg_df is not None and not ihsg_df.empty:
        ih_c = ihsg_df['Close'].reindex(df.index, method='ffill')
        raw_rs = (df['Close'] / df['Close'].iloc[0]) / (ih_c / ih_c.iloc[0])
        df['RS'] = raw_rs.rolling(5).mean() # Smoothing filter
    else: df['RS'] = 1.0
    
    # --- Sovereign Scoring System (v10.8-v13.0) ---
    df['Score'] = ((df['MFI'] > 55).astype(int) * 25) + \
                  ((df['Close'] > df['MA20']).astype(int) * 25) + \
                  ((df['RS'] > df['RS'].shift(1)).astype(int) * 25) + \
                  ((df['MA5'] > df['MA20']).astype(int) * 25)
    return df.dropna()

# =========================================================
# 3. PERFORMANCE VALIDATOR (v10.8 Backtest Engine)
# =========================================================
def run_backtest(df):
    df = df.copy()
    df['Signal'] = np.where(df['Score'] >= 75, 1, 0)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    win_rate = (df['Strategy_Returns'] > 0).sum() / (df['Strategy_Returns'] != 0).sum() * 100 if (df['Strategy_Returns'] != 0).sum() > 0 else 0
    cum_profit = (df['Strategy_Returns'] + 1).prod() - 1
    return win_rate, cum_profit * 100

# =========================================================
# 4. INTERFACE: OMNI-COMMAND CENTER (v13.0)
# =========================================================
watchlist = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "ADRO.JK", "GOTO.JK", "ANTM.JK", "PTBA.JK", "MEDC.JK", "BRIS.JK", "TPIA.JK"]
st.title("🔱 Jarvis v13.0: Sovereign Omniscience")

ihsg = fetch_master_data("^JKSE")

# Sidebar: Risk & Alert Architect
with st.sidebar:
    st.header("🛡️ Tactical Risk")
    alert_box = st.container()
    st.divider()
    cap = st.number_input("Total Modal (Rp)", value=10000000, step=1000000)
    risk_pct = st.select_slider("Risk per Trade (%)", options=[0.5, 1.0, 2.0], value=1.0)
    sel = st.selectbox("🎯 Target Select", watchlist)
    st.divider()
    st.info("Status: Monitoring Real-Time Data")

# Tab System: Menyatukan Semua Fitur Utama
tab_scan, tab_sniper, tab_validate, tab_intel = st.tabs(["🚀 GLOBAL RADAR", "🎯 TACTICAL SNIPER", "📈 VALIDATOR", "🧠 OMNI-INTEL"])

# --- TAB 1: RADAR (Global Screener & Heatmap v13.0) ---
with tab_scan:
    if st.button("🛰️ EXECUTE SUPREME SCAN"):
        res = []
        for t in watchlist:
            d = analyze_supreme_logic(fetch_master_data(t), ihsg)
            if not d.empty:
                score = d['Score'].iloc[-1]
                res.append({
                    "Ticker": t, "Price": f"{d['Close'].iloc[-1]:,.0f}", "Score": score,
                    "VPA": d['VPA_Desc'].iloc[-1], "Pattern": detect_candle_patterns(d)
                })
                if score >= 90: alert_box.warning(f"🚀 HIGH CONVICTION: {t} ({score}%)")
        
        df_res = pd.DataFrame(res)
        c1, c2 = st.columns([3, 1])
        with c1: st.dataframe(df_res.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)
        with c2: st.bar_chart(df_res.set_index('Ticker')['Score'])

# --- TAB 2: TACTICAL SNIPER (Analysis & Charting v12.5) ---
with tab_sniper:
    df = analyze_supreme_logic(fetch_master_data(sel), ihsg)
    if not df.empty:
        c_val, s_val, atr = df['Close'].iloc[-1], df['Score'].iloc[-1], df['ATR'].iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"Rp {c_val:,.0f}")
        m2.metric("Apex Score", f"{s_val}%")
        m3.metric("Candle Pattern", detect_candle_patterns(df))
        m4.metric("VPA Status", df['VPA_Desc'].iloc[-1])

        l_col, r_col = st.columns([1.5, 2.5])
        with l_col:
            st.subheader("⚔️ Execution Plan")
            sl = int(c_val - (atr * 2.2)) # Safety SL buffer
            risk_amt = cap * (risk_pct / 100)
            lots = int(risk_amt / ((c_val - sl) * 100)) if (c_val-sl) > 0 else 0
            
            st.success(f"**ENTRY:** {int(c_val):,.0f}")
            st.error(f"**STOP LOSS:** {sl:,.0f}")
            st.info(f"**POSITION SIZE:** {lots} Lots")
            
            # AI Forecast (v10.8)
            try:
                m_prop = Prophet(daily_seasonality=True).fit(df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}))
                future = m_prop.predict(m_prop.make_future_dataframe(periods=2))
                st.write(f"🔮 **AI Forecast:** Rp {future['yhat'].iloc[-1]:,.0f}")
            except: pass

        with r_col:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df.index[-60:], open=df['Open'][-60:], high=df['High'][-60:], low=df['Low'][-60:], close=df['Close'][-60:], name="Price"), row=1, col=1)
            # Volume Profile Overlay (v12.0)
            counts, bins = np.histogram(df['Close'][-60:], bins=10, weights=df['Volume'][-60:])
            for i in range(len(counts)):
                w = (counts[i]/counts.max()) * 7
                fig.add_trace(go.Scatter(x=[df.index[-60], df.index[-60+int(w)]], y=[bins[i], bins[i]], line=dict(color='rgba(0,255,255,0.1)', width=5), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['MFI'][-60:], line=dict(color='cyan'), fill='tozeroy', name="MFI"), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: VALIDATOR (Backtest v10.8) ---
with tab_validate:
    st.subheader(f"📈 Strategic Edge: {sel}")
    wr, profit = run_backtest(df.iloc[-252:]) # Test 1 tahun terakhir
    v1, v2 = st.columns(2)
    v1.metric("Historical Win Rate", f"{wr:.1f}%")
    v2.metric("Cumulative Profit vs B&H", f"{profit:.2f}%")
    df_bt = df.iloc[-252:].copy()
    df_bt['Equity'] = (np.where(df_bt['Score'] >= 75, 1, 0).astype(float) * df_bt['Close'].pct_change().fillna(0) + 1).cumprod()
    st.line_chart(df_bt['Equity'])

# --- TAB 4: OMNI-INTEL (AI & News Context v10.8-v13.0) ---
with tab_intel:
    c_l, c_r = st.columns(2)
    with c_l:
        st.subheader("🧠 Gemini Oracle Analysis")
        if st.button("🔮 Generate Deep AI Insight"):
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            prompt = f"Analisis IDX:{sel}. Harga:{c_val}, Score:{s_val}%, Pattern:{detect_candle_patterns(df)}, VPA:{df['VPA_Desc'].iloc[-1]}. Beri instruksi trading dalam 3 poin."
            res = model.generate_content(prompt)
            st.success(res.text)
        
        st.divider()
        st.markdown("**Stockbit Post Generator:**")
        st.code(f"${sel} Analysis 🔱\nScore: {s_val}%\nVPA: {df['VPA_Desc'].iloc[-1]}\nPattern: {detect_candle_patterns(df)}\nPlan: Buy {int(c_val)}, SL {sl}\n#Jarvisv13")

    with c_r:
        st.subheader("📰 Live Market Context")
        news_feed = feedparser.parse(f"https://news.google.com/rss/search?q={sel}+stock+idx")
        for entry in news_feed.entries[:5]: st.write(f"- [{entry.title}]({entry.link})")
