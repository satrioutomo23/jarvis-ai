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
st.set_page_config(page_title="Jarvis v14.0 High-Precision", layout="wide", page_icon="🔱")
st_autorefresh(interval=300000, key="jarvis_heartbeat")

if "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
else:
    st.error("❌ API Key Gemini Missing! Masukkan GEMINI_KEY di Secrets.")
    st.stop()

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# =========================================================
# 2. THE BRAIN: ENHANCED ANALYTICS
# =========================================================
@st.cache_data(ttl=300)
def fetch_master_data(ticker, period="2y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return pd.DataFrame()

def detect_candle_patterns(df):
    patterns = []
    if len(df) < 2: return "Neutral"
    last, prev = df.iloc[-1], df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    range_total = last['High'] - last['Low']
    if range_total == 0: return "Neutral"
    
    if body < (range_total * 0.3) and (last['Low'] < min(last['Open'], last['Close']) - body):
        patterns.append("🔨 Hammer")
    if last['Close'] > prev['Open'] and last['Open'] < prev['Close'] and prev['Close'] < prev['Open']:
        patterns.append("🔥 Engulfing")
    if body < (range_total * 0.1):
        patterns.append("⚖️ Doji")
    return ", ".join(patterns) if patterns else "Neutral"

def analyze_supreme_logic(df, ihsg_df=None):
    if df.empty or len(df) < 50: return pd.DataFrame()
    df = df.copy()
    
    # --- Precision Indicators ---
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # Bollinger Bands untuk SL Presisi
    df['Std_Dev'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (df['Std_Dev'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['Std_Dev'] * 2)
    
    # Money Flow Index
    tp_idx = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp_idx * df['Volume']
    pos_mf = mf.where(tp_idx.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp_idx.diff() < 0, 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / neg_mf)))
    
    # Volume Price Analysis (VPA)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['VPA_Desc'] = "Sideways"
    df.loc[(df['Vol_Ratio'] > 1.5) & (df['Close'].pct_change() > 0.02), 'VPA_Desc'] = "Accumulation"
    df.loc[(df['Vol_Ratio'] > 1.8) & (df['Close'].pct_change().abs() < 0.005), 'VPA_Desc'] = "🚨 BULL TRAP"
    
    # Relative Strength (RS)
    if ihsg_df is not None and not ihsg_df.empty:
        ih_c = ihsg_df['Close'].reindex(df.index, method='ffill')
        df['RS'] = (df['Close'] / df['Close'].iloc[0]) / (ih_c / ih_c.iloc[0])
    else: df['RS'] = 1.0
    
    # Scoring System v14 (Weighted)
    pattern = detect_candle_patterns(df)
    pattern_boost = 15 if any(p in pattern for p in ["Hammer", "Engulfing"]) else 0
    
    df['Score'] = ((df['MFI'] > 55).astype(int) * 20) + \
                  ((df['Close'] > df['MA20']).astype(int) * 25) + \
                  ((df['RS'] > df['RS'].shift(1)).astype(int) * 20) + \
                  ((df['MA5'] > df['MA20']).astype(int) * 20) + pattern_boost
    return df.dropna()

# =========================================================
# 3. INTERFACE & LOGIC SYNERGY
# =========================================================
watchlist = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "ADRO.JK", "GOTO.JK", "ANTM.JK", "PTBA.JK", "MEDC.JK", "BRIS.JK", "TPIA.JK"]
st.title("🔱 Jarvis v14.0 High-Precision")

ihsg = fetch_master_data("^JKSE")

with st.sidebar:
    st.header("🛡️ Tactical Shield")
    cap = st.number_input("Modal Capital (Rp)", value=10000000, step=1000000)
    risk_pct = st.select_slider("Risk per Trade (%)", options=[0.5, 1.0, 2.0], value=1.0)
    sel = st.selectbox("🎯 Target Select", watchlist)
    
    st.divider()
    st.subheader("📝 Log Position")
    p_price = st.number_input("Entry Price", value=0)
    p_lot = st.number_input("Lots", value=0)
    if st.button("➕ Add to Portfolio"):
        if p_price > 0 and p_lot > 0:
            st.session_state.portfolio[sel] = {"price": p_price, "lots": p_lot}
            st.success(f"Saved {sel}!")
    if st.button("🗑️ Reset Portfolio"):
        st.session_state.portfolio = {}
        st.rerun()

# --- Multi-Timeframe Filter ---
df_daily = analyze_supreme_logic(fetch_master_data(sel), ihsg)
df_weekly = fetch_master_data(sel, period="2y", interval="1wk")
weekly_trend = "BULLISH" if not df_weekly.empty and df_weekly['Close'].iloc[-1] > df_weekly['Close'].rolling(20).mean().iloc[-1] else "BEARISH"

if not df_daily.empty:
    c_val = df_daily['Close'].iloc[-1]
    atr = df_daily['ATR'].iloc[-1]
    bb_low = df_daily['BB_Lower'].iloc[-1]
    
    # Presisi Entry & SL
    # Entry ideal di area MA20 atau harga saat ini jika MA5 > MA20
    # SL menggunakan Bollinger Band Lower agar tidak mudah 'tercolek'
    st.session_state.current_ticker = sel
    st.session_state.c_val = c_val
    st.session_state.sl = int(min(bb_low, c_val - (atr * 2)))
    st.session_state.tp_1 = int(c_val + (atr * 2.5)) # TP lebih lebar sedikit untuk RRR bagus
    
    risk_ps = c_val - st.session_state.sl
    reward_ps = st.session_state.tp_1 - c_val
    rrr = reward_ps / risk_ps if risk_ps > 0 else 0

tab_radar, tab_sniper, tab_validate, tab_portfolio, tab_oracle = st.tabs(["🚀 GLOBAL RADAR", "🎯 TACTICAL SNIPER", "📈 VALIDATOR", "💼 PORTFOLIO", "🧠 OMNI-INTEL"])

with tab_radar:
    if st.button("🛰️ EXECUTE SUPREME SCAN"):
        res = []
        for t in watchlist:
            d = analyze_supreme_logic(fetch_master_data(t), ihsg)
            if not d.empty:
                cv = d['Close'].iloc[-1]
                res.append({
                    "Ticker": t, "Price": f"{cv:,.0f}", "Score": d['Score'].iloc[-1],
                    "VPA": d['VPA_Desc'].iloc[-1], "Pattern": detect_candle_patterns(d)
                })
        st.dataframe(pd.DataFrame(res).style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)

with tab_sniper:
    if not df_daily.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"Rp {c_val:,.0f}")
        m2.metric("Apex Score", f"{df_daily['Score'].iloc[-1]}%")
        m3.metric("Weekly Trend", weekly_trend, delta="Confirm" if weekly_trend == "BULLISH" else "Caution")
        m4.metric("Risk:Reward", f"1 : {rrr:.2f}")

        l_col, r_col = st.columns([1.5, 2.5])
        with l_col:
            st.subheader("⚔️ Tactical Entry")
            if weekly_trend == "BEARISH":
                st.warning("⚠️ Tren besar (Weekly) sedang turun. Entry sangat berisiko!")
            
            st.write(f"**Entry Point:** {int(c_val):,.0f}")
            st.error(f"**Safe Stop Loss:** {st.session_state.sl:,.0f}")
            st.success(f"**Target Profit:** {st.session_state.tp_1:,.0f}")
            
            risk_amt = cap * (risk_pct / 100)
            lots_rec = int(risk_amt / (risk_ps * 100)) if risk_ps > 0 else 0
            st.info(f"**Size Recommendation:** {lots_rec} Lots")
            
            if rrr < 1.5:
                st.error("❌ RRR Terlalu Rendah. Tidak disarankan Entry.")
            else:
                st.success("✅ RRR Memadai. Siap Eksekusi.")

        with r_col:
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            # Candlestick & Bollinger
            fig.add_trace(go.Candlestick(x=df_daily.index[-60:], open=df_daily['Open'][-60:], high=df_daily['High'][-60:], low=df_daily['Low'][-60:], close=df_daily['Close'][-60:], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_daily.index[-60:], y=df_daily['BB_Upper'][-60:], line=dict(color='gray', width=1), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_daily.index[-60:], y=df_daily['BB_Lower'][-60:], line=dict(color='gray', width=1), name="BB Lower"), row=1, col=1)
            
            fig.add_hline(y=st.session_state.tp_1, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=st.session_state.sl, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_trace(go.Scatter(x=df_daily.index[-60:], y=df_daily['MFI'][-60:], line=dict(color='cyan'), fill='tozeroy', name="MFI Flow"), row=2, col=1)
            fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

# (Tab Validate, Portfolio, & Oracle disinergikan dengan data di atas)
with tab_portfolio:
    st.subheader("💼 Active Portfolio Monitoring")
    if not st.session_state.portfolio:
        st.info("Portfolio kosong.")
    else:
        # Logika Portfolio tetap menggunakan perhitungan Net PnL Stockbit yang sudah kita bahas sebelumnya
        p_data = []
        total_pnl = 0
        F_BELI, F_JUAL = 0.0015, 0.0025
        for tick, info in st.session_state.portfolio.items():
            curr_d = fetch_master_data(tick)
            if not curr_d.empty:
                cp = float(curr_d['Close'].iloc[-1])
                bp, ls = float(info['price']), int(info['lots'])
                cost = (bp * ls * 100) * (1 + F_BELI)
                receive = (cp * ls * 100) * (1 - F_JUAL)
                pnl = receive - cost
                total_pnl += pnl
                p_data.append({"Ticker": tick, "Avg": f"{bp:,.0f}", "Current": f"{cp:,.0f}", "Lots": ls, "Net PnL": f"{pnl:,.0f}", "%": f"{(pnl/cost)*100:.2f}%"})
        st.table(pd.DataFrame(p_data))
        st.metric("Total Net PnL", f"Rp {total_pnl:,.0f}")

with tab_oracle:
    if st.button("🔮 Oracle Deep Analysis"):
        with st.spinner("Analisis Presisi Tinggi Sedang Berjalan..."):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""Lakukan analisis SWOT teknikal untuk {sel}. 
                Harga saat ini {c_val}, tren mingguan {weekly_trend}, Score Jarvis {df_daily['Score'].iloc[-1]}%. 
                Strategi Entry: {c_val}, TP: {st.session_state.tp_1}, SL: {st.session_state.sl}.
                Beri instruksi spesifik apakah boleh beli hari ini dan alasannya dalam Bahasa Indonesia."""
                res = model.generate_content(prompt)
                st.markdown(res.text)
            except: st.error("Koneksi Oracle Terputus.")
