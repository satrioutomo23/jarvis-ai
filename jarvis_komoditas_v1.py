import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
from plotly.subplots import make_subplots

# =========================================================
# 1. CORE SYSTEM & CONFIG (JARVIS COMMODITY ENGINE)
# =========================================================
st.set_page_config(page_title="Jarvis v1.0y - Oil Commander", layout="wide", page_icon="🛢️")

if "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
else:
    st.error("❌ API Key Missing!")
    st.stop()

# =========================================================
# 2. COMMODITY BRAIN (WTI LOGIC)
# =========================================================
@st.cache_data(ttl=60) # Update lebih cepat untuk komoditas
def fetch_oil_data(ticker="CL=F"): # WTI Crude Oil Futures
    try:
        # Mengambil data intraday 1 menit / 5 menit untuk akurasi entry
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        return data.dropna()
    except: return pd.DataFrame()

def analyze_commodity_v14(df):
    if df.empty or len(df) < 30: return pd.DataFrame()
    df = df.copy()
    
    # Indikator Khusus Oil: RSI & EMA Cross
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    
    # ATR untuk Stop Loss Komoditas (Oil sangat volatil)
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    # RSI (Melihat jenuh beli/jual di Oil)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Scoring Logic v1.0y (Fokus pada Momentum)
    # 1. EMA Cross (40%)
    # 2. RSI Zone (30%)
    # 3. Trend Alignment (30%)
    df['Score'] = ((df['EMA9'] > df['EMA21']).astype(int) * 40) + \
                  (((df['RSI'] > 40) & (df['RSI'] < 70)).astype(int) * 30) + \
                  ((df['Close'] > df['EMA9']).astype(int) * 30)
    
    return df.dropna()

# =========================================================
# 3. INTERFACE COMMAND CENTER
# =========================================================
st.title("🛢️ Jarvis_Komoditas_v1.0y")
st.caption("Strategic Intelligence for Crude Oil (WTI) Trading")

# Sidebar - Risk Management MIFX Style
with st.sidebar:
    st.header("⚙️ MT5 Demo Config")
    balance = st.number_input("Demo Balance ($)", value=10000)
    lot_size = st.number_input("Lot Size", value=0.1, step=0.01)
    st.divider()
    oil_ticker = st.selectbox("Commodity Select", ["CL=F (WTI Oil)", "BZ=F (Brent Oil)", "NG=F (Natural Gas)"])
    ticker_symbol = oil_ticker.split(" ")[0]

# Execution
df_oil = analyze_commodity_v14(fetch_oil_data(ticker_symbol))

if not df_oil.empty:
    curr_price = df_oil['Close'].iloc[-1]
    curr_score = df_oil['Score'].iloc[-1]
    atr = df_oil['ATR'].iloc[-1]
    
    # TACTICAL PLANNING (ENTRY/TP/SL)
    # Oil biasanya butuh SL sekitar 30-50 pips (0.30 - 0.50)
    sl_dist = atr * 2.5
    tp_dist = atr * 4.0
    
    # Tabs
    tab_chart, tab_intel = st.tabs(["📊 LIVE MONITOR", "🧠 OMNI-INTEL"])
    
    with tab_chart:
        # Header Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("WTI Price", f"${curr_price:.2f}")
        status = "BUY" if curr_score > 60 else "SELL" if curr_score < 40 else "WAIT"
        c2.metric("Tactical Signal", status, f"{curr_score}%")
        c3.metric("Volatility (ATR)", f"{atr:.3f}")
        
        # Trading Plan Box
        st.subheader("⚔️ MT5 Execution Plan")
        p1, p2, p3 = st.columns(3)
        if status == "BUY":
            p1.info(f"**ENTRY (BUY):** {curr_price:.2f}")
            p2.error(f"**STOP LOSS:** {(curr_price - sl_dist):.2f}")
            p3.success(f"**TAKE PROFIT:** {(curr_price + tp_dist):.2f}")
        else:
            p1.info(f"**ENTRY (SELL):** {curr_price:.2f}")
            p2.error(f"**STOP LOSS:** {(curr_price + sl_dist):.2f}")
            p3.success(f"**TAKE PROFIT:** {(curr_price - tp_dist):.2f}")

        # Chart Visual
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        # Candlestick
        fig.add_trace(go.Candlestick(x=df_oil.index, open=df_oil['Open'], high=df_oil['High'], low=df_oil['Low'], close=df_oil['Close'], name="Oil Price"), row=1, col=1)
        # EMA
        fig.add_trace(go.Scatter(x=df_oil.index, y=df_oil['EMA9'], name="EMA 9 (Fast)", line=dict(color='cyan', width=1)), row=1, col=1)
        # RSI
        fig.add_trace(go.Scatter(x=df_oil.index, y=df_oil['RSI'], name="RSI", line=dict(color='yellow')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab_intel:
        if st.button("🔮 Consult Gemini Oracle"):
            with st.spinner("Analyzing Global Energy Data..."):
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"""
                Analisis teknikal Oil (WTI) saat ini di harga {curr_price}.
                Indikator Score: {curr_score}% (EMA Cross & RSI).
                Beri instruksi singkat untuk trading 0.1 lot di MT5. 
                Sebutkan potensi arah pergerakan berdasarkan struktur chart saat ini dalam Bahasa Indonesia.
                """
                response = model.generate_content(prompt)
                st.markdown(response.text)
