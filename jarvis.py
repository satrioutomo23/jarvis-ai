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
st.set_page_config(page_title="Jarvis v13.1 Omni-Portfolio", layout="wide", page_icon="🔱")
st_autorefresh(interval=300000, key="jarvis_heartbeat")

if "GEMINI_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_KEY"])
else:
    st.error("❌ API Key Gemini Missing! Masukkan GEMINI_KEY di Secrets.")
    st.stop()

# Inisialisasi Database Portfolio di Session State
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

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
    df = df.copy()
    if 'Score' not in df.columns: return 0, 0
    df['Signal'] = np.where(df['Score'] >= 75, 1, 0)
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
    valid_trades = (df['Strategy_Returns'] != 0).sum()
    win_rate = (df['Strategy_Returns'] > 0).sum() / valid_trades * 100 if valid_trades > 0 else 0
    cum_profit = (df['Strategy_Returns'] + 1).prod() - 1
    return win_rate, cum_profit * 100

def detect_candle_patterns(df):
    patterns = []
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
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    
    tp_idx = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp_idx * df['Volume']
    pos_mf = mf.where(tp_idx.diff() > 0, 0).rolling(14).sum()
    neg_mf = mf.where(tp_idx.diff() < 0, 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + (pos_mf / neg_mf)))
    
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['VPA_Desc'] = "Sideways"
    df.loc[(df['Vol_Ratio'] > 1.5) & (df['Close'].pct_change() > 0.02), 'VPA_Desc'] = "Accumulation"
    df.loc[(df['Vol_Ratio'] > 1.8) & (df['Close'].pct_change().abs() < 0.005), 'VPA_Desc'] = "🚨 BULL TRAP"
    
    if ihsg_df is not None and not ihsg_df.empty:
        ih_c = ihsg_df['Close'].reindex(df.index, method='ffill')
        raw_rs = (df['Close'] / df['Close'].iloc[0]) / (ih_c / ih_c.iloc[0])
        df['RS'] = raw_rs.rolling(5).mean()
    else: df['RS'] = 1.0
    
    df['Score'] = ((df['MFI'] > 55).astype(int) * 25) + \
                  ((df['Close'] > df['MA20']).astype(int) * 25) + \
                  ((df['RS'] > df['RS'].shift(1)).astype(int) * 25) + \
                  ((df['MA5'] > df['MA20']).astype(int) * 25)
    return df.dropna()

# =========================================================
# 3. INTERFACE
# =========================================================
watchlist = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", "ASII.JK", "ADRO.JK", "GOTO.JK", "ANTM.JK", "PTBA.JK", "MEDC.JK", "BRIS.JK", "TPIA.JK"]
st.title("🔱 Jarvis v13.1 Omni-Portfolio")

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

df_main = analyze_supreme_logic(fetch_master_data(sel), ihsg)

if not df_main.empty:
    st.session_state.current_ticker = sel
    st.session_state.c_val = df_main['Close'].iloc[-1]
    st.session_state.atr = df_main['ATR'].iloc[-1]
    st.session_state.tp_1 = int(st.session_state.c_val + (st.session_state.atr * 2))
    st.session_state.sl = int(st.session_state.c_val - (st.session_state.atr * 2))

tab_radar, tab_sniper, tab_validate, tab_portfolio, tab_oracle = st.tabs(["🚀 GLOBAL RADAR", "🎯 TACTICAL SNIPER", "📈 VALIDATOR", "💼 PORTFOLIO", "🧠 OMNI-INTEL"])

with tab_radar:
    if st.button("🛰️ EXECUTE SUPREME SCAN"):
        res = []
        for t in watchlist:
            d = analyze_supreme_logic(fetch_master_data(t), ihsg)
            if not d.empty:
                cv = d['Close'].iloc[-1]
                av = d['ATR'].iloc[-1]
                res.append({
                    "Ticker": t, "Price": f"{cv:,.0f}", "Score": d['Score'].iloc[-1],
                    "TP1": f"{int(cv + (av * 2)):,.0f}", "VPA": d['VPA_Desc'].iloc[-1]
                })
        st.dataframe(pd.DataFrame(res).style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)

with tab_sniper:
    if not df_main.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Price", f"Rp {st.session_state.c_val:,.0f}")
        m2.metric("Apex Score", f"{df_main['Score'].iloc[-1]}%")
        m3.metric("Pattern", detect_candle_patterns(df_main))
        m4.metric("VPA Status", df_main['VPA_Desc'].iloc[-1])

        l_col, r_col = st.columns([1.5, 2.5])
        with l_col:
            st.subheader("⚔️ Trading Plan")
            st.write(f"**Entry:** {int(st.session_state.c_val):,.0f}")
            st.error(f"**Stop Loss:** {st.session_state.sl:,.0f}")
            st.success(f"**TP 1:** {st.session_state.tp_1:,.0f}")
            risk_amt = cap * (risk_pct / 100)
            risk_ps = st.session_state.c_val - st.session_state.sl
            lots = int(risk_amt / (risk_ps * 100)) if risk_ps > 0 else 0
            st.info(f"**Size Recommendation:** {lots} Lots")

        with r_col:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_main.index[-60:], open=df_main['Open'][-60:], high=df_main['High'][-60:], low=df_main['Low'][-60:], close=df_main['Close'][-60:], name="Price"), row=1, col=1)
            fig.add_hline(y=st.session_state.tp_1, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=st.session_state.sl, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_trace(go.Scatter(x=df_main.index[-60:], y=df_main['MFI'][-60:], line=dict(color='cyan'), fill='tozeroy', name="MFI"), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

with tab_validate:
    if not df_main.empty:
        st.subheader(f"📈 Strategic Validation: {sel}")
        wr, profit = run_backtest(df_main.iloc[-252:])
        st.metric("Win Rate (1Y)", f"{wr:.1f}%")
        st.line_chart(df_main.iloc[-252:]['Close'])

with tab_portfolio:
    st.subheader("📊 My Active Positions")
    if not st.session_state.portfolio:
        st.info("Portfolio kosong. Tambahkan posisi melalui sidebar.")
    else:
        p_data = []
        total_pnl = 0
        for tick, info in st.session_state.portfolio.items():
            # Ambil harga terbaru untuk hitung PnL
            current_df = fetch_master_data(tick)
            if not current_df.empty:
                current_p = current_df['Close'].iloc[-1]
                buy_p = info['price']
                lots = info['lots']
                value_now = current_p * lots * 100
                value_buy = buy_p * lots * 100
                pnl = value_now - value_buy
                pnl_pct = (pnl / value_buy) * 100
                total_pnl += pnl
                
                p_data.append({
                    "Ticker": tick,
                    "Avg Price": f"{buy_p:,.0f}",
                    "Current Price": f"{current_p:,.0f}",
                    "Lots": lots,
                    "PnL (Rp)": f"{pnl:,.0f}",
                    "PnL (%)": f"{pnl_pct:.2f}%"
                })
        
        st.table(pd.DataFrame(p_data))
        st.metric("Total Unrealized PnL", f"Rp {total_pnl:,.0f}", delta=f"{total_pnl:,.0f}")

with tab_oracle:
    st.subheader("🧠 Gemini Oracle Analysis")
    if st.button("🔮 Run Deep Analysis"):
        with st.spinner("Menghubungi Oracle..."):
            models = ['gemini-1.5-flash', 'gemini-pro']
            success = False
            for m_name in models:
                try:
                    model = genai.GenerativeModel(m_name)
                    prompt = f"Analisis teknikal IDX:{st.session_state.current_ticker}. Harga:{st.session_state.c_val}, TP:{st.session_state.tp_1}, SL:{st.session_state.sl}. Berikan strategi singkat dalam Bahasa Indonesia."
                    res = model.generate_content(prompt)
                    st.success(f"Analisis Berhasil (Model: {m_name})")
                    st.markdown(res.text)
                    success = True
                    break
                except Exception as e:
                    continue
            if not success:
                st.error("Gagal terhubung ke AI. Cek kuota atau koneksi internet.")
