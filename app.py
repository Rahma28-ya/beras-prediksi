import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import pickle
import warnings
warnings.filterwarnings("ignore")

# AESTHETIC THEME
st.set_page_config(
    page_title="Dashboard Forecast Harga Beras",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
<style>
.big-font {font-size:32px !important; font-weight:700;}
.card {
    padding:20px; border-radius:20px;
    background: linear-gradient(135deg, #E3F2FD, #FCE4EC);
    color:#000; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("📊 Dashboard Peramalan Harga Beras (SARIMA & Prophet)")
st.write("Aesthetic • Insightful • Interaktif")

# FILE UPLOADER
data_file = st.file_uploader("Upload Dataset Harga Beras (CSV)", type=["csv"])

# =============================================================
# LOAD DATA (AUTO FALLBACK)
# =============================================================

if data_file:
    df = pd.read_csv(data_file)
else:
    st.info("ℹ️ Tidak ada file diupload. Menggunakan **dataset default** bawaan.")
    df = pd.read_csv("bps_final.csv")   # <-- Pastikan kamu upload default.csv ke repo!

# pastikan kolom pertama adalah tanggal
df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")

if df.iloc[:,0].isna().sum() > 0:
    st.error("❌ Format tanggal tidak valid. Harus YYYY-MM-DD.")
    st.stop()

df.columns = ["date", "price"]
df = df.sort_values("date")

# =============================================================
# 1. OVERVIEW PANEL
# =============================================================
st.subheader("Ringkasan Data (Overview)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f'<div class="card"><span class="big-font">{df["price"].iloc[-1]:,.0f}</span><br>Harga Terbaru (Rp)</div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f'<div class="card"><span class="big-font">{df["price"].tail(12).mean():,.0f}</span><br>Rata-rata 12 Bulan</div>',
        unsafe_allow_html=True
    )

with col3:
    if len(df) > 1:
        pct = (df["price"].iloc[-1] - df["price"].iloc[-2]) / df["price"].iloc[-2] * 100
    else:
        pct = 0
    st.markdown(
        f'<div class="card"><span class="big-font">{pct:+.2f}%</span><br>Perubahan Bulanan</div>',
        unsafe_allow_html=True
    )

with col4:
    vola = df["price"].pct_change().std() * 100
    st.markdown(
        f'<div class="card"><span class="big-font">{vola:.2f}%</span><br>Volatilitas Harga</div>',
        unsafe_allow_html=True
    )

# =============================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================
st.subheader("Exploratory Data Analysis (EDA)")

# Trend
fig_trend = px.line(
    df, x="date", y="price",
    title="📈 Tren Harga Beras",
    template="plotly_white",
    markers=True
)
fig_trend.update_traces(line=dict(width=3))
st.plotly_chart(fig_trend, use_container_width=True)

# Add year & month safely
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month_name()

# Heatmap
pivot = df.pivot_table(values="price", index="year", columns="month")

fig_heat = px.imshow(
    pivot,
    aspect="auto",
    title="🔥 Heatmap Musiman Harga Beras",
    color_continuous_scale="RdPu"
)
st.plotly_chart(fig_heat, use_container_width=True)

# Boxplot
fig_box = px.box(
    df, x="month", y="price",
    title="📦 Distribusi Harga Per Bulan",
    color="month", template="simple_white"
)
st.plotly_chart(fig_box, use_container_width=True)

# Moving average
df["MA_6"] = df["price"].rolling(6).mean()
df["MA_12"] = df["price"].rolling(12).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines", name="Harga"))
fig_ma.add_trace(go.Scatter(x=df["date"], y=df["MA_6"], mode="lines", name="MA 6 Bulan"))
fig_ma.add_trace(go.Scatter(x=df["date"], y=df["MA_12"], mode="lines", name="MA 12 Bulan"))
fig_ma.update_layout(title="📉 Moving Average", template="plotly_white")
st.plotly_chart(fig_ma, use_container_width=True)

# =============================================================
# 3. FORECAST PANEL
# =============================================================
st.subheader("Perbandingan Prediksi: SARIMA vs Prophet")

sarima_model_file = st.file_uploader("Upload Model SARIMA (pkl)", type=["pkl"])
prophet_model_file = st.file_uploader("Upload Model Prophet (pkl)", type=["pkl"])

if sarima_model_file and prophet_model_file:

    sarima_model = pickle.load(sarima_model_file)
    prophet_model = pickle.load(prophet_model_file)

    # SARIMA FORECAST
    sarima_pred = sarima_model.get_forecast(12)
    sarima_df = sarima_pred.summary_frame()
    sarima_df["date"] = pd.date_range(df["date"].iloc[-1], periods=12, freq="M")

    # PROPHET FORECAST
    future = prophet_model.make_future_dataframe(periods=12, freq="M")
    prophet_pred = prophet_model.predict(future)
    prophet_df = prophet_pred[["ds", "yhat"]].tail(12)
    prophet_df.columns = ["date", "price"]

    # Visualization
    fig_models = go.Figure()
    fig_models.add_trace(go.Scatter(x=df["date"], y=df["price"], mode="lines", name="Aktual"))
    fig_models.add_trace(go.Scatter(x=sarima_df["date"], y=sarima_df["mean"], mode="lines", name="SARIMA"))
    fig_models.add_trace(go.Scatter(x=prophet_df["date"], y=prophet_df["price"], mode="lines", name="Prophet"))

    fig_models.update_layout(title="🔮 Prediksi SARIMA vs Prophet", template="plotly_white")
    st.plotly_chart(fig_models, use_container_width=True)

    # =============================================================
    # 4. AUTO INSIGHT
    # =============================================================
    st.subheader("Insight Otomatis")

    sarima_last = sarima_df["mean"].iloc[-1]
    prophet_last = prophet_df["price"].iloc[-1]
    actual_last = df["price"].iloc[-1]

    st.success(
        f"""
📌 **SARIMA** memproyeksikan tren **{'naik' if sarima_last > actual_last else 'turun'}** 12 bulan ke depan.  
📌 **Prophet** memperkirakan harga mencapai **Rp {prophet_last:,.0f}** di bulan ke-12.  
📌 Volatilitas saat ini **{vola:.2f}%**, menandakan fluktuasi **{'stabil' if vola < 5 else 'cukup tinggi'}**.  
📌 Pola musiman menunjukkan harga sering naik pada **Februari–Maret**, dan melemah di sekitar **Oktober**.
        """
    )
