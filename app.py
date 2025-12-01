import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Harga Beras Indonesia",
    page_icon="🌾",
    layout="wide"
)

# 2. SIDEBAR NAVIGASI
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dashboard", "Prediksi SARIMA", "Prediksi Prophet", "Tentang"],
        icons=["bar-chart", "graph-up", "graph-up-arrow", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# 3. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("bps_final.csv")
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.rename(columns={"tanggal": "ds", "harga": "y"})
    return df

df = load_data()

# 4. DASHBOARD UTAMA
if selected == "Dashboard":
    st.title("Dashboard Harga Beras Indonesia")

    # Card Metrics
    harga_terakhir = df["y"].iloc[-1]
    harga_awal = df["y"].iloc[-2]
    persen_perubahan = ((harga_terakhir - harga_awal) / harga_awal) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Harga Terbaru", f"Rp {harga_terakhir:,.0f}")
    col2.metric("Harga Sebelumnya", f"Rp {harga_awal:,.0f}")
    col3.metric("Perubahan (%)", f"{persen_perubahan:.2f}%")

    # Grafik Interaktif Harga
    fig = px.line(df, x="ds", y="y",
                  title="Trend Harga Beras dari Waktu ke Waktu",
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

# 5. PREDIKSI DENGAN SARIMA
elif selected == "Prediksi SARIMA":
    st.title("📈 Prediksi Harga Beras Menggunakan SARIMA")

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12)

    # Model SARIMA
    model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    pred = model_fit.forecast(steps=periode)

    # Tampilkan dataframe prediksi
    pred_df = pd.DataFrame({
        "Tanggal": pd.date_range(start=df["ds"].iloc[-1], periods=periode+1, freq="MS")[1:],
        "Prediksi Harga": pred
    })

    # Grafik
    fig2 = px.line(pred_df, x="Tanggal", y="Prediksi Harga",
                   title="Prediksi Harga Beras - SARIMA",
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(pred_df)

# 6. PREDIKSI DENGAN PROPHET
elif selected == "Prediksi Prophet":
    st.title("🔮 Prediksi Harga Beras Menggunakan Prophet")

    periode = st.slider("Pilih Periode Prediksi (bulan)", 1, 24, 12, key="p")

    # Prophet sudah menggunakan kolom ds & y
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periode, freq="MS")
    forecast = model.predict(future)

    # Ambil hanya bagian prediksi
    pred_prophet = forecast[["ds", "yhat"]].tail(periode)

    # Grafik
    fig3 = px.line(pred_prophet, x="ds", y="yhat",
                   title="Prediksi Harga Beras - Prophet",
                   markers=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(pred_prophet)

# 7. TENTANG APLIKASI
elif selected == "Tentang":
    st.title("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan untuk memprediksi harga beras di Indonesia 
    menggunakan dua model peramalan: **SARIMA** dan **Prophet**.  
    Data berasal dari **BPS (data bulanan)** dan telah melalui proses pembersihan.  

    Fitur aplikasi:
    - Dashboard trend interaktif
    - Prediksi SARIMA
    - Prediksi Prophet
    - Grafik interaktif Plotly
    - Tampilan modern dengan sidebar navigation
    """)
